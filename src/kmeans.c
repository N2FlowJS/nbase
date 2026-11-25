#include "nbase_internal.h"

#include <float.h>
#include <math.h>
#include <pthread.h>
#include <string.h>

typedef struct kmeans_pool_ctx {
  const float *vectors;
  const float *centroids;
  uint32_t count;
  uint32_t dim;
  uint32_t k;
  uint32_t *next_assign;
  float *point_err;
  float *dist_bufs;
  nbase_status_t *statuses;
  pthread_mutex_t mu;
  pthread_cond_t start_cv;
  pthread_cond_t done_cv;
  uint32_t epoch;
  uint32_t done_count;
  uint32_t total_workers;
  uint32_t spawned;
  bool stop;
} kmeans_pool_ctx_t;

typedef struct kmeans_worker_arg {
  kmeans_pool_ctx_t *ctx;
  uint32_t tid;
} kmeans_worker_arg_t;

static void kmeans_assign_slice(kmeans_pool_ctx_t *ctx, uint32_t tid) {
  uint32_t start = (uint32_t)(((uint64_t)tid * (uint64_t)ctx->count) / (uint64_t)ctx->total_workers);
  uint32_t end =
      (uint32_t)(((uint64_t)(tid + 1U) * (uint64_t)ctx->count) / (uint64_t)ctx->total_workers);
  float *dist_buf = ctx->dist_bufs + (size_t)tid * (size_t)ctx->k;
  nbase_status_t wst = NBASE_OK;

  for (uint32_t i = start; i < end; i++) {
    const float *vec = ctx->vectors + (size_t)i * (size_t)ctx->dim;
    wst = nbase_distance_batch(
        NBASE_METRIC_SQUARED_EUCLIDEAN, vec, ctx->centroids, ctx->k, ctx->dim, dist_buf);
    if (wst != NBASE_OK) {
      break;
    }
    uint32_t best_c = 0U;
    float best_d = FLT_MAX;
    for (uint32_t c = 0U; c < ctx->k; c++) {
      if (dist_buf[c] < best_d) {
        best_d = dist_buf[c];
        best_c = c;
      }
    }
    ctx->point_err[i] = best_d;
    ctx->next_assign[i] = best_c;
  }

  ctx->statuses[tid] = wst;
}

static void *kmeans_worker_main(void *raw) {
  kmeans_worker_arg_t *arg = (kmeans_worker_arg_t *)raw;
  kmeans_pool_ctx_t *ctx = arg->ctx;
  uint32_t tid = arg->tid;
  uint32_t seen_epoch = 0U;

  for (;;) {
    pthread_mutex_lock(&ctx->mu);
    while (!ctx->stop && ctx->epoch == seen_epoch) {
      pthread_cond_wait(&ctx->start_cv, &ctx->mu);
    }
    if (ctx->stop) {
      pthread_mutex_unlock(&ctx->mu);
      break;
    }
    seen_epoch = ctx->epoch;
    pthread_mutex_unlock(&ctx->mu);

    kmeans_assign_slice(ctx, tid);

    pthread_mutex_lock(&ctx->mu);
    ctx->done_count++;
    if (ctx->done_count == ctx->spawned) {
      pthread_cond_signal(&ctx->done_cv);
    }
    pthread_mutex_unlock(&ctx->mu);
  }
  return NULL;
}

nbase_status_t nbase_kmeans_train(
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    uint32_t k,
    uint32_t max_iters,
    uint64_t seed,
    float *out_centroids,
    uint32_t *out_assignments) {
  return nbase_kmeans_train_mt(
      vectors, count, dim, k, max_iters, seed, 0U, out_centroids, out_assignments);
}

nbase_status_t nbase_kmeans_train_mt(
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    uint32_t k,
    uint32_t max_iters,
    uint64_t seed,
    uint32_t num_threads,
    float *out_centroids,
    uint32_t *out_assignments) {
  if (!vectors || !out_centroids || count == 0U || dim == 0U || dim > NBASE_MAX_DIM || k == 0U || k > count) {
    return NBASE_ERR_INVALID_ARG;
  }
  if ((size_t)count > SIZE_MAX / (size_t)dim || (size_t)k > SIZE_MAX / (size_t)dim) {
    return NBASE_ERR_NOMEM;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)dim, dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  uint64_t rng = (seed != 0ULL) ? seed : 42ULL;

  float *min_dist_sq = malloc((size_t)count * sizeof(float));
  float *dist_buf = malloc((size_t)count * sizeof(float));
  uint32_t *assign = malloc((size_t)count * sizeof(uint32_t));
  float *point_err = malloc((size_t)count * sizeof(float));
  double *sums = malloc((size_t)k * (size_t)dim * sizeof(double));
  uint32_t *counts = malloc((size_t)k * sizeof(uint32_t));

  if (!min_dist_sq || !dist_buf || !assign || !point_err || !sums || !counts) {
    free(min_dist_sq);
    free(dist_buf);
    free(assign);
    free(point_err);
    free(sums);
    free(counts);
    return NBASE_ERR_NOMEM;
  }

  /* K-Means++ initialization */
  uint32_t first = (uint32_t)(nbase_rng_next_u64(&rng) % (uint64_t)count);
  memcpy(out_centroids, vectors + (size_t)first * (size_t)dim, (size_t)dim * sizeof(float));
  nbase_status_t st = nbase_distance_batch(
      NBASE_METRIC_SQUARED_EUCLIDEAN, out_centroids, vectors, count, dim, min_dist_sq);
  if (st != NBASE_OK) {
    free(min_dist_sq);
    free(dist_buf);
    free(assign);
    free(point_err);
    free(sums);
    free(counts);
    return st;
  }

  for (uint32_t c = 1U; c < k; c++) {
    double total = 0.0;
    for (uint32_t i = 0U; i < count; i++) {
      total += (double)min_dist_sq[i];
    }

    uint32_t chosen = count - 1U;
    if (total <= 0.0) {
      chosen = (uint32_t)(nbase_rng_next_u64(&rng) % (uint64_t)count);
    } else {
      double target = nbase_rng_next_double(&rng) * total;
      double acc = 0.0;
      for (uint32_t i = 0U; i < count; i++) {
        acc += (double)min_dist_sq[i];
        if (acc >= target) {
          chosen = i;
          break;
        }
      }
    }

    float *dst_centroid = out_centroids + (size_t)c * (size_t)dim;
    memcpy(dst_centroid, vectors + (size_t)chosen * (size_t)dim, (size_t)dim * sizeof(float));

    st = nbase_distance_batch(
        NBASE_METRIC_SQUARED_EUCLIDEAN, dst_centroid, vectors, count, dim, dist_buf);
    if (st != NBASE_OK) {
      free(min_dist_sq);
      free(dist_buf);
      free(assign);
      free(point_err);
      free(sums);
      free(counts);
      return st;
    }
    for (uint32_t i = 0U; i < count; i++) {
      if (dist_buf[i] < min_dist_sq[i]) {
        min_dist_sq[i] = dist_buf[i];
      }
    }
  }

  for (uint32_t i = 0U; i < count; i++) {
    assign[i] = UINT32_MAX;
  }

  uint32_t eff_threads = 1U;
  if (num_threads != 1U && count >= 64U && k >= 2U &&
      (uint64_t)count * (uint64_t)k * (uint64_t)dim >= 262144ULL) {
    eff_threads = (num_threads == 0U) ? 4U : num_threads;
    uint32_t max_by_count = count / 16U;
    if (max_by_count == 0U) {
      max_by_count = 1U;
    }
    if (eff_threads > max_by_count) {
      eff_threads = max_by_count;
    }
    if (eff_threads > 64U) {
      eff_threads = 64U;
    }
  }

  float *worker_dists = malloc((size_t)eff_threads * (size_t)k * sizeof(float));
  uint32_t *next_assign = malloc((size_t)count * sizeof(uint32_t));
  nbase_status_t *worker_statuses = malloc((size_t)eff_threads * sizeof(nbase_status_t));
  if (!worker_dists || !next_assign || !worker_statuses) {
    free(worker_dists);
    free(next_assign);
    free(worker_statuses);
    free(min_dist_sq);
    free(dist_buf);
    free(assign);
    free(point_err);
    free(sums);
    free(counts);
    return NBASE_ERR_NOMEM;
  }

  kmeans_pool_ctx_t pool;
  memset(&pool, 0, sizeof(pool));
  pool.vectors = vectors;
  pool.centroids = out_centroids;
  pool.count = count;
  pool.dim = dim;
  pool.k = k;
  pool.next_assign = next_assign;
  pool.point_err = point_err;
  pool.dist_bufs = worker_dists;
  pool.statuses = worker_statuses;

  pthread_t tids[63];
  kmeans_worker_arg_t wargs[63];
  bool pool_sync_init = false;
  if (eff_threads > 1U) {
    if (pthread_mutex_init(&pool.mu, NULL) == 0) {
      if (pthread_cond_init(&pool.start_cv, NULL) == 0 &&
          pthread_cond_init(&pool.done_cv, NULL) == 0) {
        pool_sync_init = true;
        for (uint32_t w = 1U; w < eff_threads; w++) {
          wargs[w - 1U].ctx = &pool;
          wargs[w - 1U].tid = w;
          if (pthread_create(&tids[w - 1U], NULL, kmeans_worker_main, &wargs[w - 1U]) != 0) {
            break;
          }
          pool.spawned++;
        }
      } else {
        pthread_cond_destroy(&pool.start_cv);
        pthread_mutex_destroy(&pool.mu);
      }
    }
  }
  pool.total_workers = pool.spawned + 1U;

  uint32_t iters = (max_iters > 0U) ? max_iters : 25U;
  for (uint32_t iter = 0U; iter < iters; iter++) {
    uint32_t changed = 0U;
    memset(sums, 0, (size_t)k * (size_t)dim * sizeof(double));
    memset(counts, 0, (size_t)k * sizeof(uint32_t));

    if (pool.spawned > 0U) {
      pthread_mutex_lock(&pool.mu);
      pool.done_count = 0U;
      pool.epoch = iter + 1U;
      pthread_cond_broadcast(&pool.start_cv);
      pthread_mutex_unlock(&pool.mu);

      kmeans_assign_slice(&pool, 0U);

      pthread_mutex_lock(&pool.mu);
      while (pool.done_count < pool.spawned) {
        pthread_cond_wait(&pool.done_cv, &pool.mu);
      }
      pthread_mutex_unlock(&pool.mu);
    } else {
      kmeans_assign_slice(&pool, 0U);
    }

    nbase_status_t iter_st = NBASE_OK;
    for (uint32_t w = 0U; w < pool.total_workers; w++) {
      if (worker_statuses[w] != NBASE_OK) {
        iter_st = worker_statuses[w];
        break;
      }
    }
    if (iter_st != NBASE_OK) {
      st = iter_st;
      break;
    }

    for (uint32_t i = 0U; i < count; i++) {
      uint32_t best_c = next_assign[i];
      if (assign[i] != best_c) {
        assign[i] = best_c;
        changed++;
      }
      counts[best_c]++;
      const float *vec = vectors + (size_t)i * (size_t)dim;
      double *csum = sums + (size_t)best_c * (size_t)dim;
      for (uint32_t d = 0U; d < dim; d++) {
        csum[d] += (double)vec[d];
      }
    }

    /* Recover any empty cluster by stealing the farthest point from a multi-member cluster. */
    for (uint32_t c = 0U; c < k; c++) {
      if (counts[c] > 0U) {
        continue;
      }
      uint32_t far_i = UINT32_MAX;
      float max_err = -1.0f;
      for (uint32_t i = 0U; i < count; i++) {
        uint32_t owner = assign[i];
        if (counts[owner] > 1U && point_err[i] > max_err) {
          max_err = point_err[i];
          far_i = i;
        }
      }
      if (far_i == UINT32_MAX) {
        far_i = c % count;
      }
      uint32_t old_c = assign[far_i];
      const float *vec = vectors + (size_t)far_i * (size_t)dim;
      double *old_sum = sums + (size_t)old_c * (size_t)dim;
      double *new_sum = sums + (size_t)c * (size_t)dim;
      for (uint32_t d = 0U; d < dim; d++) {
        old_sum[d] -= (double)vec[d];
        new_sum[d] += (double)vec[d];
      }
      counts[old_c]--;
      counts[c]++;
      assign[far_i] = c;
      point_err[far_i] = 0.0f;
      changed++;
    }

    double max_shift_sq = 0.0;
    for (uint32_t c = 0U; c < k; c++) {
      float *dst = out_centroids + (size_t)c * (size_t)dim;
      const double *csum = sums + (size_t)c * (size_t)dim;
      double inv = 1.0 / (double)counts[c];
      double shift_sq = 0.0;
      for (uint32_t d = 0U; d < dim; d++) {
        float next_v = (float)(csum[d] * inv);
        double delta = (double)next_v - (double)dst[d];
        shift_sq += delta * delta;
        dst[d] = next_v;
      }
      if (shift_sq > max_shift_sq) {
        max_shift_sq = shift_sq;
      }
    }

    if (changed == 0U || max_shift_sq < 1e-12) {
      break;
    }
  }

  if (pool_sync_init) {
    pthread_mutex_lock(&pool.mu);
    pool.stop = true;
    pthread_cond_broadcast(&pool.start_cv);
    pthread_mutex_unlock(&pool.mu);
    for (uint32_t t = 0U; t < pool.spawned; t++) {
      pthread_join(tids[t], NULL);
    }
    pthread_cond_destroy(&pool.start_cv);
    pthread_cond_destroy(&pool.done_cv);
    pthread_mutex_destroy(&pool.mu);
  }

  if (st == NBASE_OK && out_assignments != NULL) {
    memcpy(out_assignments, assign, (size_t)count * sizeof(uint32_t));
  }

  free(worker_dists);
  free(next_assign);
  free(worker_statuses);
  free(min_dist_sq);
  free(dist_buf);
  free(assign);
  free(point_err);
  free(sums);
  free(counts);
  return st;
}
