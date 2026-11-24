#include "test_common.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

static bool ivfpq_even_filter(uint64_t id, void *ctx) {
  (void)ctx;
  return (id % 2ULL) == 0ULL;
}

static void make_clustered_vectors(
    float *out,
    uint32_t clusters,
    uint32_t per_cluster,
    uint32_t dim,
    uint64_t seed) {
  uint32_t total = clusters * per_cluster;
  test_fill_vectors(out, total, dim, seed);
  for (uint32_t c = 0U; c < clusters; c++) {
    for (uint32_t i = 0U; i < per_cluster; i++) {
      float *v = out + ((size_t)c * per_cluster + i) * dim;
      for (uint32_t d = 0U; d < dim; d++) {
        v[d] = v[d] * 0.1f + ((d == (c % dim)) ? 4.0f : 0.0f);
      }
    }
  }
}

static void test_ivfpq_train_add_search_euclidean_and_save_load(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  char path[300];
  snprintf(path, sizeof(path), "%s/index.nbiv", dir);

  const uint32_t dim = 32U;
  const uint32_t clusters = 8U;
  const uint32_t per_cluster = 120U;
  const uint32_t n = clusters * per_cluster;

  float *vecs = malloc((size_t)n * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);

  make_clustered_vectors(vecs, clusters, per_cluster, dim, 1111ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = 1000ULL + i;
  }

  nbase_ivfpq_config_t cfg = nbase_ivfpq_config_default(dim, 8U, 8U);
  cfg.num_centroids = 64U;
  cfg.max_iters = 15U;
  cfg.seed = 2222ULL;

  nbase_ivfpq_t *idx = NULL;
  ASSERT_STATUS(nbase_ivfpq_train(&cfg, vecs, n, &idx), NBASE_OK);
  ASSERT_STATUS(nbase_ivfpq_add(idx, ids, vecs, n), NBASE_OK);

  nbase_ivfpq_stats_t st;
  ASSERT_STATUS(nbase_ivfpq_stats(idx, &st), NBASE_OK);
  ASSERT_EQ_U64(st.count, n);
  ASSERT_EQ_U64(st.nlist, 8ULL);

  float query[32];
  for (uint32_t d = 0U; d < dim; d++) {
    query[d] = vecs[2U * dim + d] + 0.01f;
  }

  nbase_result_t oracle[10];
  uint32_t oracle_cnt = 0U;
  test_brute_force(ids, vecs, NULL, n, dim, query, 10U, NBASE_METRIC_EUCLIDEAN, NULL, NULL, oracle, &oracle_cnt);

  nbase_result_t hits[10];
  uint32_t hit_cnt = 0U;
  ASSERT_STATUS(nbase_ivfpq_search(idx, query, 10U, 4U, hits, &hit_cnt), NBASE_OK);
  ASSERT_EQ_U64(hit_cnt, 10ULL);
  for (uint32_t i = 1U; i < hit_cnt; i++) {
    ASSERT_TRUE(hits[i - 1U].distance <= hits[i].distance);
  }
  double recall = test_recall_at(hits, hit_cnt, oracle, oracle_cnt, 10U);
  ASSERT_TRUE(recall >= 0.6);

  /* Filtered IVF-PQ search keeps only even IDs */
  nbase_result_t fhits[10];
  uint32_t fcnt = 0U;
  ASSERT_STATUS(
      nbase_ivfpq_search_filtered(idx, query, 10U, 4U, ivfpq_even_filter, NULL, fhits, &fcnt),
      NBASE_OK);
  ASSERT_TRUE(fcnt > 0U);
  for (uint32_t i = 0U; i < fcnt; i++) {
    ASSERT_TRUE((fhits[i].id % 2ULL) == 0ULL);
  }

  ASSERT_STATUS(nbase_ivfpq_save(idx, path), NBASE_OK);
  nbase_ivfpq_destroy(idx);

  nbase_ivfpq_t *loaded = NULL;
  ASSERT_STATUS(nbase_ivfpq_load(path, &loaded), NBASE_OK);
  nbase_result_t hits2[10];
  uint32_t hit_cnt2 = 0U;
  ASSERT_STATUS(nbase_ivfpq_search(loaded, query, 10U, 4U, hits2, &hit_cnt2), NBASE_OK);
  ASSERT_EQ_U64(hit_cnt2, hit_cnt);
  for (uint32_t i = 0U; i < hit_cnt; i++) {
    ASSERT_EQ_U64(hits2[i].id, hits[i].id);
    ASSERT_TRUE(fabsf(hits2[i].distance - hits[i].distance) < 1e-6f);
  }
  nbase_ivfpq_destroy(loaded);

  ASSERT_TRUE(test_flip_file_byte(path, 150L));
  ASSERT_STATUS(nbase_ivfpq_load(path, &loaded), NBASE_ERR_CORRUPT);

  free(vecs);
  free(ids);
  test_rm_rf(dir);
}

static void test_ivfpq_cosine_and_inner_product(void) {
  const uint32_t dim = 16U;
  const uint32_t n = 256U;
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);

  make_clustered_vectors(vecs, 8U, 32U, dim, 3333ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = i + 1ULL;
  }

  nbase_metric_t metrics[2] = {NBASE_METRIC_COSINE, NBASE_METRIC_INNER_PRODUCT};
  for (int mi = 0; mi < 2; mi++) {
    nbase_ivfpq_config_t cfg = nbase_ivfpq_config_default(dim, 8U, 4U);
    cfg.metric = metrics[mi];
    cfg.num_centroids = 32U;
    cfg.max_iters = 12U;

    nbase_ivfpq_t *idx = NULL;
    ASSERT_STATUS(nbase_ivfpq_train(&cfg, vecs, n, &idx), NBASE_OK);
    ASSERT_STATUS(nbase_ivfpq_add(idx, ids, vecs, n), NBASE_OK);

    nbase_result_t hits[5];
    uint32_t got = 0U;
    ASSERT_STATUS(nbase_ivfpq_search(idx, vecs, 5U, 8U, hits, &got), NBASE_OK);
    ASSERT_EQ_U64(got, 5ULL);
    nbase_ivfpq_destroy(idx);
  }

  nbase_ivfpq_config_t bad = nbase_ivfpq_config_default(dim, 4U, 4U);
  bad.metric = NBASE_METRIC_MANHATTAN;
  nbase_ivfpq_t *bad_idx = NULL;
  ASSERT_STATUS(nbase_ivfpq_train(&bad, vecs, n, &bad_idx), NBASE_ERR_UNSUPPORTED);

  free(vecs);
  free(ids);
}

typedef struct ivfpq_tsan_ctx {
  nbase_ivfpq_t *index;
  const float *query;
  atomic_bool stop;
  atomic_int errors;
} ivfpq_tsan_ctx_t;

static void *ivfpq_reader_fn(void *raw) {
  ivfpq_tsan_ctx_t *ctx = (ivfpq_tsan_ctx_t *)raw;
  for (uint32_t i = 0U; i < 40U && !atomic_load_explicit(&ctx->stop, memory_order_relaxed); i++) {
    nbase_result_t hits[5];
    uint32_t got = 0U;
    if (nbase_ivfpq_search(ctx->index, ctx->query, 5U, 4U, hits, &got) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    if (nbase_ivfpq_search_filtered(ctx->index, ctx->query, 5U, 4U, ivfpq_even_filter, NULL, hits, &got) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    nbase_ivfpq_stats_t st;
    if (nbase_ivfpq_stats(ctx->index, &st) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
  }
  return NULL;
}

static void test_ivfpq_concurrent_readers_writer_tsan(void) {
  const uint32_t dim = 16U;
  const uint32_t n = 160U;
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);
  make_clustered_vectors(vecs, 4U, 40U, dim, 7700ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = (uint64_t)i + 1ULL;
  }

  nbase_ivfpq_config_t cfg = nbase_ivfpq_config_default(dim, 8U, 4U);
  cfg.metric = NBASE_METRIC_INNER_PRODUCT;
  cfg.num_centroids = 32U;
  cfg.max_iters = 10U;
  nbase_ivfpq_t *index = NULL;
  ASSERT_STATUS(nbase_ivfpq_train(&cfg, vecs, n, &index), NBASE_OK);
  ASSERT_STATUS(nbase_ivfpq_add(index, ids, vecs, 80U), NBASE_OK);

  ivfpq_tsan_ctx_t ctx = {index, vecs, false, 0};
  pthread_t r1, r2;
  ASSERT_TRUE(pthread_create(&r1, NULL, ivfpq_reader_fn, &ctx) == 0);
  ASSERT_TRUE(pthread_create(&r2, NULL, ivfpq_reader_fn, &ctx) == 0);

  for (uint32_t i = 80U; i < n; i += 10U) {
    ASSERT_STATUS(nbase_ivfpq_add(index, ids + i, vecs + (size_t)i * dim, 10U), NBASE_OK);
  }
  atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
  pthread_join(r1, NULL);
  pthread_join(r2, NULL);
  ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);

  nbase_ivfpq_stats_t st;
  ASSERT_STATUS(nbase_ivfpq_stats(index, &st), NBASE_OK);
  ASSERT_EQ_U64(st.count, n);

  free(vecs);
  free(ids);
  nbase_ivfpq_destroy(index);
}

static void *ivfpq_batch_reader_fn(void *raw) {
  ivfpq_tsan_ctx_t *ctx = (ivfpq_tsan_ctx_t *)raw;
  const uint32_t qcnt = 6U;
  const uint32_t k = 5U;
  nbase_result_t *res = malloc((size_t)qcnt * (size_t)k * sizeof(nbase_result_t));
  uint32_t *counts = malloc((size_t)qcnt * sizeof(uint32_t));
  if (!res || !counts) {
    free(res);
    free(counts);
    atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
    return NULL;
  }
  for (uint32_t i = 0U; i < 20U && !atomic_load_explicit(&ctx->stop, memory_order_relaxed); i++) {
    nbase_status_t st = nbase_ivfpq_search_batch(ctx->index, ctx->query, qcnt, k, 4U, 2U, res, counts);
    if (st != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
  }
  free(res);
  free(counts);
  return NULL;
}

static void test_ivfpq_search_batch_mt_and_tsan(void) {
  const uint32_t dim = 16U;
  const uint32_t n = 160U;
  const uint32_t qcnt = 10U;
  const uint32_t k = 5U;
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  nbase_result_t *batch_res = malloc((size_t)qcnt * k * sizeof(nbase_result_t));
  nbase_result_t *single_res = malloc((size_t)k * sizeof(nbase_result_t));
  uint32_t *batch_cnts = malloc((size_t)qcnt * sizeof(uint32_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL && batch_res != NULL && single_res != NULL && batch_cnts != NULL);

  make_clustered_vectors(vecs, 4U, 40U, dim, 8800ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = (uint64_t)i + 1ULL;
  }

  static const nbase_metric_t ms[2] = {NBASE_METRIC_EUCLIDEAN, NBASE_METRIC_COSINE};
  for (int mi = 0; mi < 2; mi++) {
    nbase_ivfpq_config_t cfg = nbase_ivfpq_config_default(dim, 8U, 4U);
    cfg.metric = ms[mi];
    cfg.num_centroids = 32U;
    cfg.max_iters = 10U;
    nbase_ivfpq_t *index = NULL;
    ASSERT_STATUS(nbase_ivfpq_train(&cfg, vecs, n, &index), NBASE_OK);
    ASSERT_STATUS(nbase_ivfpq_add(index, ids, vecs, n), NBASE_OK);

    ASSERT_STATUS(nbase_ivfpq_search_batch(index, vecs, 0U, k, 4U, 4U, NULL, NULL), NBASE_OK);
    ASSERT_STATUS(nbase_ivfpq_search_batch(index, vecs, qcnt, k, 4U, 4U, batch_res, batch_cnts), NBASE_OK);

    for (uint32_t q = 0U; q < qcnt; q++) {
      uint32_t s_cnt = 0U;
      ASSERT_STATUS(nbase_ivfpq_search(index, vecs + (size_t)q * dim, k, 4U, single_res, &s_cnt), NBASE_OK);
      ASSERT_EQ_U64(batch_cnts[q], s_cnt);
      for (uint32_t r = 0U; r < s_cnt; r++) {
        ASSERT_EQ_U64(batch_res[(size_t)q * k + r].id, single_res[r].id);
        ASSERT_NEAR(batch_res[(size_t)q * k + r].distance, single_res[r].distance, 1e-6, 1e-6);
      }
    }

    float bad_q[16];
    memcpy(bad_q, vecs, sizeof(bad_q));
    bad_q[3] = (float)NAN;
    ASSERT_STATUS(nbase_ivfpq_search_batch(index, bad_q, 1U, k, 4U, 2U, batch_res, batch_cnts), NBASE_ERR_INVALID_VECTOR);
    ASSERT_STATUS(nbase_ivfpq_search_batch(index, NULL, 1U, k, 4U, 2U, batch_res, batch_cnts), NBASE_ERR_INVALID_ARG);

    nbase_ivfpq_destroy(index);
  }

  /* Concurrent MT batch search readers + writer add under TSan */
  nbase_ivfpq_config_t cfg = nbase_ivfpq_config_default(dim, 8U, 4U);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.num_centroids = 32U;
  nbase_ivfpq_t *mt_idx = NULL;
  ASSERT_STATUS(nbase_ivfpq_train(&cfg, vecs, n, &mt_idx), NBASE_OK);
  ASSERT_STATUS(nbase_ivfpq_add(mt_idx, ids, vecs, 80U), NBASE_OK);

  ivfpq_tsan_ctx_t ctx = {mt_idx, vecs, false, 0};
  pthread_t r1, r2;
  ASSERT_TRUE(pthread_create(&r1, NULL, ivfpq_batch_reader_fn, &ctx) == 0);
  ASSERT_TRUE(pthread_create(&r2, NULL, ivfpq_batch_reader_fn, &ctx) == 0);
  for (uint32_t i = 80U; i < n; i += 10U) {
    ASSERT_STATUS(nbase_ivfpq_add(mt_idx, ids + i, vecs + (size_t)i * dim, 10U), NBASE_OK);
  }
  atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
  pthread_join(r1, NULL);
  pthread_join(r2, NULL);
  ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);

  nbase_ivfpq_destroy(mt_idx);
  free(batch_cnts);
  free(single_res);
  free(batch_res);
  free(ids);
  free(vecs);
}

void test_ivfpq_run(void) {
  RUN_TEST(test_ivfpq_train_add_search_euclidean_and_save_load);
  RUN_TEST(test_ivfpq_cosine_and_inner_product);
  RUN_TEST(test_ivfpq_concurrent_readers_writer_tsan);
  RUN_TEST(test_ivfpq_search_batch_mt_and_tsan);
}
