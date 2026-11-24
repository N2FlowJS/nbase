#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define NBASE_TWO_PI 6.28318530717958647692
#define NBASE_LSH_MAX_TABLES 64U
#define NBASE_LSH_MAX_BITS 20U

typedef struct lsh_bucket {
  uint32_t *slots;
  uint32_t len;
  uint32_t cap;
} lsh_bucket_t;

struct nbase_lsh {
  uint32_t dim;
  nbase_metric_t metric;
  uint32_t num_tables;
  uint32_t num_bits;
  uint32_t num_buckets;
  float bucket_width;
  uint64_t seed;
  nbase_store_t *store;
  float *planes;
  float *offsets;
  lsh_bucket_t *buckets;
  pthread_rwlock_t rwlock;
  bool rwlock_init;
};

static int cmp_lsh_u64_asc(const void *pa, const void *pb) {
  uint64_t a = *(const uint64_t *)pa;
  uint64_t b = *(const uint64_t *)pb;
  return (a > b) - (a < b);
}

typedef struct lsh_bit_margin {
  uint32_t bit;
  float margin;
} lsh_bit_margin_t;

static int cmp_margin_asc(const void *pa, const void *pb) {
  const lsh_bit_margin_t *a = (const lsh_bit_margin_t *)pa;
  const lsh_bit_margin_t *b = (const lsh_bit_margin_t *)pb;
  if (a->margin < b->margin) {
    return -1;
  }
  if (a->margin > b->margin) {
    return 1;
  }
  if (a->bit < b->bit) {
    return -1;
  }
  if (a->bit > b->bit) {
    return 1;
  }
  return 0;
}

static bool lsh_metric_supported(nbase_metric_t metric) {
  return metric == NBASE_METRIC_COSINE || metric == NBASE_METRIC_EUCLIDEAN;
}

static float rng_gaussian(uint64_t *state) {
  double u1 = nbase_rng_next_double(state);
  double u2 = nbase_rng_next_double(state);
  if (u1 < 1e-12) {
    u1 = 1e-12;
  }
  double z = sqrt(-2.0 * log(u1)) * cos(NBASE_TWO_PI * u2);
  return (float)z;
}

static uint32_t lsh_hash_from_ip_dists(
    const nbase_lsh_t *lsh,
    uint32_t table,
    const float *table_ip_dists,
    lsh_bit_margin_t *out_margins) {
  uint32_t h = 0U;
  const float *t_offsets = lsh->offsets + (size_t)table * (size_t)lsh->num_bits;

  for (uint32_t b = 0U; b < lsh->num_bits; b++) {
    double dot = -(double)table_ip_dists[b];

    if (lsh->metric == NBASE_METRIC_COSINE) {
      if (dot >= 0.0) {
        h |= (1U << b);
      }
      if (out_margins) {
        out_margins[b].bit = b;
        out_margins[b].margin = (float)fabs(dot);
      }
    } else {
      double shifted = (dot + (double)t_offsets[b]) / (double)lsh->bucket_width;
      double fl = floor(shifted);
      int64_t bin = (int64_t)fl;
      if ((bin & 1LL) != 0LL) {
        h |= (1U << b);
      }
      if (out_margins) {
        double frac = shifted - fl;
        double m = (frac < 0.5) ? frac : (1.0 - frac);
        out_margins[b].bit = b;
        out_margins[b].margin = (float)m;
      }
    }
  }
  return h;
}

static nbase_status_t lsh_compute_all_ip_dists(
    const nbase_lsh_t *lsh,
    const float *vec,
    float *out_ip_dists) {
  uint32_t total_planes = lsh->num_tables * lsh->num_bits;
  return nbase_distance_batch(
      NBASE_METRIC_INNER_PRODUCT, vec, lsh->planes, total_planes, lsh->dim, out_ip_dists);
}

static nbase_status_t bucket_push(lsh_bucket_t *b, uint32_t slot) {
  if (b->len == b->cap) {
    uint32_t new_cap = (b->cap == 0U) ? 8U : b->cap * 2U;
    uint32_t *next = realloc(b->slots, (size_t)new_cap * sizeof(uint32_t));
    if (!next) {
      return NBASE_ERR_NOMEM;
    }
    b->slots = next;
    b->cap = new_cap;
  }
  b->slots[b->len++] = slot;
  return NBASE_OK;
}

static void bucket_remove(lsh_bucket_t *b, uint32_t slot) {
  for (uint32_t i = 0U; i < b->len; i++) {
    if (b->slots[i] == slot) {
      b->slots[i] = b->slots[b->len - 1U];
      b->len--;
      return;
    }
  }
}

nbase_lsh_config_t nbase_lsh_config_default(uint32_t dim) {
  nbase_lsh_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.dim = dim;
  cfg.metric = NBASE_METRIC_COSINE;
  cfg.num_tables = 8U;
  cfg.num_bits = 12U;
  cfg.bucket_width = 4.0f;
  cfg.seed = 0x4E424153454C5348ULL;
  return cfg;
}

static nbase_status_t lsh_alloc_internal(
    uint32_t dim,
    nbase_metric_t metric,
    uint32_t num_tables,
    uint32_t num_bits,
    float bucket_width,
    uint64_t seed,
    uint32_t init_cap,
    nbase_lsh_t **out_lsh) {
  uint32_t num_buckets = 1U << num_bits;
  size_t plane_floats = (size_t)num_tables * (size_t)num_bits * (size_t)dim;
  size_t offset_floats = (size_t)num_tables * (size_t)num_bits;
  size_t total_buckets = (size_t)num_tables * (size_t)num_buckets;

  if (plane_floats > SIZE_MAX / sizeof(float) ||
      total_buckets > SIZE_MAX / sizeof(lsh_bucket_t)) {
    return NBASE_ERR_NOMEM;
  }

  nbase_lsh_t *lsh = calloc(1, sizeof(*lsh));
  float *planes = malloc(plane_floats * sizeof(float));
  float *offsets = calloc(offset_floats, sizeof(float));
  lsh_bucket_t *buckets = calloc(total_buckets, sizeof(lsh_bucket_t));
  nbase_store_t *store = NULL;
  nbase_status_t st = nbase_store_create(dim, metric, init_cap, &store);

  if (!lsh || !planes || !offsets || !buckets || st != NBASE_OK) {
    free(lsh);
    free(planes);
    free(offsets);
    free(buckets);
    nbase_store_destroy(store);
    return (st != NBASE_OK) ? st : NBASE_ERR_NOMEM;
  }

  lsh->dim = dim;
  lsh->metric = metric;
  lsh->num_tables = num_tables;
  lsh->num_bits = num_bits;
  lsh->num_buckets = num_buckets;
  lsh->bucket_width = bucket_width;
  lsh->seed = seed != 0ULL ? seed : 0x4E424153454C5348ULL;
  lsh->store = store;
  lsh->planes = planes;
  lsh->offsets = offsets;
  lsh->buckets = buckets;
  if (pthread_rwlock_init(&lsh->rwlock, NULL) != 0) {
    nbase_lsh_destroy(lsh);
    return NBASE_ERR_NOMEM;
  }
  lsh->rwlock_init = true;
  *out_lsh = lsh;
  return NBASE_OK;
}

nbase_status_t nbase_lsh_create(const nbase_lsh_config_t *cfg, nbase_lsh_t **out_lsh) {
  if (!cfg || !out_lsh) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_lsh = NULL;

  if (!nbase_metric_is_valid(cfg->metric)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (!lsh_metric_supported(cfg->metric)) {
    return NBASE_ERR_UNSUPPORTED;
  }
  if (cfg->dim == 0U || cfg->dim > NBASE_MAX_DIM ||
      cfg->num_tables == 0U || cfg->num_tables > NBASE_LSH_MAX_TABLES ||
      cfg->num_bits == 0U || cfg->num_bits > NBASE_LSH_MAX_BITS) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (cfg->metric == NBASE_METRIC_EUCLIDEAN && (!isfinite(cfg->bucket_width) || cfg->bucket_width <= 0.0f)) {
    return NBASE_ERR_INVALID_ARG;
  }

  float bw = (cfg->metric == NBASE_METRIC_EUCLIDEAN) ? cfg->bucket_width : 1.0f;
  uint64_t seed = (cfg->seed == 0ULL) ? 0x4E424153454C5348ULL : cfg->seed;

  nbase_lsh_t *lsh = NULL;
  nbase_status_t st = lsh_alloc_internal(cfg->dim, cfg->metric, cfg->num_tables, cfg->num_bits, bw, seed, 64U, &lsh);
  if (st != NBASE_OK) {
    return st;
  }

  uint64_t rng = seed;
  size_t plane_cnt = (size_t)cfg->num_tables * (size_t)cfg->num_bits * (size_t)cfg->dim;
  for (size_t i = 0U; i < plane_cnt; i++) {
    lsh->planes[i] = rng_gaussian(&rng);
  }
  if (cfg->metric == NBASE_METRIC_EUCLIDEAN) {
    size_t off_cnt = (size_t)cfg->num_tables * (size_t)cfg->num_bits;
    for (size_t i = 0U; i < off_cnt; i++) {
      lsh->offsets[i] = (float)(nbase_rng_next_double(&rng) * (double)bw);
    }
  }

  *out_lsh = lsh;
  return NBASE_OK;
}

void nbase_lsh_destroy(nbase_lsh_t *lsh) {
  if (!lsh) {
    return;
  }
  if (lsh->rwlock_init) {
    pthread_rwlock_destroy(&lsh->rwlock);
  }
  if (lsh->buckets) {
    size_t total = (size_t)lsh->num_tables * (size_t)lsh->num_buckets;
    for (size_t i = 0U; i < total; i++) {
      free(lsh->buckets[i].slots);
    }
    free(lsh->buckets);
  }
  free(lsh->planes);
  free(lsh->offsets);
  nbase_store_destroy(lsh->store);
  free(lsh);
}

static nbase_status_t lsh_add_unlocked(
    nbase_lsh_t *lsh,
    uint64_t id,
    const float *vec,
    uint32_t dim) {
  if (dim != lsh->dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }

  uint32_t slot = NBASE_INVALID_SLOT;
  nbase_status_t st = nbase_store_add(lsh->store, id, vec, &slot);
  if (st != NBASE_OK) {
    return st;
  }

  float ip_dists[NBASE_LSH_MAX_TABLES * NBASE_LSH_MAX_BITS] = {0.0f};
  st = lsh_compute_all_ip_dists(lsh, vec, ip_dists);
  if (st != NBASE_OK) {
    nbase_store_rollback_slot(lsh->store, slot);
    return st;
  }

  uint32_t hashes[NBASE_LSH_MAX_TABLES] = {0U};
  for (uint32_t t = 0U; t < lsh->num_tables; t++) {
    uint32_t h = lsh_hash_from_ip_dists(
        lsh, t, ip_dists + (size_t)t * (size_t)lsh->num_bits, NULL);
    hashes[t] = h;
    st = bucket_push(&lsh->buckets[(size_t)t * (size_t)lsh->num_buckets + h], slot);
    if (st != NBASE_OK) {
      for (uint32_t prev = 0U; prev < t; prev++) {
        bucket_remove(
            &lsh->buckets[(size_t)prev * (size_t)lsh->num_buckets + hashes[prev]], slot);
      }
      nbase_store_rollback_slot(lsh->store, slot);
      return st;
    }
  }
  return NBASE_OK;
}

static nbase_status_t lsh_delete_unlocked(nbase_lsh_t *lsh, uint64_t id) {
  uint32_t slot = NBASE_INVALID_SLOT;
  if (!nbase_store_find_slot(lsh->store, id, &slot)) {
    return NBASE_ERR_NOT_FOUND;
  }
  const float *vec = nbase_store_get_vector(lsh->store, slot);
  if (vec) {
    float ip_dists[NBASE_LSH_MAX_TABLES * NBASE_LSH_MAX_BITS] = {0.0f};
    if (lsh_compute_all_ip_dists(lsh, vec, ip_dists) == NBASE_OK) {
      for (uint32_t t = 0U; t < lsh->num_tables; t++) {
        uint32_t h = lsh_hash_from_ip_dists(
            lsh, t, ip_dists + (size_t)t * (size_t)lsh->num_bits, NULL);
        bucket_remove(&lsh->buckets[(size_t)t * (size_t)lsh->num_buckets + h], slot);
      }
    }
  }
  return nbase_store_delete(lsh->store, id, NULL);
}

nbase_status_t nbase_lsh_add(nbase_lsh_t *lsh, uint64_t id, const float *vec, uint32_t dim) {
  if (!lsh || !vec) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (dim != lsh->dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }
  pthread_rwlock_wrlock(&lsh->rwlock);
  nbase_status_t st = lsh_add_unlocked(lsh, id, vec, dim);
  pthread_rwlock_unlock(&lsh->rwlock);
  return st;
}

nbase_status_t nbase_lsh_add_batch(
    nbase_lsh_t *lsh,
    const uint64_t *ids,
    const float *vectors,
    uint32_t count,
    uint32_t dim) {
  if (!lsh || (count > 0U && (!ids || !vectors))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (dim != lsh->dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (count == 0U) {
    return NBASE_OK;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)dim, dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }
  if (count > 1U) {
    uint64_t *sorted = malloc((size_t)count * sizeof(uint64_t));
    if (!sorted) {
      return NBASE_ERR_NOMEM;
    }
    memcpy(sorted, ids, (size_t)count * sizeof(uint64_t));
    qsort(sorted, count, sizeof(uint64_t), cmp_lsh_u64_asc);
    for (uint32_t i = 1U; i < count; i++) {
      if (sorted[i] == sorted[i - 1U]) {
        free(sorted);
        return NBASE_ERR_ALREADY_EXISTS;
      }
    }
    free(sorted);
  }

  pthread_rwlock_wrlock(&lsh->rwlock);
  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < count; i++) {
    if (nbase_store_find_slot(lsh->store, ids[i], NULL)) {
      st = NBASE_ERR_ALREADY_EXISTS;
      break;
    }
  }
  for (uint32_t i = 0U; i < count && st == NBASE_OK; i++) {
    st = lsh_add_unlocked(lsh, ids[i], vectors + (size_t)i * (size_t)dim, dim);
    if (st != NBASE_OK) {
      for (uint32_t r = 0U; r < i; r++) {
        (void)lsh_delete_unlocked(lsh, ids[r]);
      }
      break;
    }
  }
  pthread_rwlock_unlock(&lsh->rwlock);
  return st;
}

nbase_status_t nbase_lsh_delete(nbase_lsh_t *lsh, uint64_t id) {
  if (!lsh) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&lsh->rwlock);
  nbase_status_t st = lsh_delete_unlocked(lsh, id);
  pthread_rwlock_unlock(&lsh->rwlock);
  return st;
}

bool nbase_lsh_contains(const nbase_lsh_t *lsh, uint64_t id) {
  if (!lsh) {
    return false;
  }
  nbase_lsh_t *mutable_lsh = (nbase_lsh_t *)(uintptr_t)lsh;
  pthread_rwlock_rdlock(&mutable_lsh->rwlock);
  bool found = nbase_store_find_slot(lsh->store, id, NULL);
  pthread_rwlock_unlock(&mutable_lsh->rwlock);
  return found;
}

static nbase_status_t lsh_search_filtered_unlocked(
    const nbase_lsh_t *lsh,
    const float *query,
    uint32_t dim,
    uint32_t k,
    uint32_t multi_probe,
    nbase_filter_fn filter,
    void *filter_ctx,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (dim != lsh->dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (!nbase_vec_is_finite(query, dim)) {
    return NBASE_ERR_INVALID_VECTOR;
  }
  if (k == 0U || lsh->store->active_count == 0ULL) {
    return NBASE_OK;
  }

  float ip_dists[NBASE_LSH_MAX_TABLES * NBASE_LSH_MAX_BITS] = {0.0f};
  nbase_status_t ip_st = lsh_compute_all_ip_dists(lsh, query, ip_dists);
  if (ip_st != NBASE_OK) {
    return ip_st;
  }

  uint8_t *seen = calloc((size_t)lsh->store->slot_count, sizeof(uint8_t));
  if (!seen) {
    return NBASE_ERR_NOMEM;
  }
  nbase_heap_t topk;
  nbase_heap_init(&topk, true);

  lsh_bit_margin_t margins[NBASE_LSH_MAX_BITS];
  nbase_dist_fn dist_fn = nbase_get_distance_fn(lsh->metric);
  uint32_t max_flips = (multi_probe > lsh->num_bits) ? lsh->num_bits : multi_probe;

  for (uint32_t t = 0U; t < lsh->num_tables; t++) {
    uint32_t base_h = lsh_hash_from_ip_dists(
        lsh,
        t,
        ip_dists + (size_t)t * (size_t)lsh->num_bits,
        max_flips > 0U ? margins : NULL);
    if (max_flips > 0U) {
      qsort(margins, lsh->num_bits, sizeof(lsh_bit_margin_t), cmp_margin_asc);
    }

    for (uint32_t p = 0U; p <= max_flips; p++) {
      uint32_t h = (p == 0U) ? base_h : (base_h ^ (1U << margins[p - 1U].bit));
      const lsh_bucket_t *b = &lsh->buckets[(size_t)t * (size_t)lsh->num_buckets + h];
      for (uint32_t i = 0U; i < b->len; i++) {
        uint32_t slot = b->slots[i];
        if (slot >= lsh->store->slot_count || seen[slot] != 0U) {
          continue;
        }
        seen[slot] = 1U;
        if (lsh->store->slot_flags[slot] != NBASE_SLOT_ACTIVE) {
          continue;
        }
        if (filter && !filter(lsh->store->ids[slot], filter_ctx)) {
          continue;
        }

        float d = dist_fn(query, nbase_store_get_vector(lsh->store, slot), dim);
        nbase_heap_item_t item = {slot, d};
        if (topk.len < k) {
          if (!nbase_heap_push(&topk, item)) {
            nbase_heap_free(&topk);
            free(seen);
            return NBASE_ERR_NOMEM;
          }
        } else if (topk.len > 0U && nbase_heap_higher(true, topk.data[0], item)) {
          (void)nbase_heap_pop(&topk);
          (void)nbase_heap_push(&topk, item);
        }
      }
    }
  }

  uint32_t nres = (uint32_t)topk.len;
  for (uint32_t i = 0U; i < nres; i++) {
    nbase_heap_item_t item = nbase_heap_pop(&topk);
    out_results[nres - 1U - i].id = lsh->store->ids[item.slot];
    out_results[nres - 1U - i].distance = item.dist;
  }
  qsort(out_results, nres, sizeof(nbase_result_t), nbase_cmp_result_asc);
  *out_count = nres;

  nbase_heap_free(&topk);
  free(seen);
  return NBASE_OK;
}

nbase_status_t nbase_lsh_search_filtered(
    const nbase_lsh_t *lsh,
    const float *query,
    uint32_t dim,
    uint32_t k,
    uint32_t multi_probe,
    nbase_filter_fn filter,
    void *filter_ctx,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!lsh || !query || !out_count || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  nbase_lsh_t *mutable_lsh = (nbase_lsh_t *)(uintptr_t)lsh;
  pthread_rwlock_rdlock(&mutable_lsh->rwlock);
  nbase_status_t st = lsh_search_filtered_unlocked(
      lsh, query, dim, k, multi_probe, filter, filter_ctx, out_results, out_count);
  pthread_rwlock_unlock(&mutable_lsh->rwlock);
  return st;
}

nbase_status_t nbase_lsh_search(
    const nbase_lsh_t *lsh,
    const float *query,
    uint32_t dim,
    uint32_t k,
    uint32_t multi_probe,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  return nbase_lsh_search_filtered(lsh, query, dim, k, multi_probe, NULL, NULL, out_results, out_count);
}

typedef struct lsh_batch_ctx {
  const nbase_lsh_t *lsh;
  const float *queries;
  uint32_t query_count;
  uint32_t k;
  uint32_t multi_probe;
  nbase_result_t *out_results;
  uint32_t *out_counts;
  atomic_uint next_q;
  atomic_int first_err;
} lsh_batch_ctx_t;

static void *lsh_search_batch_worker(void *raw) {
  lsh_batch_ctx_t *ctx = (lsh_batch_ctx_t *)raw;
  while (true) {
    if (atomic_load_explicit(&ctx->first_err, memory_order_acquire) != (int)NBASE_OK) {
      break;
    }
    uint32_t q = atomic_fetch_add_explicit(&ctx->next_q, 1U, memory_order_relaxed);
    if (q >= ctx->query_count) {
      break;
    }
    uint32_t got = 0U;
    nbase_status_t st = lsh_search_filtered_unlocked(
        ctx->lsh,
        ctx->queries + (size_t)q * (size_t)ctx->lsh->dim,
        ctx->lsh->dim,
        ctx->k,
        ctx->multi_probe,
        NULL,
        NULL,
        ctx->out_results + (size_t)q * (size_t)ctx->k,
        &got);
    ctx->out_counts[q] = got;
    if (st != NBASE_OK) {
      int expected = (int)NBASE_OK;
      (void)atomic_compare_exchange_strong(&ctx->first_err, &expected, (int)st);
      break;
    }
  }
  return NULL;
}

nbase_status_t nbase_lsh_search_batch(
    const nbase_lsh_t *lsh,
    const float *queries,
    uint32_t query_count,
    uint32_t dim,
    uint32_t k,
    uint32_t multi_probe,
    uint32_t num_threads,
    nbase_result_t *out_results,
    uint32_t *out_counts) {
  if (!lsh) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (dim != lsh->dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (query_count == 0U) {
    return NBASE_OK;
  }
  if (!queries || !out_counts || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if ((size_t)query_count > SIZE_MAX / (size_t)dim ||
      (k > 0U && (size_t)query_count > SIZE_MAX / (size_t)k)) {
    return NBASE_ERR_NOMEM;
  }
  for (uint32_t q = 0U; q < query_count; q++) {
    out_counts[q] = 0U;
    if (!nbase_vec_is_finite(queries + (size_t)q * (size_t)dim, dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }
  if (k == 0U) {
    return NBASE_OK;
  }

  uint32_t workers = (num_threads == 0U) ? 4U : num_threads;
  if (workers > 8U) {
    workers = 8U;
  }
  if (workers > query_count) {
    workers = query_count;
  }

  nbase_lsh_t *mutable_lsh = (nbase_lsh_t *)(uintptr_t)lsh;
  pthread_rwlock_rdlock(&mutable_lsh->rwlock);

  lsh_batch_ctx_t ctx;
  ctx.lsh = lsh;
  ctx.queries = queries;
  ctx.query_count = query_count;
  ctx.k = k;
  ctx.multi_probe = multi_probe;
  ctx.out_results = out_results;
  ctx.out_counts = out_counts;
  atomic_init(&ctx.next_q, 0U);
  atomic_init(&ctx.first_err, (int)NBASE_OK);

  if (workers <= 1U) {
    (void)lsh_search_batch_worker(&ctx);
  } else {
    pthread_t th[8];
    uint32_t created = 0U;
    for (uint32_t t = 0U; t < workers; t++) {
      if (pthread_create(&th[t], NULL, lsh_search_batch_worker, &ctx) != 0) {
        int expected = (int)NBASE_OK;
        (void)atomic_compare_exchange_strong(&ctx.first_err, &expected, (int)NBASE_ERR_NOMEM);
        break;
      }
      created++;
    }
    for (uint32_t t = 0U; t < created; t++) {
      pthread_join(th[t], NULL);
    }
  }

  pthread_rwlock_unlock(&mutable_lsh->rwlock);
  return (nbase_status_t)atomic_load_explicit(&ctx.first_err, memory_order_acquire);
}

nbase_status_t nbase_lsh_stats(const nbase_lsh_t *lsh, nbase_lsh_stats_t *out_stats) {
  if (!lsh || !out_stats) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_lsh_t *mutable_lsh = (nbase_lsh_t *)(uintptr_t)lsh;
  pthread_rwlock_rdlock(&mutable_lsh->rwlock);
  out_stats->dim = lsh->dim;
  out_stats->metric = lsh->metric;
  out_stats->num_tables = lsh->num_tables;
  out_stats->num_bits = lsh->num_bits;
  out_stats->bucket_width = lsh->bucket_width;
  out_stats->active_vectors = lsh->store->active_count;
  pthread_rwlock_unlock(&mutable_lsh->rwlock);
  return NBASE_OK;
}

static nbase_status_t lsh_save_unlocked(const nbase_lsh_t *lsh, const char *file_path) {

  size_t plane_bytes = (size_t)lsh->num_tables * (size_t)lsh->num_bits * (size_t)lsh->dim * sizeof(float);
  size_t off_bytes = (size_t)lsh->num_tables * (size_t)lsh->num_bits * sizeof(float);
  uint64_t active = lsh->store->active_count;
  size_t ids_bytes = (size_t)active * sizeof(uint64_t);
  size_t vecs_bytes = (size_t)active * (size_t)lsh->dim * sizeof(float);

  uint64_t *active_ids = NULL;
  float *active_vecs = NULL;
  if (active > 0ULL) {
    active_ids = malloc(ids_bytes);
    active_vecs = malloc(vecs_bytes);
    if (!active_ids || !active_vecs) {
      free(active_ids);
      free(active_vecs);
      return NBASE_ERR_NOMEM;
    }
    uint64_t pos = 0ULL;
    for (uint32_t s = 0U; s < lsh->store->slot_count && pos < active; s++) {
      if (lsh->store->slot_flags[s] == NBASE_SLOT_ACTIVE) {
        active_ids[pos] = lsh->store->ids[s];
        memcpy(active_vecs + (size_t)pos * (size_t)lsh->dim, nbase_store_get_vector(lsh->store, s), (size_t)lsh->dim * sizeof(float));
        pos++;
      }
    }
  }

  nbase_io_chunk_t chunks[4] = {
      {lsh->planes, plane_bytes},
      {lsh->offsets, off_bytes},
      {active_ids, ids_bytes},
      {active_vecs, vecs_bytes},
  };
  uint32_t payload_crc = 0U;
  for (size_t i = 0U; i < 4U; i++) {
    if (chunks[i].len > 0U && chunks[i].ptr != NULL) {
      payload_crc = nbase_crc32_update(payload_crc, chunks[i].ptr, chunks[i].len);
    }
  }

  nblh_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBLH_MAGIC, 8);
  hdr.version = NBASE_NBLH_VERSION;
  hdr.dim = lsh->dim;
  hdr.metric = (uint32_t)lsh->metric;
  hdr.num_tables = lsh->num_tables;
  hdr.num_bits = lsh->num_bits;
  hdr.bucket_width = lsh->bucket_width;
  hdr.seed = lsh->seed;
  hdr.active_vectors = active;
  hdr.payload_bytes = (uint64_t)plane_bytes + (uint64_t)off_bytes + (uint64_t)ids_bytes + (uint64_t)vecs_bytes;
  hdr.payload_crc32 = payload_crc;
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nblh_header_t, header_crc32));

  nbase_status_t st = nbase_write_file_atomic(file_path, &hdr, sizeof(hdr), chunks, 4U);
  free(active_ids);
  free(active_vecs);
  return st;
}

nbase_status_t nbase_lsh_save(const nbase_lsh_t *lsh, const char *file_path) {
  if (!lsh || !file_path || file_path[0] == '\0') {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_lsh_t *mutable_lsh = (nbase_lsh_t *)(uintptr_t)lsh;
  pthread_rwlock_rdlock(&mutable_lsh->rwlock);
  nbase_status_t st = lsh_save_unlocked(lsh, file_path);
  pthread_rwlock_unlock(&mutable_lsh->rwlock);
  return st;
}

nbase_status_t nbase_lsh_load(const char *file_path, nbase_lsh_t **out_lsh) {
  if (!file_path || file_path[0] == '\0' || !out_lsh) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_lsh = NULL;

  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    return errno == ENOENT ? NBASE_ERR_NOT_FOUND : NBASE_ERR_IO;
  }

  struct stat st_buf;
  if (fstat(fd, &st_buf) != 0) {
    close(fd);
    return NBASE_ERR_IO;
  }
  if (st_buf.st_size < (off_t)sizeof(nblh_header_t)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;

  nblh_header_t hdr;
  nbase_status_t st = nbase_read_exact(fd, &hdr, sizeof(hdr));
  if (st != NBASE_OK) {
    close(fd);
    return st;
  }
  if (memcmp(hdr.magic, NBASE_NBLH_MAGIC, 8) != 0 || hdr.version != NBASE_NBLH_VERSION) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.header_crc32 != nbase_crc32(&hdr, offsetof(nblh_header_t, header_crc32))) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  nbase_metric_t metric = (nbase_metric_t)hdr.metric;
  if (!lsh_metric_supported(metric) || hdr.dim == 0U || hdr.dim > NBASE_MAX_DIM ||
      hdr.num_tables == 0U || hdr.num_tables > NBASE_LSH_MAX_TABLES ||
      hdr.num_bits == 0U || hdr.num_bits > NBASE_LSH_MAX_BITS ||
      !isfinite(hdr.bucket_width) || hdr.bucket_width <= 0.0f) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint64_t plane_bytes_u64 = (uint64_t)hdr.num_tables * (uint64_t)hdr.num_bits * (uint64_t)hdr.dim * (uint64_t)sizeof(float);
  uint64_t off_bytes_u64 = (uint64_t)hdr.num_tables * (uint64_t)hdr.num_bits * (uint64_t)sizeof(float);
  uint64_t vec_record_u64 = (uint64_t)sizeof(uint64_t) + (uint64_t)hdr.dim * (uint64_t)sizeof(float);
  if (hdr.active_vectors > UINT32_MAX ||
      hdr.active_vectors > hdr.payload_bytes / vec_record_u64) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t ids_bytes_u64 = hdr.active_vectors * (uint64_t)sizeof(uint64_t);
  uint64_t vecs_bytes_u64 = hdr.active_vectors * (uint64_t)hdr.dim * (uint64_t)sizeof(float);
  uint64_t expected_u64 = plane_bytes_u64 + off_bytes_u64 + ids_bytes_u64 + vecs_bytes_u64;
  if (hdr.payload_bytes != expected_u64 ||
      file_size != sizeof(nblh_header_t) + expected_u64 ||
      expected_u64 > (uint64_t)SIZE_MAX) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint8_t *payload = malloc((size_t)expected_u64);
  if (!payload) {
    close(fd);
    return NBASE_ERR_NOMEM;
  }
  st = nbase_read_exact(fd, payload, (size_t)expected_u64);
  close(fd);
  if (st != NBASE_OK) {
    free(payload);
    return st;
  }
  if (nbase_crc32(payload, (size_t)expected_u64) != hdr.payload_crc32) {
    free(payload);
    return NBASE_ERR_CORRUPT;
  }

  uint32_t init_cap = (hdr.active_vectors > 64ULL) ? (uint32_t)hdr.active_vectors : 64U;
  nbase_lsh_t *lsh = NULL;
  st = lsh_alloc_internal(hdr.dim, metric, hdr.num_tables, hdr.num_bits, hdr.bucket_width, hdr.seed, init_cap, &lsh);
  if (st != NBASE_OK) {
    free(payload);
    return st;
  }

  size_t off = 0U;
  memcpy(lsh->planes, payload + off, (size_t)plane_bytes_u64);
  off += (size_t)plane_bytes_u64;
  memcpy(lsh->offsets, payload + off, (size_t)off_bytes_u64);
  off += (size_t)off_bytes_u64;

  uint32_t plane_floats = hdr.num_tables * hdr.num_bits * hdr.dim;
  uint32_t off_floats = hdr.num_tables * hdr.num_bits;
  if (!nbase_vec_is_finite(lsh->planes, plane_floats) || !nbase_vec_is_finite(lsh->offsets, off_floats)) {
    free(payload);
    nbase_lsh_destroy(lsh);
    return NBASE_ERR_CORRUPT;
  }

  if (hdr.active_vectors > 0ULL) {
    uint64_t *ids = malloc((size_t)ids_bytes_u64);
    float *vecs = malloc((size_t)vecs_bytes_u64);
    if (!ids || !vecs) {
      free(ids);
      free(vecs);
      free(payload);
      nbase_lsh_destroy(lsh);
      return NBASE_ERR_NOMEM;
    }
    memcpy(ids, payload + off, (size_t)ids_bytes_u64);
    off += (size_t)ids_bytes_u64;
    memcpy(vecs, payload + off, (size_t)vecs_bytes_u64);

    for (uint32_t i = 0U; i < (uint32_t)hdr.active_vectors; i++) {
      st = lsh_add_unlocked(lsh, ids[i], vecs + (size_t)i * (size_t)hdr.dim, hdr.dim);
      if (st != NBASE_OK) {
        free(ids);
        free(vecs);
        free(payload);
        nbase_lsh_destroy(lsh);
        return (st == NBASE_ERR_INVALID_VECTOR || st == NBASE_ERR_ALREADY_EXISTS) ? NBASE_ERR_CORRUPT : st;
      }
    }
    free(ids);
    free(vecs);
  }

  free(payload);
  *out_lsh = lsh;
  return NBASE_OK;
}
