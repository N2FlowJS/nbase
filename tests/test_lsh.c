#include "test_common.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

static bool lsh_even_filter(uint64_t id, void *ctx) {
  (void)ctx;
  return (id % 2ULL) == 0ULL;
}

static void make_clusters(float *out, uint32_t clusters, uint32_t per_cluster, uint32_t dim, uint64_t seed) {
  test_fill_vectors(out, clusters * per_cluster, dim, seed);
  for (uint32_t c = 0U; c < clusters; c++) {
    for (uint32_t i = 0U; i < per_cluster; i++) {
      float *v = out + ((size_t)c * per_cluster + i) * dim;
      for (uint32_t d = 0U; d < dim; d++) {
        v[d] = v[d] * 0.1f + ((d == (c % dim)) ? 5.0f : 0.0f);
      }
    }
  }
}

static void test_lsh_cosine_and_multiprobe(void) {
  const uint32_t dim = 32U;
  const uint32_t n = 400U;
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);
  make_clusters(vecs, 8U, 50U, dim, 8801ULL);

  nbase_lsh_config_t cfg = nbase_lsh_config_default(dim);
  cfg.metric = NBASE_METRIC_COSINE;
  cfg.num_tables = 10U;
  cfg.num_bits = 8U;
  cfg.seed = 9001ULL;

  nbase_lsh_t *lsh = NULL;
  ASSERT_STATUS(nbase_lsh_create(&cfg, &lsh), NBASE_OK);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = 100ULL + i;
    ASSERT_STATUS(nbase_lsh_add(lsh, ids[i], vecs + (size_t)i * dim, dim), NBASE_OK);
  }

  nbase_lsh_stats_t st;
  ASSERT_STATUS(nbase_lsh_stats(lsh, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, n);

  nbase_result_t oracle[10];
  nbase_result_t hits[10];
  uint32_t on = 0U;
  uint32_t hn = 0U;
  test_brute_force(ids, vecs, NULL, n, dim, vecs, 10U, NBASE_METRIC_COSINE, NULL, NULL, oracle, &on);

  ASSERT_STATUS(nbase_lsh_search(lsh, vecs, dim, 10U, 2U, hits, &hn), NBASE_OK);
  ASSERT_TRUE(hn >= 5U);
  ASSERT_EQ_U64(hits[0].id, 100ULL);
  ASSERT_TRUE(test_recall_at(hits, hn, oracle, on, 10U) >= 0.7);

  /* Delete id 100 and confirm it no longer appears. */
  ASSERT_STATUS(nbase_lsh_delete(lsh, 100ULL), NBASE_OK);
  ASSERT_STATUS(nbase_lsh_delete(lsh, 100ULL), NBASE_ERR_NOT_FOUND);
  ASSERT_STATUS(nbase_lsh_search(lsh, vecs, dim, 10U, 2U, hits, &hn), NBASE_OK);
  for (uint32_t i = 0U; i < hn; i++) {
    ASSERT_TRUE(hits[i].id != 100ULL);
  }

  ASSERT_STATUS(
      nbase_lsh_search_filtered(lsh, vecs, dim, 10U, 2U, lsh_even_filter, NULL, hits, &hn),
      NBASE_OK);
  ASSERT_TRUE(hn > 0U);
  for (uint32_t i = 0U; i < hn; i++) {
    ASSERT_TRUE((hits[i].id % 2ULL) == 0ULL);
  }

  nbase_lsh_destroy(lsh);
  free(vecs);
  free(ids);
}

static void test_lsh_euclidean_save_load_and_errors(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  char path[300];
  snprintf(path, sizeof(path), "%s/index.nblh", dir);

  const uint32_t dim = 16U;
  const uint32_t n = 200U;
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);
  make_clusters(vecs, 5U, 40U, dim, 8802ULL);

  nbase_lsh_config_t cfg = nbase_lsh_config_default(dim);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.num_tables = 8U;
  cfg.num_bits = 8U;
  cfg.bucket_width = 4.0f;
  cfg.seed = 9002ULL;

  nbase_lsh_t *lsh = NULL;
  ASSERT_STATUS(nbase_lsh_create(&cfg, &lsh), NBASE_OK);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = i + 1ULL;
    ASSERT_STATUS(nbase_lsh_add(lsh, ids[i], vecs + (size_t)i * dim, dim), NBASE_OK);
  }

  nbase_result_t h1[5];
  uint32_t c1 = 0U;
  ASSERT_STATUS(nbase_lsh_search(lsh, vecs + dim, dim, 5U, 2U, h1, &c1), NBASE_OK);
  ASSERT_TRUE(c1 > 0U);
  ASSERT_EQ_U64(h1[0].id, 2ULL);

  ASSERT_STATUS(nbase_lsh_save(lsh, path), NBASE_OK);
  nbase_lsh_destroy(lsh);

  nbase_lsh_t *loaded = NULL;
  ASSERT_STATUS(nbase_lsh_load(path, &loaded), NBASE_OK);
  nbase_result_t h2[5];
  uint32_t c2 = 0U;
  ASSERT_STATUS(nbase_lsh_search(loaded, vecs + dim, dim, 5U, 2U, h2, &c2), NBASE_OK);
  ASSERT_EQ_U64(c2, c1);
  for (uint32_t i = 0U; i < c1; i++) {
    ASSERT_EQ_U64(h2[i].id, h1[i].id);
  }
  nbase_lsh_destroy(loaded);

  ASSERT_TRUE(test_flip_file_byte(path, 140L));
  ASSERT_STATUS(nbase_lsh_load(path, &loaded), NBASE_ERR_CORRUPT);

  nbase_lsh_config_t bad = nbase_lsh_config_default(dim);
  bad.metric = NBASE_METRIC_HAMMING;
  nbase_lsh_t *bad_lsh = NULL;
  ASSERT_STATUS(nbase_lsh_create(&bad, &bad_lsh), NBASE_ERR_UNSUPPORTED);

  free(vecs);
  free(ids);
  test_rm_rf(dir);
}

typedef struct lsh_tsan_ctx {
  nbase_lsh_t *lsh;
  const float *query;
  uint32_t dim;
  atomic_bool stop;
  atomic_int errors;
} lsh_tsan_ctx_t;

static void *lsh_reader_fn(void *raw) {
  lsh_tsan_ctx_t *ctx = (lsh_tsan_ctx_t *)raw;
  for (uint32_t i = 0U; i < 40U && !atomic_load_explicit(&ctx->stop, memory_order_relaxed); i++) {
    nbase_result_t hits[5];
    uint32_t got = 0U;
    if (nbase_lsh_search_filtered(ctx->lsh, ctx->query, ctx->dim, 5U, 2U, lsh_even_filter, NULL, hits, &got) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    nbase_lsh_stats_t st;
    if (nbase_lsh_stats(ctx->lsh, &st) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    (void)nbase_lsh_contains(ctx->lsh, 100ULL);
  }
  return NULL;
}

static void test_lsh_add_batch_contains_and_concurrent_tsan(void) {
  const uint32_t dim = 16U;
  nbase_lsh_config_t cfg = nbase_lsh_config_default(dim);
  cfg.metric = NBASE_METRIC_COSINE;
  cfg.num_tables = 6U;
  cfg.num_bits = 8U;
  nbase_lsh_t *lsh = NULL;
  ASSERT_STATUS(nbase_lsh_create(&cfg, &lsh), NBASE_OK);

  const uint32_t n = 120U;
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);
  test_fill_vectors(vecs, n, dim, 5500ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = (uint64_t)i + 1ULL;
  }

  /* Atomicity: duplicate within batch rejects without inserting any ID. */
  uint64_t dup_ids[3] = {1ULL, 2ULL, 1ULL};
  ASSERT_STATUS(nbase_lsh_add_batch(lsh, dup_ids, vecs, 3U, dim), NBASE_ERR_ALREADY_EXISTS);
  ASSERT_TRUE(!nbase_lsh_contains(lsh, 1ULL));

  ASSERT_STATUS(nbase_lsh_add_batch(lsh, ids, vecs, 80U, dim), NBASE_OK);
  ASSERT_TRUE(nbase_lsh_contains(lsh, 1ULL));
  ASSERT_TRUE(!nbase_lsh_contains(lsh, 81ULL));

  /* Existing ID inside batch rejects cleanly without adding 81. */
  uint64_t overlap_ids[2] = {81ULL, 1ULL};
  ASSERT_STATUS(nbase_lsh_add_batch(lsh, overlap_ids, vecs, 2U, dim), NBASE_ERR_ALREADY_EXISTS);
  ASSERT_TRUE(!nbase_lsh_contains(lsh, 81ULL));

  lsh_tsan_ctx_t ctx = {lsh, vecs, dim, false, 0};
  pthread_t r1, r2;
  ASSERT_TRUE(pthread_create(&r1, NULL, lsh_reader_fn, &ctx) == 0);
  ASSERT_TRUE(pthread_create(&r2, NULL, lsh_reader_fn, &ctx) == 0);

  for (uint32_t i = 80U; i < n; i += 4U) {
    ASSERT_STATUS(nbase_lsh_add_batch(lsh, ids + i, vecs + (size_t)i * dim, 4U, dim), NBASE_OK);
  }
  atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
  pthread_join(r1, NULL);
  pthread_join(r2, NULL);
  ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);

  nbase_lsh_stats_t st;
  ASSERT_STATUS(nbase_lsh_stats(lsh, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, n);

  free(vecs);
  free(ids);
  nbase_lsh_destroy(lsh);
}

typedef struct lsh_batch_tsan_ctx {
  nbase_lsh_t *lsh;
  const float *queries;
  uint32_t nqueries;
  uint32_t dim;
  atomic_bool stop;
  atomic_int errors;
} lsh_batch_tsan_ctx_t;

static void *lsh_batch_reader_fn(void *raw) {
  lsh_batch_tsan_ctx_t *ctx = (lsh_batch_tsan_ctx_t *)raw;
  nbase_result_t res[4U * 5U];
  uint32_t cnts[4U];
  while (!atomic_load_explicit(&ctx->stop, memory_order_relaxed)) {
    if (nbase_lsh_search_batch(ctx->lsh, ctx->queries, 4U, ctx->dim, 5U, 2U, 2U, res, cnts) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
  }
  return NULL;
}

static void test_lsh_search_batch_mt_and_parity(void) {
  static const nbase_metric_t metrics[2] = {NBASE_METRIC_COSINE, NBASE_METRIC_EUCLIDEAN};
  const uint32_t dim = 24U;
  const uint32_t n = 200U;
  const uint32_t nqueries = 12U;
  const uint32_t k = 6U;

  float *vecs = malloc((size_t)n * dim * sizeof(float));
  float *queries = malloc((size_t)nqueries * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  nbase_result_t *batch_res = malloc((size_t)nqueries * k * sizeof(nbase_result_t));
  uint32_t *batch_cnts = malloc((size_t)nqueries * sizeof(uint32_t));
  ASSERT_TRUE(vecs && queries && ids && batch_res && batch_cnts);

  make_clusters(vecs, 5U, 40U, dim, 6611ULL);
  make_clusters(queries, 3U, 4U, dim, 6622ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = 1000ULL + i;
  }

  for (size_t mi = 0; mi < 2; mi++) {
    nbase_lsh_config_t cfg = nbase_lsh_config_default(dim);
    cfg.metric = metrics[mi];
    cfg.num_tables = 8U;
    cfg.num_bits = 8U;
    cfg.bucket_width = 4.0f;
    cfg.seed = 4242ULL + mi;

    nbase_lsh_t *lsh = NULL;
    ASSERT_STATUS(nbase_lsh_create(&cfg, &lsh), NBASE_OK);
    ASSERT_STATUS(nbase_lsh_add_batch(lsh, ids, vecs, n, dim), NBASE_OK);

    /* Compare single-thread and 4-thread nbase_lsh_search_batch vs nbase_lsh_search */
    const uint32_t threads[2] = {1U, 4U};
    for (size_t ti = 0; ti < 2; ti++) {
      ASSERT_STATUS(
          nbase_lsh_search_batch(lsh, queries, nqueries, dim, k, 2U, threads[ti], batch_res, batch_cnts),
          NBASE_OK);
      for (uint32_t q = 0U; q < nqueries; q++) {
        nbase_result_t single[6];
        uint32_t sc = 0U;
        ASSERT_STATUS(
            nbase_lsh_search(lsh, queries + (size_t)q * dim, dim, k, 2U, single, &sc),
            NBASE_OK);
        ASSERT_EQ_U64(batch_cnts[q], sc);
        for (uint32_t r = 0U; r < sc; r++) {
          ASSERT_EQ_U64(batch_res[(size_t)q * k + r].id, single[r].id);
          ASSERT_NEAR(batch_res[(size_t)q * k + r].distance, single[r].distance, 1e-6, 1e-6);
        }
      }
    }

    /* Validation & edge paths */
    ASSERT_STATUS(nbase_lsh_search_batch(lsh, NULL, 0U, dim, k, 2U, 2U, NULL, NULL), NBASE_OK);
    batch_cnts[0] = 99U;
    ASSERT_STATUS(nbase_lsh_search_batch(lsh, queries, 2U, dim, 0U, 2U, 2U, NULL, batch_cnts), NBASE_OK);
    ASSERT_EQ_U64(batch_cnts[0], 0ULL);
    ASSERT_STATUS(nbase_lsh_search_batch(lsh, queries, 2U, dim - 1U, k, 2U, 2U, batch_res, batch_cnts), NBASE_ERR_DIM_MISMATCH);
    queries[dim + 3U] = NAN;
    ASSERT_STATUS(nbase_lsh_search_batch(lsh, queries, 2U, dim, k, 2U, 2U, batch_res, batch_cnts), NBASE_ERR_INVALID_VECTOR);
    queries[dim + 3U] = 0.25f;

    if (mi == 0U) {
      lsh_batch_tsan_ctx_t ctx = {lsh, queries, nqueries, dim, false, 0};
      pthread_t r1, r2;
      ASSERT_TRUE(pthread_create(&r1, NULL, lsh_batch_reader_fn, &ctx) == 0);
      ASSERT_TRUE(pthread_create(&r2, NULL, lsh_batch_reader_fn, &ctx) == 0);
      for (uint32_t s = 0U; s < 16U; s++) {
        ASSERT_STATUS(
            nbase_lsh_add(lsh, 5000ULL + s, vecs + (size_t)s * dim, dim),
            NBASE_OK);
      }
      atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
      pthread_join(r1, NULL);
      pthread_join(r2, NULL);
      ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);
    }

    nbase_lsh_destroy(lsh);
  }

  free(batch_cnts);
  free(batch_res);
  free(ids);
  free(queries);
  free(vecs);
}

static void *lsh_crud_reader_fn(void *raw) {
  lsh_tsan_ctx_t *ctx = (lsh_tsan_ctx_t *)raw;
  float vbuf[16];
  uint64_t ids_buf[8];
  nbase_result_t res[5];
  while (!atomic_load_explicit(&ctx->stop, memory_order_relaxed)) {
    uint32_t cnt = 0U;
    uint64_t total = 0ULL;
    if (nbase_lsh_list_ids(ctx->lsh, 0ULL, 8U, ids_buf, &cnt, &total) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    if (cnt > 0U) {
      nbase_status_t gst = nbase_lsh_get(ctx->lsh, ids_buf[0], vbuf, ctx->dim);
      if (gst != NBASE_OK && gst != NBASE_ERR_NOT_FOUND) {
        atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
        break;
      }
    }
    uint32_t hn = 0U;
    if (nbase_lsh_search(ctx->lsh, ctx->query, ctx->dim, 5U, 2U, res, &hn) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
  }
  return NULL;
}

static void test_lsh_upsert_get_list_compact_and_tsan(void) {
  const uint32_t dim = 16U;
  nbase_lsh_config_t cfg = nbase_lsh_config_default(dim);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.num_tables = 6U;
  cfg.num_bits = 8U;
  cfg.bucket_width = 3.0f;

  nbase_lsh_t *lsh = NULL;
  ASSERT_STATUS(nbase_lsh_create(&cfg, &lsh), NBASE_OK);

  const uint32_t n = 40U;
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  ASSERT_TRUE(vecs && ids);
  test_fill_vectors(vecs, n, dim, 7700ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = 1000ULL - i * 10ULL; /* deliberately descending order */
  }
  ASSERT_STATUS(nbase_lsh_add_batch(lsh, ids, vecs, n, dim), NBASE_OK);

  /* Upsert overwrite + invalid NaN preserving existing vector */
  float replacement[16];
  for (uint32_t d = 0U; d < dim; d++) {
    replacement[d] = (d == 0U) ? 9.5f : 0.0f;
  }
  ASSERT_STATUS(nbase_lsh_upsert(lsh, 1000ULL, replacement, dim), NBASE_OK);
  replacement[1] = NAN;
  ASSERT_STATUS(nbase_lsh_upsert(lsh, 1000ULL, replacement, dim), NBASE_ERR_INVALID_VECTOR);

  float got[16] = {0};
  ASSERT_STATUS(nbase_lsh_get(lsh, 1000ULL, got, dim), NBASE_OK);
  ASSERT_NEAR(got[0], 9.5, 1e-6, 0.0);
  ASSERT_NEAR(got[1], 0.0, 1e-6, 0.0);
  ASSERT_STATUS(nbase_lsh_get(lsh, 1000ULL, got, dim - 1U), NBASE_ERR_DIM_MISMATCH);
  ASSERT_STATUS(nbase_lsh_get(lsh, 999999ULL, got, dim), NBASE_ERR_NOT_FOUND);

  /* Sorted paginated list_ids: smallest ID is 610 */
  uint64_t page[5];
  uint32_t pcnt = 0U;
  uint64_t total = 0ULL;
  ASSERT_STATUS(nbase_lsh_list_ids(lsh, 0ULL, 5U, page, &pcnt, &total), NBASE_OK);
  ASSERT_EQ_U64(total, 40ULL);
  ASSERT_EQ_U64(pcnt, 5ULL);
  ASSERT_EQ_U64(page[0], 610ULL);
  ASSERT_EQ_U64(page[1], 620ULL);
  ASSERT_STATUS(nbase_lsh_list_ids(lsh, 100ULL, 5U, page, &pcnt, &total), NBASE_OK);
  ASSERT_EQ_U64(pcnt, 0ULL);

  /* Delete 30 items and verify compact preserves live vectors & search */
  for (uint32_t i = 5U; i < 35U; i++) {
    ASSERT_STATUS(nbase_lsh_delete(lsh, ids[i]), NBASE_OK);
  }
  ASSERT_STATUS(nbase_lsh_compact(lsh), NBASE_OK);
  ASSERT_STATUS(nbase_lsh_compact(lsh), NBASE_OK);
  ASSERT_STATUS(nbase_lsh_compact(NULL), NBASE_ERR_INVALID_ARG);

  nbase_lsh_stats_t st;
  ASSERT_STATUS(nbase_lsh_stats(lsh, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, 10ULL);
  ASSERT_STATUS(nbase_lsh_get(lsh, 1000ULL, got, dim), NBASE_OK);
  ASSERT_NEAR(got[0], 9.5, 1e-6, 0.0);

  nbase_result_t hits[5];
  uint32_t hn = 0U;
  replacement[1] = 0.0f;
  ASSERT_STATUS(nbase_lsh_search(lsh, replacement, dim, 5U, 4U, hits, &hn), NBASE_OK);
  ASSERT_TRUE(hn >= 1U);
  ASSERT_EQ_U64(hits[0].id, 1000ULL);

  /* Concurrent readers (list_ids + get + search) vs upsert writer under TSan */
  lsh_tsan_ctx_t ctx = {lsh, replacement, dim, false, 0};
  pthread_t r1, r2;
  ASSERT_TRUE(pthread_create(&r1, NULL, lsh_crud_reader_fn, &ctx) == 0);
  ASSERT_TRUE(pthread_create(&r2, NULL, lsh_crud_reader_fn, &ctx) == 0);
  for (uint32_t step = 0U; step < 25U; step++) {
    float uv[16] = {0.0f};
    uv[step % dim] = 1.0f + (float)step * 0.05f;
    ASSERT_STATUS(nbase_lsh_upsert(lsh, 2000ULL + (step % 5ULL), uv, dim), NBASE_OK);
  }
  atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
  pthread_join(r1, NULL);
  pthread_join(r2, NULL);
  ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);

  /* Save/load post-compact round-trip */
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  char path[300];
  snprintf(path, sizeof(path), "%s/compacted.nblh", dir);
  ASSERT_STATUS(nbase_lsh_save(lsh, path), NBASE_OK);
  nbase_lsh_t *loaded = NULL;
  ASSERT_STATUS(nbase_lsh_load(path, &loaded), NBASE_OK);
  ASSERT_TRUE(nbase_lsh_contains(loaded, 1000ULL));
  nbase_lsh_destroy(loaded);
  test_rm_rf(dir);

  free(ids);
  free(vecs);
  nbase_lsh_destroy(lsh);
}

static void test_lsh_upsert_delete_get_batch_and_tsan(void) {
  const uint32_t dim = 16U;
  nbase_lsh_config_t cfg = nbase_lsh_config_default(dim);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.num_tables = 6U;
  cfg.num_bits = 8U;
  cfg.seed = 4242ULL;

  nbase_lsh_t *lsh = NULL;
  ASSERT_STATUS(nbase_lsh_create(&cfg, &lsh), NBASE_OK);

  const uint32_t init_n = 6U;
  uint64_t ids[6] = {5000ULL, 5001ULL, 5002ULL, 5003ULL, 5004ULL, 5005ULL};
  float *vecs = calloc((size_t)init_n * (size_t)dim, sizeof(float));
  ASSERT_TRUE(vecs != NULL);
  for (uint32_t i = 0U; i < init_n; i++) {
    vecs[(size_t)i * (size_t)dim + (i % dim)] = (float)(i + 1U);
  }

  ASSERT_STATUS(nbase_lsh_upsert_batch(lsh, NULL, NULL, 0U, dim), NBASE_OK);
  ASSERT_STATUS(nbase_lsh_upsert_batch(NULL, ids, vecs, init_n, dim), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_lsh_upsert_batch(lsh, ids, vecs, init_n, dim + 1U), NBASE_ERR_DIM_MISMATCH);
  ASSERT_STATUS(nbase_lsh_upsert_batch(lsh, ids, vecs, init_n, dim), NBASE_OK);

  /* Reject duplicate IDs within one batch cleanly, then upsert mixed overwrite + new IDs. */
  uint64_t dup_ids[4] = {5001ULL, 5006ULL, 5001ULL, 5007ULL};
  uint64_t up_ids[3] = {5001ULL, 5006ULL, 5007ULL};
  float *up_vecs = calloc((size_t)4U * (size_t)dim, sizeof(float));
  ASSERT_TRUE(up_vecs != NULL);
  up_vecs[0U * dim + 0U] = 13.5f;
  up_vecs[1U * dim + 1U] = 12.0f;
  up_vecs[2U * dim + 2U] = 14.0f;
  ASSERT_STATUS(nbase_lsh_upsert_batch(lsh, dup_ids, up_vecs, 4U, dim), NBASE_ERR_ALREADY_EXISTS);
  ASSERT_STATUS(nbase_lsh_upsert_batch(lsh, up_ids, up_vecs, 3U, dim), NBASE_OK);

  nbase_lsh_stats_t st;
  ASSERT_STATUS(nbase_lsh_stats(lsh, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, 8ULL);

  /* get_batch with mix of valid, missing, and duplicate IDs. */
  uint64_t get_ids[4] = {5001ULL, 999999ULL, 5006ULL, 5007ULL};
  float *got_vecs = calloc((size_t)4U * (size_t)dim, sizeof(float));
  bool got_found[4] = {false, true, false, false};
  ASSERT_TRUE(got_vecs != NULL);
  ASSERT_STATUS(nbase_lsh_get_batch(lsh, NULL, 0U, dim, NULL, NULL), NBASE_OK);
  ASSERT_STATUS(nbase_lsh_get_batch(lsh, get_ids, 4U, dim + 1U, got_vecs, got_found), NBASE_ERR_DIM_MISMATCH);
  ASSERT_STATUS(nbase_lsh_get_batch(lsh, get_ids, 4U, dim, got_vecs, got_found), NBASE_OK);
  ASSERT_TRUE(got_found[0] == true);
  ASSERT_TRUE(got_found[1] == false);
  ASSERT_TRUE(got_found[2] == true);
  ASSERT_TRUE(got_found[3] == true);
  ASSERT_NEAR(got_vecs[0U * dim + 0U], 13.5, 1e-6, 0.0);
  ASSERT_NEAR(got_vecs[2U * dim + 1U], 12.0, 1e-6, 0.0);
  ASSERT_NEAR(got_vecs[3U * dim + 2U], 14.0, 1e-6, 0.0);

  /* delete_batch with missing + duplicate IDs. */
  uint64_t del_ids[4] = {5002ULL, 888888ULL, 5002ULL, 5004ULL};
  uint32_t del_cnt = 99U;
  ASSERT_STATUS(nbase_lsh_delete_batch(lsh, NULL, 0U, &del_cnt), NBASE_OK);
  ASSERT_EQ_U64(del_cnt, 0ULL);
  ASSERT_STATUS(nbase_lsh_delete_batch(NULL, del_ids, 4U, &del_cnt), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_lsh_delete_batch(lsh, del_ids, 4U, &del_cnt), NBASE_OK);
  ASSERT_EQ_U64(del_cnt, 2ULL);
  ASSERT_TRUE(!nbase_lsh_contains(lsh, 5002ULL));
  ASSERT_TRUE(!nbase_lsh_contains(lsh, 5004ULL));
  ASSERT_TRUE(nbase_lsh_contains(lsh, 5001ULL));
  ASSERT_STATUS(nbase_lsh_stats(lsh, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, 6ULL);

  /* Concurrent readers vs upsert_batch/delete_batch under TSan. */
  float qvec[16] = {0.0f};
  qvec[0] = 13.5f;
  lsh_tsan_ctx_t ctx = {lsh, qvec, dim, false, 0};
  pthread_t r1, r2;
  ASSERT_TRUE(pthread_create(&r1, NULL, lsh_crud_reader_fn, &ctx) == 0);
  ASSERT_TRUE(pthread_create(&r2, NULL, lsh_crud_reader_fn, &ctx) == 0);
  for (uint32_t step = 0U; step < 12U; step++) {
    uint64_t bid[2] = {6000ULL + (step % 3ULL), 6100ULL + (step % 3ULL)};
    float bv[32] = {0.0f};
    bv[step % dim] = 2.0f + (float)step * 0.1f;
    bv[dim + ((step + 1U) % dim)] = 3.0f;
    ASSERT_STATUS(nbase_lsh_upsert_batch(lsh, bid, bv, 2U, dim), NBASE_OK);
    if ((step % 3U) == 2U) {
      ASSERT_STATUS(nbase_lsh_delete_batch(lsh, bid, 2U, &del_cnt), NBASE_OK);
    }
  }
  atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
  pthread_join(r1, NULL);
  pthread_join(r2, NULL);
  ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);

  free(got_vecs);
  free(up_vecs);
  free(vecs);
  nbase_lsh_destroy(lsh);
}

typedef struct lsh_radius_tsan_ctx {
  nbase_lsh_t *lsh;
  const float *queries;
  uint32_t dim;
  atomic_bool stop;
  atomic_int errors;
} lsh_radius_tsan_ctx_t;

static void *lsh_radius_batch_reader_fn(void *raw) {
  lsh_radius_tsan_ctx_t *ctx = (lsh_radius_tsan_ctx_t *)raw;
  nbase_result_t r_hits[16];
  nbase_result_t b_hits[4U * 8U];
  uint32_t b_cnts[4];
  while (!atomic_load_explicit(&ctx->stop, memory_order_relaxed)) {
    uint32_t got = 0U;
    if (nbase_lsh_search_radius(
            ctx->lsh, ctx->queries, ctx->dim, 50.0f, 16U, 2U, NULL, NULL, r_hits, &got) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    if (nbase_lsh_search_batch(ctx->lsh, ctx->queries, 4U, ctx->dim, 8U, 2U, 2U, b_hits, b_cnts) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
  }
  return NULL;
}

static void test_lsh_search_radius_and_batch_scratch_tsan(void) {
  const uint32_t dim = 16U;
  const uint32_t n = 120U;
  const uint32_t qcnt = 80U;
  float *vecs = calloc((size_t)n * (size_t)dim, sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  float *queries = calloc((size_t)qcnt * (size_t)dim, sizeof(float));
  nbase_result_t *batch_res = malloc((size_t)qcnt * 8U * sizeof(nbase_result_t));
  uint32_t *batch_cnts = malloc((size_t)qcnt * sizeof(uint32_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL && queries != NULL && batch_res != NULL && batch_cnts != NULL);

  make_clusters(vecs, 6U, 20U, dim, 5151ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = 1000ULL + (uint64_t)i;
  }
  test_fill_vectors(queries, qcnt, dim, 6161ULL);

  static const nbase_metric_t metrics[2] = {NBASE_METRIC_COSINE, NBASE_METRIC_EUCLIDEAN};
  for (size_t mi = 0U; mi < 2U; mi++) {
    nbase_lsh_config_t cfg = nbase_lsh_config_default(dim);
    cfg.metric = metrics[mi];
    cfg.num_tables = 8U;
    cfg.num_bits = 8U;
    cfg.bucket_width = 4.0f;
    cfg.seed = 7070ULL + (uint64_t)mi;

    nbase_lsh_t *lsh = NULL;
    ASSERT_STATUS(nbase_lsh_create(&cfg, &lsh), NBASE_OK);
    ASSERT_STATUS(nbase_lsh_add_batch(lsh, ids, vecs, n, dim), NBASE_OK);

    nbase_result_t hits[16];
    uint32_t got = 77U;
    ASSERT_STATUS(
        nbase_lsh_search_radius(NULL, queries, dim, 1.0f, 8U, 2U, NULL, NULL, hits, &got),
        NBASE_ERR_INVALID_ARG);
    ASSERT_STATUS(
        nbase_lsh_search_radius(lsh, queries, dim, NAN, 8U, 2U, NULL, NULL, hits, &got),
        NBASE_ERR_INVALID_ARG);
    ASSERT_STATUS(
        nbase_lsh_search_radius(lsh, queries, dim, INFINITY, 8U, 2U, NULL, NULL, hits, &got),
        NBASE_ERR_INVALID_ARG);
    ASSERT_STATUS(
        nbase_lsh_search_radius(lsh, queries, dim + 1U, 1.0f, 8U, 2U, NULL, NULL, hits, &got),
        NBASE_ERR_DIM_MISMATCH);
    ASSERT_STATUS(
        nbase_lsh_search_radius(lsh, queries, dim, 1.0f, 0U, 2U, NULL, NULL, NULL, &got),
        NBASE_OK);
    ASSERT_EQ_U64(got, 0ULL);

    ASSERT_STATUS(
        nbase_lsh_search_radius(lsh, vecs, dim, -0.01f, 8U, 2U, NULL, NULL, hits, &got),
        NBASE_OK);
    ASSERT_EQ_U64(got, 0ULL);

    float rad = (metrics[mi] == NBASE_METRIC_COSINE) ? 0.25f : 1.5f;
    ASSERT_STATUS(
        nbase_lsh_search_radius(lsh, vecs, dim, rad, 16U, 4U, NULL, NULL, hits, &got),
        NBASE_OK);
    ASSERT_TRUE(got >= 1U);
    ASSERT_EQ_U64(hits[0].id, 1000ULL);
    for (uint32_t j = 0U; j < got; j++) {
      ASSERT_TRUE(hits[j].distance <= rad);
      if (j > 0U) {
        ASSERT_TRUE(hits[j - 1U].distance <= hits[j].distance);
      }
    }

    ASSERT_STATUS(
        nbase_lsh_search_radius(
            lsh, vecs, dim, rad, 16U, 4U, lsh_even_filter, NULL, hits, &got),
        NBASE_OK);
    for (uint32_t j = 0U; j < got; j++) {
      ASSERT_TRUE((hits[j].id % 2ULL) == 0ULL);
      ASSERT_TRUE(hits[j].distance <= rad);
    }

    /* 80-query multi-threaded batch parity exercising per-worker generation-stamped scratch */
    ASSERT_STATUS(
        nbase_lsh_search_batch(lsh, queries, qcnt, dim, 8U, 2U, 4U, batch_res, batch_cnts),
        NBASE_OK);
    for (uint32_t q = 0U; q < qcnt; q++) {
      nbase_result_t serial_res[8];
      uint32_t serial_cnt = 0U;
      ASSERT_STATUS(
          nbase_lsh_search(lsh, queries + (size_t)q * (size_t)dim, dim, 8U, 2U, serial_res, &serial_cnt),
          NBASE_OK);
      ASSERT_EQ_U64(batch_cnts[q], serial_cnt);
      for (uint32_t j = 0U; j < serial_cnt; j++) {
        ASSERT_EQ_U64(batch_res[(size_t)q * 8U + j].id, serial_res[j].id);
        ASSERT_NEAR(batch_res[(size_t)q * 8U + j].distance, serial_res[j].distance, 1e-6, 0.0);
      }
    }

    if (mi == 1U) {
      lsh_radius_tsan_ctx_t ctx = {lsh, queries, dim, false, 0};
      pthread_t r1, r2;
      ASSERT_TRUE(pthread_create(&r1, NULL, lsh_radius_batch_reader_fn, &ctx) == 0);
      ASSERT_TRUE(pthread_create(&r2, NULL, lsh_radius_batch_reader_fn, &ctx) == 0);
      for (uint32_t step = 0U; step < 8U; step++) {
        uint64_t up_id = 1000ULL + (step % 5ULL);
        ASSERT_STATUS(nbase_lsh_upsert_batch(lsh, &up_id, vecs + (size_t)step * dim, 1U, dim), NBASE_OK);
      }
      atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
      pthread_join(r1, NULL);
      pthread_join(r2, NULL);
      ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);
    }

    nbase_lsh_destroy(lsh);
  }

  free(batch_cnts);
  free(batch_res);
  free(queries);
  free(ids);
  free(vecs);
}

typedef struct lsh_sim_tsan_ctx {
  const nbase_lsh_t *lsh;
  atomic_bool stop;
  atomic_int errors;
} lsh_sim_tsan_ctx_t;

static void *lsh_sim_reader_fn(void *raw) {
  lsh_sim_tsan_ctx_t *ctx = (lsh_sim_tsan_ctx_t *)raw;
  while (!atomic_load_explicit(&ctx->stop, memory_order_relaxed)) {
    nbase_result_t out[4];
    uint32_t got = 0U;
    if (nbase_lsh_find_similar(ctx->lsh, 500ULL, 4U, 2U, NULL, NULL, out, &got) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    for (uint32_t i = 0U; i < got; i++) {
      if (out[i].id == 500ULL) {
        atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      }
    }
  }
  return NULL;
}

static void test_lsh_find_similar_and_tsan(void) {
  const uint32_t dim = 16U;
  const uint32_t n = 40U;
  uint64_t *ids = malloc(n * sizeof(uint64_t));
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  ASSERT_TRUE(ids != NULL && vecs != NULL);
  test_fill_vectors(vecs, n, dim, 6218ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = 500ULL + i;
  }
  /* Guarantee 501 (odd) and 502 (even) are nearest neighbors of 500 in both cosine & L2. */
  for (uint32_t d = 0U; d < dim; d++) {
    vecs[1U * dim + d] = vecs[d] * 1.01f;
    vecs[2U * dim + d] = vecs[d] * 1.03f;
  }

  for (uint32_t m = 0U; m < 2U; m++) {
    nbase_lsh_config_t cfg = nbase_lsh_config_default(dim);
    cfg.metric = (m == 0U) ? NBASE_METRIC_COSINE : NBASE_METRIC_EUCLIDEAN;
    cfg.num_tables = 8U;
    cfg.num_bits = 8U;
    cfg.seed = 6318ULL + (unsigned long long)m;

    nbase_lsh_t *lsh = NULL;
    ASSERT_STATUS(nbase_lsh_create(&cfg, &lsh), NBASE_OK);
    ASSERT_STATUS(nbase_lsh_add_batch(lsh, ids, vecs, n, dim), NBASE_OK);

    nbase_result_t res[5];
    uint32_t got = 0U;
    ASSERT_STATUS(nbase_lsh_find_similar(NULL, 500ULL, 5U, 2U, NULL, NULL, res, &got), NBASE_ERR_INVALID_ARG);
    ASSERT_STATUS(nbase_lsh_find_similar(lsh, 500ULL, 5U, 2U, NULL, NULL, NULL, &got), NBASE_ERR_INVALID_ARG);
    ASSERT_STATUS(nbase_lsh_find_similar(lsh, 500ULL, 5U, 2U, NULL, NULL, res, NULL), NBASE_ERR_INVALID_ARG);
    ASSERT_STATUS(nbase_lsh_find_similar(lsh, 9999ULL, 5U, 2U, NULL, NULL, res, &got), NBASE_ERR_NOT_FOUND);
    ASSERT_STATUS(nbase_lsh_find_similar(lsh, 500ULL, 0U, 2U, NULL, NULL, NULL, &got), NBASE_OK);
    ASSERT_EQ_U64(got, 0ULL);

    ASSERT_STATUS(nbase_lsh_find_similar(lsh, 500ULL, 5U, 2U, NULL, NULL, res, &got), NBASE_OK);
    ASSERT_TRUE(got >= 2U);
    ASSERT_EQ_U64(res[0].id, 501ULL);
    for (uint32_t i = 0U; i < got; i++) {
      ASSERT_TRUE(res[i].id != 500ULL);
    }

    ASSERT_STATUS(nbase_lsh_find_similar(lsh, 500ULL, 4U, 2U, lsh_even_filter, NULL, res, &got), NBASE_OK);
    ASSERT_TRUE(got >= 1U);
    ASSERT_EQ_U64(res[0].id, 502ULL);
    for (uint32_t i = 0U; i < got; i++) {
      ASSERT_TRUE((res[i].id % 2ULL) == 0ULL && res[i].id != 500ULL);
    }

    if (m == 0U) {
      lsh_sim_tsan_ctx_t ctx;
      ctx.lsh = lsh;
      atomic_init(&ctx.stop, false);
      atomic_init(&ctx.errors, 0);
      pthread_t r1, r2;
      ASSERT_TRUE(pthread_create(&r1, NULL, lsh_sim_reader_fn, &ctx) == 0);
      ASSERT_TRUE(pthread_create(&r2, NULL, lsh_sim_reader_fn, &ctx) == 0);
      for (uint32_t step = 0U; step < 8U; step++) {
        ASSERT_STATUS(
            nbase_lsh_upsert(lsh, 510ULL + (step % 4U), vecs + (size_t)(10U + step) * dim, dim),
            NBASE_OK);
      }
      atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
      pthread_join(r1, NULL);
      pthread_join(r2, NULL);
      ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);
    }

    nbase_lsh_destroy(lsh);
  }

  free(vecs);
  free(ids);
}

typedef struct lsh_flt_batch_tsan_ctx {
  nbase_lsh_t *lsh;
  const float *qs;
  uint32_t dim;
  atomic_bool stop;
  atomic_int errors;
} lsh_flt_batch_tsan_ctx_t;

static void *lsh_flt_batch_reader_fn(void *raw) {
  lsh_flt_batch_tsan_ctx_t *ctx = (lsh_flt_batch_tsan_ctx_t *)raw;
  nbase_result_t res[20];
  uint32_t cnts[4];
  while (!atomic_load_explicit(&ctx->stop, memory_order_relaxed)) {
    if (nbase_lsh_search_batch_filtered(
            ctx->lsh, ctx->qs, 4U, ctx->dim, 5U, 2U, 2U, lsh_even_filter, NULL, res, cnts) !=
        NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    for (uint32_t q = 0U; q < 4U; q++) {
      for (uint32_t i = 0U; i < cnts[q]; i++) {
        if ((res[(size_t)q * 5U + i].id % 2ULL) != 0ULL) {
          atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
          return NULL;
        }
      }
    }
  }
  return NULL;
}

static void test_lsh_search_batch_filtered_and_tsan(void) {
  const uint32_t dim = 16U;
  const uint32_t n = 160U;
  const uint32_t nq = 6U;
  const uint32_t k = 5U;
  static const nbase_metric_t metrics[2] = {NBASE_METRIC_COSINE, NBASE_METRIC_EUCLIDEAN};

  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  float *vecs = malloc((size_t)n * (size_t)dim * sizeof(float));
  float *qs = malloc((size_t)nq * (size_t)dim * sizeof(float));
  nbase_result_t *bres = malloc((size_t)nq * k * sizeof(nbase_result_t));
  uint32_t *bcnts = malloc((size_t)nq * sizeof(uint32_t));
  ASSERT_TRUE(ids && vecs && qs && bres && bcnts);

  make_clusters(vecs, 8U, 20U, dim, 6101ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = 100ULL + i;
  }
  for (uint32_t q = 0U; q < nq; q++) {
    memcpy(qs + (size_t)q * dim, vecs + (size_t)(q * 20U) * dim, (size_t)dim * sizeof(float));
  }

  for (size_t mi = 0; mi < 2; mi++) {
    nbase_lsh_config_t cfg = nbase_lsh_config_default(dim);
    cfg.metric = metrics[mi];
    cfg.num_tables = 6U;
    cfg.num_bits = 8U;
    cfg.seed = 6202ULL + (uint64_t)mi;

    nbase_lsh_t *lsh = NULL;
    ASSERT_STATUS(nbase_lsh_create(&cfg, &lsh), NBASE_OK);
    ASSERT_STATUS(nbase_lsh_add_batch(lsh, ids, vecs, n, dim), NBASE_OK);

    const uint32_t threads[2] = {1U, 4U};
    for (size_t ti = 0; ti < 2; ti++) {
      ASSERT_STATUS(
          nbase_lsh_search_batch_filtered(
              lsh, qs, nq, dim, k, 2U, threads[ti], lsh_even_filter, NULL, bres, bcnts),
          NBASE_OK);
      for (uint32_t q = 0U; q < nq; q++) {
        nbase_result_t sres[5];
        uint32_t scnt = 0U;
        ASSERT_STATUS(
            nbase_lsh_search_filtered(
                lsh, qs + (size_t)q * dim, dim, k, 2U, lsh_even_filter, NULL, sres, &scnt),
            NBASE_OK);
        ASSERT_EQ_U64(bcnts[q], scnt);
        for (uint32_t r = 0U; r < scnt; r++) {
          ASSERT_EQ_U64(bres[(size_t)q * k + r].id, sres[r].id);
          ASSERT_TRUE((sres[r].id % 2ULL) == 0ULL);
          ASSERT_NEAR((double)bres[(size_t)q * k + r].distance, (double)sres[r].distance, 1e-6, 0.0);
        }
      }
    }

    if (mi == 0U) {
      ASSERT_STATUS(
          nbase_lsh_search_batch_filtered(
              NULL, qs, nq, dim, k, 2U, 2U, lsh_even_filter, NULL, bres, bcnts),
          NBASE_ERR_INVALID_ARG);
      ASSERT_STATUS(
          nbase_lsh_search_batch_filtered(
              lsh, NULL, 0U, dim, k, 2U, 2U, lsh_even_filter, NULL, NULL, NULL),
          NBASE_OK);
      bcnts[0] = 66U;
      ASSERT_STATUS(
          nbase_lsh_search_batch_filtered(
              lsh, qs, 2U, dim, 0U, 2U, 2U, lsh_even_filter, NULL, NULL, bcnts),
          NBASE_OK);
      ASSERT_EQ_U64(bcnts[0], 0ULL);
      ASSERT_STATUS(
          nbase_lsh_search_batch_filtered(
              lsh, qs, 2U, dim + 1U, k, 2U, 2U, lsh_even_filter, NULL, bres, bcnts),
          NBASE_ERR_DIM_MISMATCH);

      lsh_flt_batch_tsan_ctx_t ctx = {lsh, qs, dim, false, 0};
      pthread_t r1, r2;
      ASSERT_TRUE(pthread_create(&r1, NULL, lsh_flt_batch_reader_fn, &ctx) == 0);
      ASSERT_TRUE(pthread_create(&r2, NULL, lsh_flt_batch_reader_fn, &ctx) == 0);
      for (uint32_t step = 0U; step < 12U; step++) {
        uint64_t tid = 2000ULL + (step % 4ULL);
        ASSERT_STATUS(nbase_lsh_upsert(lsh, tid, vecs + (size_t)step * dim, dim), NBASE_OK);
        if ((step % 3U) == 2U) {
          ASSERT_STATUS(nbase_lsh_delete(lsh, tid), NBASE_OK);
        }
      }
      atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
      pthread_join(r1, NULL);
      pthread_join(r2, NULL);
      ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);
    }

    nbase_lsh_destroy(lsh);
  }

  free(bcnts);
  free(bres);
  free(qs);
  free(vecs);
  free(ids);
}

typedef struct lsh_sim_batch_tsan_ctx {
  const nbase_lsh_t *lsh;
  const uint64_t *qids;
  uint32_t qcnt;
  atomic_bool stop;
  atomic_int errors;
} lsh_sim_batch_tsan_ctx_t;

static bool lsh_sim_even_filter(uint64_t id, void *ctx) {
  (void)ctx;
  return (id % 2ULL) == 0ULL;
}

static void *lsh_sim_batch_reader_fn(void *raw) {
  lsh_sim_batch_tsan_ctx_t *ctx = (lsh_sim_batch_tsan_ctx_t *)raw;
  nbase_result_t res[16];
  uint32_t cnts[4] = {0};
  while (!atomic_load_explicit(&ctx->stop, memory_order_relaxed)) {
    if (nbase_lsh_find_similar_batch(
            ctx->lsh, ctx->qids, ctx->qcnt, 4U, 2U, 2U, lsh_sim_even_filter, NULL, res, cnts) !=
        NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    for (uint32_t q = 0U; q < ctx->qcnt; q++) {
      for (uint32_t j = 0U; j < cnts[q]; j++) {
        uint64_t rid = res[(size_t)q * 4U + (size_t)j].id;
        if (rid == ctx->qids[q] || (rid % 2ULL) != 0ULL) {
          atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
          break;
        }
      }
    }
  }
  return NULL;
}

static void test_lsh_find_similar_batch_and_tsan(void) {
  const uint32_t n = 80U;
  const uint32_t dim = 16U;
  const uint32_t k = 5U;

  nbase_lsh_config_t cfg = nbase_lsh_config_default(dim);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.num_tables = 6U;
  cfg.num_bits = 8U;
  cfg.seed = 7788ULL;

  nbase_lsh_t *lsh = NULL;
  ASSERT_STATUS(nbase_lsh_create(&cfg, &lsh), NBASE_OK);

  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  float *vecs = malloc((size_t)n * (size_t)dim * sizeof(float));
  ASSERT_TRUE(ids && vecs);
  test_fill_vectors(vecs, n, dim, 7800ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = (i == 0U) ? 0ULL : (400ULL + (uint64_t)i);
  }
  ASSERT_STATUS(nbase_lsh_add_batch(lsh, ids, vecs, n, dim), NBASE_OK);

  const uint64_t qids[4] = {0ULL, 411ULL, 434ULL, 467ULL};
  const uint32_t qcnt = 4U;
  nbase_result_t b1[4U * 5U];
  nbase_result_t b4[4U * 5U];
  nbase_result_t single[5U];
  uint32_t c1[4] = {0};
  uint32_t c4[4] = {0};

  const uint64_t missing[2] = {0ULL, 999999ULL};
  ASSERT_STATUS(
      nbase_lsh_find_similar_batch(lsh, missing, 2U, k, 2U, 2U, NULL, NULL, b1, c1),
      NBASE_ERR_NOT_FOUND);
  ASSERT_STATUS(
      nbase_lsh_find_similar_batch(lsh, NULL, 0U, k, 2U, 2U, NULL, NULL, NULL, NULL),
      NBASE_OK);
  uint32_t zc[2] = {6U, 6U};
  ASSERT_STATUS(
      nbase_lsh_find_similar_batch(lsh, qids, 2U, 0U, 2U, 2U, NULL, NULL, NULL, zc),
      NBASE_OK);
  ASSERT_EQ_U64(zc[0], 0ULL);
  ASSERT_EQ_U64(zc[1], 0ULL);
  ASSERT_STATUS(
      nbase_lsh_find_similar_batch(lsh, qids, 2U, k, 2U, 2U, NULL, NULL, NULL, c1),
      NBASE_ERR_INVALID_ARG);

  ASSERT_STATUS(
      nbase_lsh_find_similar_batch(lsh, qids, qcnt, k, 2U, 1U, lsh_sim_even_filter, NULL, b1, c1),
      NBASE_OK);
  ASSERT_STATUS(
      nbase_lsh_find_similar_batch(lsh, qids, qcnt, k, 2U, 4U, lsh_sim_even_filter, NULL, b4, c4),
      NBASE_OK);
  for (uint32_t q = 0U; q < qcnt; q++) {
    uint32_t sc = 0U;
    ASSERT_STATUS(
        nbase_lsh_find_similar(lsh, qids[q], k, 2U, lsh_sim_even_filter, NULL, single, &sc),
        NBASE_OK);
    ASSERT_EQ_U64(c1[q], sc);
    ASSERT_EQ_U64(c4[q], sc);
    for (uint32_t j = 0U; j < sc; j++) {
      size_t idx = (size_t)q * (size_t)k + (size_t)j;
      ASSERT_TRUE(b1[idx].id != qids[q]);
      ASSERT_EQ_U64(b1[idx].id, single[j].id);
      ASSERT_EQ_U64(b4[idx].id, single[j].id);
      ASSERT_NEAR(b1[idx].distance, single[j].distance, 1e-6, 1e-6);
      ASSERT_EQ_U64(b1[idx].id % 2ULL, 0ULL);
    }
  }

  /* TSan concurrent find_similar_batch readers + upserts/deletes */
  lsh_sim_batch_tsan_ctx_t ctx = {lsh, qids, qcnt, false, 0};
  pthread_t r1, r2;
  ASSERT_TRUE(pthread_create(&r1, NULL, lsh_sim_batch_reader_fn, &ctx) == 0);
  ASSERT_TRUE(pthread_create(&r2, NULL, lsh_sim_batch_reader_fn, &ctx) == 0);
  for (uint32_t step = 0U; step < 12U; step++) {
    uint64_t tid = 2000ULL + (uint64_t)(step % 4U);
    ASSERT_STATUS(nbase_lsh_upsert(lsh, tid, vecs + (size_t)step * (size_t)dim, dim), NBASE_OK);
    if ((step % 3U) == 2U) {
      ASSERT_STATUS(nbase_lsh_delete(lsh, tid), NBASE_OK);
    }
  }
  atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
  pthread_join(r1, NULL);
  pthread_join(r2, NULL);
  ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);

  nbase_lsh_destroy(lsh);
  free(vecs);
  free(ids);
}

void test_lsh_run(void) {
  RUN_TEST(test_lsh_cosine_and_multiprobe);
  RUN_TEST(test_lsh_euclidean_save_load_and_errors);
  RUN_TEST(test_lsh_add_batch_contains_and_concurrent_tsan);
  RUN_TEST(test_lsh_search_batch_mt_and_parity);
  RUN_TEST(test_lsh_upsert_get_list_compact_and_tsan);
  RUN_TEST(test_lsh_upsert_delete_get_batch_and_tsan);
  RUN_TEST(test_lsh_search_radius_and_batch_scratch_tsan);
  RUN_TEST(test_lsh_find_similar_and_tsan);
  RUN_TEST(test_lsh_search_batch_filtered_and_tsan);
  RUN_TEST(test_lsh_find_similar_batch_and_tsan);
}
