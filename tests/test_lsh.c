#include "test_common.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>

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

void test_lsh_run(void) {
  RUN_TEST(test_lsh_cosine_and_multiprobe);
  RUN_TEST(test_lsh_euclidean_save_load_and_errors);
  RUN_TEST(test_lsh_add_batch_contains_and_concurrent_tsan);
  RUN_TEST(test_lsh_search_batch_mt_and_parity);
  RUN_TEST(test_lsh_upsert_get_list_compact_and_tsan);
}
