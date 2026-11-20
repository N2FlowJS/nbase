#include "test_common.h"

#include <pthread.h>
#include <stdatomic.h>

typedef struct stress_ctx {
  nbase_db_t *db;
  uint32_t dim;
  atomic_bool stop;
  atomic_int error_count;
} stress_ctx_t;

static void *reader_thread(void *raw) {
  stress_ctx_t *ctx = (stress_ctx_t *)raw;
  float query[16];
  test_fill_vectors(query, 1U, ctx->dim, 777ULL);

  for (uint32_t iter = 0U; iter < 60U && !atomic_load_explicit(&ctx->stop, memory_order_relaxed); iter++) {
    nbase_result_t hits[5];
    uint32_t count = 0U;
    nbase_status_t st = nbase_search(ctx->db, query, ctx->dim, 5U, NULL, hits, &count);
    if (st != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->error_count, 1, memory_order_relaxed);
      break;
    }

    float vec_buf[16];
    st = nbase_get(ctx->db, 10ULL, vec_buf, ctx->dim);
    if (st != NBASE_OK && st != NBASE_ERR_NOT_FOUND) {
      atomic_fetch_add_explicit(&ctx->error_count, 1, memory_order_relaxed);
      break;
    }

    nbase_meta_value_t mv;
    st = nbase_get_meta(ctx->db, 10ULL, "ver", &mv, NULL, 0U);
    if (st != NBASE_OK && st != NBASE_ERR_NOT_FOUND) {
      atomic_fetch_add_explicit(&ctx->error_count, 1, memory_order_relaxed);
      break;
    }
  }
  return NULL;
}

static void *writer_thread(void *raw) {
  stress_ctx_t *ctx = (stress_ctx_t *)raw;
  float v[16];
  for (uint32_t i = 0U; i < 60U; i++) {
    test_fill_vectors(v, 1U, ctx->dim, 5000ULL + i);
    uint64_t id = 1000ULL + i;
    if (nbase_add(ctx->db, id, v, ctx->dim) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->error_count, 1, memory_order_relaxed);
      break;
    }
    nbase_meta_field_t fld = {"ver", {.type = NBASE_META_INT64, .as = {.i64 = (int64_t)i}}};
    if (nbase_set_meta(ctx->db, id, &fld, 1U) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->error_count, 1, memory_order_relaxed);
      break;
    }
    if (nbase_upsert(ctx->db, 10ULL, v, ctx->dim) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->error_count, 1, memory_order_relaxed);
      break;
    }
    if (i >= 10U && (i % 3U == 0U)) {
      if (nbase_delete(ctx->db, id - 5ULL) != NBASE_OK) {
        atomic_fetch_add_explicit(&ctx->error_count, 1, memory_order_relaxed);
        break;
      }
    }
  }
  atomic_store_explicit(&ctx->stop, true, memory_order_release);
  return NULL;
}

static void test_search_batch_multithreaded_matches_serial(void) {
  const uint32_t count = 1000U;
  const uint32_t dim = 32U;
  const uint32_t nqueries = 40U;
  const uint32_t k = 10U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  float *queries = malloc((size_t)nqueries * dim * sizeof(float));
  nbase_result_t *res1 = malloc((size_t)nqueries * k * sizeof(nbase_result_t));
  nbase_result_t *res4 = malloc((size_t)nqueries * k * sizeof(nbase_result_t));
  uint32_t *cnt1 = malloc(nqueries * sizeof(uint32_t));
  uint32_t *cnt4 = malloc(nqueries * sizeof(uint32_t));
  ASSERT_TRUE(vecs && queries && res1 && res4 && cnt1 && cnt4);

  test_fill_vectors(vecs, count, dim, 1010ULL);
  test_fill_vectors(queries, nqueries, dim, 2020ULL);

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.seed = 3030ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);
  for (uint32_t i = 0U; i < count; i++) {
    ASSERT_STATUS(nbase_add(db, (uint64_t)i, vecs + (size_t)i * dim, dim), NBASE_OK);
  }

  nbase_search_opts_t opts = nbase_search_opts_default();
  opts.num_threads = 1U;
  ASSERT_STATUS(nbase_search_batch(db, queries, nqueries, dim, k, &opts, res1, cnt1), NBASE_OK);

  opts.num_threads = 4U;
  ASSERT_STATUS(nbase_search_batch(db, queries, nqueries, dim, k, &opts, res4, cnt4), NBASE_OK);

  for (uint32_t q = 0U; q < nqueries; q++) {
    ASSERT_EQ_U64(cnt4[q], cnt1[q]);
    for (uint32_t j = 0U; j < cnt1[q]; j++) {
      ASSERT_EQ_U64(res4[q * k + j].id, res1[q * k + j].id);
      ASSERT_NEAR(res4[q * k + j].distance, res1[q * k + j].distance, 0.0, 0.0);
    }
  }

  nbase_close(db);
  free(vecs);
  free(queries);
  free(res1);
  free(res4);
  free(cnt1);
  free(cnt4);
}

static void test_concurrent_readers_and_writer_rwlock(void) {
  const uint32_t dim = 16U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.seed = 9090ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float seed_vecs[100 * 16];
  test_fill_vectors(seed_vecs, 100U, dim, 123ULL);
  for (uint32_t i = 0U; i < 100U; i++) {
    ASSERT_STATUS(nbase_add(db, (uint64_t)i, seed_vecs + (size_t)i * dim, dim), NBASE_OK);
    nbase_meta_field_t fld = {"ver", {.type = NBASE_META_INT64, .as = {.i64 = 1}}};
    ASSERT_STATUS(nbase_set_meta(db, (uint64_t)i, &fld, 1U), NBASE_OK);
  }

  stress_ctx_t ctx;
  ctx.db = db;
  ctx.dim = dim;
  atomic_init(&ctx.stop, false);
  atomic_init(&ctx.error_count, 0);

  pthread_t readers[4];
  pthread_t writer;
  for (int i = 0; i < 4; i++) {
    ASSERT_TRUE(pthread_create(&readers[i], NULL, reader_thread, &ctx) == 0);
  }
  ASSERT_TRUE(pthread_create(&writer, NULL, writer_thread, &ctx) == 0);

  pthread_join(writer, NULL);
  for (int i = 0; i < 4; i++) {
    pthread_join(readers[i], NULL);
  }

  ASSERT_EQ_I64(atomic_load(&ctx.error_count), 0LL);
  nbase_close(db);
}

static void test_add_batch_and_rebuild_mt_tsan(void) {
  const uint32_t dim = 32U;
  const uint32_t n = 400U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.hnsw_ef_search = 80U;
  cfg.seed = 4242ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float *vecs = malloc((size_t)n * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);
  test_fill_vectors(vecs, n, dim, 5150ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = 100ULL + i;
  }

  /* Atomic validation: duplicate within batch rejects without partial insert. */
  uint64_t dup_ids[3] = {1ULL, 2ULL, 1ULL};
  ASSERT_STATUS(nbase_add_batch(db, dup_ids, vecs, 3U, dim, 4U), NBASE_ERR_ALREADY_EXISTS);
  ASSERT_TRUE(!nbase_contains(db, 1ULL));

  ASSERT_STATUS(nbase_add_batch(db, ids, vecs, n, dim, 4U), NBASE_OK);

  nbase_stats_t st;
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, n);
  ASSERT_EQ_U64(st.hnsw_active_nodes, n);

  float query[32];
  test_fill_vectors(query, 1U, dim, 9999ULL);
  nbase_result_t oracle[10];
  nbase_result_t hits[10];
  uint32_t oracle_n = 0U;
  uint32_t hit_n = 0U;
  test_brute_force(ids, vecs, NULL, n, dim, query, 10U, NBASE_METRIC_EUCLIDEAN, NULL, NULL, oracle, &oracle_n);

  ASSERT_STATUS(nbase_search(db, query, dim, 10U, NULL, hits, &hit_n), NBASE_OK);
  ASSERT_EQ_U64(hit_n, 10ULL);
  ASSERT_TRUE(test_recall_at(hits, hit_n, oracle, oracle_n, 10U) >= 0.8);

  ASSERT_STATUS(nbase_rebuild_hnsw_mt(db, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_search(db, query, dim, 10U, NULL, hits, &hit_n), NBASE_OK);
  ASSERT_EQ_U64(hit_n, 10ULL);
  ASSERT_TRUE(test_recall_at(hits, hit_n, oracle, oracle_n, 10U) >= 0.8);

  free(vecs);
  free(ids);
  nbase_close(db);
}

void test_concurrency_run(void) {
  RUN_TEST(test_search_batch_multithreaded_matches_serial);
  RUN_TEST(test_concurrent_readers_and_writer_rwlock);
  RUN_TEST(test_add_batch_and_rebuild_mt_tsan);
}
