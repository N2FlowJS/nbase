#include "test_common.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

static void test_db_upsert_with_and_without_hnsw(void) {
  const uint32_t dim = 8U;
  float v1[8] = {1, 0, 0, 0, 0, 0, 0, 0};
  float v2[8] = {0, 1, 0, 0, 0, 0, 0, 0};
  float v3[8] = {0, 0, 1, 0, 0, 0, 0, 0};
  float out[8] = {0};

  for (int use_hnsw = 0; use_hnsw <= 1; use_hnsw++) {
    nbase_config_t cfg = nbase_config_default(dim);
    cfg.enable_hnsw = (use_hnsw != 0);
    cfg.seed = 55ULL;

    nbase_db_t *db = NULL;
    ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

    ASSERT_STATUS(nbase_upsert(db, 100ULL, v1, dim), NBASE_OK);
    ASSERT_STATUS(nbase_add(db, 200ULL, v2, dim), NBASE_OK);
    ASSERT_STATUS(nbase_upsert(db, 100ULL, v3, dim), NBASE_OK);

    nbase_stats_t st;
    ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
    ASSERT_EQ_U64(st.active_vectors, 2ULL);

    ASSERT_STATUS(nbase_get(db, 100ULL, out, dim), NBASE_OK);
    ASSERT_NEAR(out[0], 0.0, 1e-6, 0.0);
    ASSERT_NEAR(out[2], 1.0, 1e-6, 0.0);

    nbase_result_t res[2];
    uint32_t count = 0U;
    ASSERT_STATUS(nbase_search(db, v3, dim, 2U, NULL, res, &count), NBASE_OK);
    ASSERT_EQ_U64(count, 2ULL);
    ASSERT_EQ_U64(res[0].id, 100ULL);
    ASSERT_NEAR(res[0].distance, 0.0, 1e-6, 0.0);

    nbase_close(db);
  }
}

static void test_db_batch_search(void) {
  const uint32_t count = 200U;
  const uint32_t dim = 16U;
  const uint32_t nqueries = 10U;
  const uint32_t k = 5U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  float *queries = malloc((size_t)nqueries * dim * sizeof(float));
  ASSERT_TRUE(vecs != NULL && queries != NULL);
  test_fill_vectors(vecs, count, dim, 9001ULL);
  test_fill_vectors(queries, nqueries, dim, 9002ULL);

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.seed = 9003ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  for (uint32_t i = 0U; i < count; i++) {
    ASSERT_STATUS(nbase_add(db, (uint64_t)i, vecs + (size_t)i * dim, dim), NBASE_OK);
  }

  nbase_result_t batch_res[10 * 5];
  uint32_t batch_counts[10];
  ASSERT_STATUS(nbase_search_batch(db, queries, nqueries, dim, k, NULL, batch_res, batch_counts), NBASE_OK);

  for (uint32_t q = 0U; q < nqueries; q++) {
    ASSERT_EQ_U64(batch_counts[q], k);
    nbase_result_t single_res[5];
    uint32_t single_n = 0U;
    ASSERT_STATUS(nbase_search(db, queries + (size_t)q * dim, dim, k, NULL, single_res, &single_n), NBASE_OK);
    ASSERT_EQ_U64(single_n, k);
    for (uint32_t j = 0U; j < k; j++) {
      ASSERT_EQ_U64(batch_res[q * k + j].id, single_res[j].id);
      ASSERT_NEAR(batch_res[q * k + j].distance, single_res[j].distance, 1e-6, 0.0);
    }
  }

  nbase_close(db);
  free(vecs);
  free(queries);
}

static void test_db_persistence_mmap_and_errors(void) {
  nbase_config_t bad_cfg = nbase_config_default(0U);
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&bad_cfg, &db), NBASE_ERR_INVALID_ARG);

  bad_cfg = nbase_config_default(4U);
  bad_cfg.hnsw_m = 1U;
  ASSERT_STATUS(nbase_create(&bad_cfg, &db), NBASE_ERR_INVALID_ARG);

  nbase_config_t cfg = nbase_config_default(4U);
  cfg.enable_hnsw = false;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float v4[4] = {1, 2, 3, 4};
  float v3[3] = {1, 2, 3};
  nbase_result_t res[4];
  uint32_t out_n = 0U;

  ASSERT_STATUS(nbase_add(db, 1ULL, v3, 3U), NBASE_ERR_DIM_MISMATCH);
  ASSERT_STATUS(nbase_get(db, 1ULL, v3, 3U), NBASE_ERR_DIM_MISMATCH);
  ASSERT_STATUS(nbase_get(db, 999ULL, v4, 4U), NBASE_ERR_NOT_FOUND);
  ASSERT_STATUS(nbase_search(db, v3, 3U, 1U, NULL, res, &out_n), NBASE_ERR_DIM_MISMATCH);

  float qnan[4] = {1, NAN, 0, 0};
  ASSERT_STATUS(nbase_search(db, qnan, 4U, 1U, NULL, res, &out_n), NBASE_ERR_INVALID_VECTOR);

  nbase_search_opts_t hnsw_only = nbase_search_opts_default();
  hnsw_only.mode = NBASE_SEARCH_HNSW;
  ASSERT_STATUS(nbase_search(db, v4, 4U, 1U, &hnsw_only, res, &out_n), NBASE_ERR_UNSUPPORTED);
  ASSERT_STATUS(nbase_rebuild_hnsw(db), NBASE_ERR_UNSUPPORTED);

  nbase_close(db);

  /* Full save -> open_mmap -> search -> .nbi corruption check. */
  char dir[128];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  cfg = nbase_config_default(4U);
  cfg.seed = 42ULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 10ULL, v4, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_save(db, dir), NBASE_OK);
  nbase_close(db);

  ASSERT_STATUS(nbase_open_mmap(dir, &db), NBASE_OK);
  nbase_stats_t st;
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_TRUE(st.mmap_backed);
  ASSERT_TRUE(st.hnsw_enabled);
  ASSERT_EQ_U64(st.hnsw_active_nodes, 1ULL);
  ASSERT_STATUS(nbase_rebuild_hnsw(db), NBASE_ERR_UNSUPPORTED);
  ASSERT_STATUS(nbase_search(db, v4, 4U, 1U, NULL, res, &out_n), NBASE_OK);
  ASSERT_EQ_U64(out_n, 1ULL);
  ASSERT_EQ_U64(res[0].id, 10ULL);
  nbase_close(db);

  /* Corrupt hnsw.nbi -> open fails with NBASE_ERR_CORRUPT. */
  char idx_path[256];
  snprintf(idx_path, sizeof(idx_path), "%s/hnsw.nbi", dir);
  FILE *fp = fopen(idx_path, "r+b");
  ASSERT_TRUE(fp != NULL);
  ASSERT_TRUE(fseek(fp, 20L, SEEK_SET) == 0);
  int ch = fgetc(fp);
  ASSERT_TRUE(ch != EOF);
  ASSERT_TRUE(fseek(fp, 20L, SEEK_SET) == 0);
  ASSERT_TRUE(fputc(ch ^ 0xFF, fp) != EOF);
  fclose(fp);

  ASSERT_STATUS(nbase_open(dir, &db), NBASE_ERR_CORRUPT);

  test_rm_rf(dir);
}

static void test_db_compact(void) {
  const uint32_t count = 300U;
  const uint32_t dim = 16U;
  const uint32_t k = 10U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)count * sizeof(uint64_t));
  bool *deleted = calloc(count, sizeof(bool));
  ASSERT_TRUE(vecs != NULL && ids != NULL && deleted != NULL);
  test_fill_vectors(vecs, count, dim, 5151ULL);

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.seed = 707ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  for (uint32_t i = 0U; i < count; i++) {
    ids[i] = (uint64_t)i;
    ASSERT_STATUS(nbase_add(db, ids[i], vecs + (size_t)i * dim, dim), NBASE_OK);
    nbase_meta_field_t fld = {"idx", {.type = NBASE_META_INT64, .as = {.i64 = (int64_t)i}}};
    ASSERT_STATUS(nbase_set_meta(db, ids[i], &fld, 1U), NBASE_OK);
  }

  for (uint32_t i = 0U; i < 150U; i++) {
    deleted[i] = true;
    ASSERT_STATUS(nbase_delete(db, ids[i]), NBASE_OK);
  }

  nbase_stats_t st;
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.tombstones, 150ULL);
  ASSERT_EQ_U64(st.hnsw_deleted_nodes, 150ULL);

  ASSERT_STATUS(nbase_compact(db), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, 150ULL);
  ASSERT_EQ_U64(st.total_slots, 150ULL);
  ASSERT_EQ_U64(st.tombstones, 0ULL);
  ASSERT_EQ_U64(st.hnsw_active_nodes, 150ULL);
  ASSERT_EQ_U64(st.hnsw_deleted_nodes, 0ULL);

  nbase_meta_value_t mv;
  ASSERT_STATUS(nbase_get_meta(db, 200ULL, "idx", &mv, NULL, 0U), NBASE_OK);
  ASSERT_EQ_I64(mv.as.i64, 200LL);

  float query[16];
  test_fill_vectors(query, 1U, dim, 6161ULL);
  nbase_result_t actual[10];
  nbase_result_t expected[10];
  uint32_t act_n = 0U;
  uint32_t exp_n = 0U;
  ASSERT_STATUS(nbase_search(db, query, dim, k, NULL, actual, &act_n), NBASE_OK);
  test_brute_force(ids, vecs, deleted, count, dim, query, k, NBASE_METRIC_EUCLIDEAN, NULL, NULL, expected, &exp_n);
  ASSERT_TRUE(test_recall_at(actual, act_n, expected, exp_n, k) >= 0.95);

  char dir[128];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  ASSERT_STATUS(nbase_save(db, dir), NBASE_OK);
  nbase_close(db);

  ASSERT_STATUS(nbase_open(dir, &db), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.total_slots, 150ULL);
  ASSERT_EQ_U64(st.tombstones, 0ULL);
  ASSERT_STATUS(nbase_get_meta(db, 299ULL, "idx", &mv, NULL, 0U), NBASE_OK);
  ASSERT_EQ_I64(mv.as.i64, 299LL);
  nbase_close(db);

  test_rm_rf(dir);
  free(vecs);
  free(ids);
  free(deleted);
}

static void test_db_batch_and_filter_delete_and_memory(void) {
  nbase_config_t cfg = nbase_config_default(4U);
  cfg.seed = 808ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float vecs[5 * 4];
  uint64_t ids[5] = {1ULL, 2ULL, 3ULL, 4ULL, 5ULL};
  test_fill_vectors(vecs, 5U, 4U, 999ULL);
  ASSERT_STATUS(nbase_add_batch(db, ids, vecs, 5U, 4U, 2U), NBASE_OK);

  for (uint32_t i = 0U; i < 5U; i++) {
    nbase_meta_field_t fld = {"cat", {.type = NBASE_META_STRING, .as = {.str = (i % 2U == 0U) ? "temp" : "keep"}}};
    ASSERT_STATUS(nbase_set_meta(db, ids[i], &fld, 1U), NBASE_OK);
  }
  ASSERT_STATUS(nbase_set_text(db, 1ULL, "hello world"), NBASE_OK);

  nbase_stats_t st;
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_TRUE(st.memory_bytes > 0ULL);
  uint64_t mem_before = st.memory_bytes;
  (void)mem_before;

  /* Delete batch containing existing id 2 and missing id 999 */
  uint64_t del_ids[2] = {2ULL, 999ULL};
  uint32_t del_cnt = 0U;
  ASSERT_STATUS(nbase_delete_batch(db, del_ids, 2U, &del_cnt), NBASE_OK);
  ASSERT_EQ_U64(del_cnt, 1ULL);
  ASSERT_TRUE(!nbase_contains(db, 2ULL));

  /* Delete by metadata filter removes ids 1, 3, 5 ("temp") leaving only id 4 */
  nbase_meta_cond_t cond = {"cat", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "temp"}}};
  nbase_meta_filter_t mflt = {&cond, 1U};
  ASSERT_STATUS(nbase_delete_by_filter(db, &mflt, &del_cnt), NBASE_OK);
  ASSERT_EQ_U64(del_cnt, 3ULL);
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, 1ULL);
  ASSERT_TRUE(nbase_contains(db, 4ULL));

  nbase_close(db);
}

static void test_db_query_lru_cache_and_invalidation(void) {
  const uint32_t dim = 3U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.enable_hnsw = false;
  cfg.cache_capacity = 2U;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float v1[3] = {1.0f, 0.0f, 0.0f};
  float v2[3] = {0.0f, 1.0f, 0.0f};
  float v3[3] = {0.0f, 0.0f, 1.0f};
  ASSERT_STATUS(nbase_add(db, 1ULL, v1, dim), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 2ULL, v2, dim), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 3ULL, v3, dim), NBASE_OK);

  nbase_meta_field_t f1 = {"grp", {.type = NBASE_META_STRING, .as = {.str = "a"}}};
  nbase_meta_field_t f2 = {"grp", {.type = NBASE_META_STRING, .as = {.str = "b"}}};
  ASSERT_STATUS(nbase_set_meta(db, 1ULL, &f1, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 2ULL, &f2, 1U), NBASE_OK);

  nbase_result_t res[3];
  uint32_t cnt = 0U;
  /* Miss then hit on q=v1 */
  ASSERT_STATUS(nbase_search(db, v1, dim, 2U, NULL, res, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 2ULL);
  ASSERT_EQ_U64(res[0].id, 1ULL);
  ASSERT_STATUS(nbase_search(db, v1, dim, 2U, NULL, res, &cnt), NBASE_OK);

  nbase_stats_t st;
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.cache_capacity, 2ULL);
  ASSERT_EQ_U64(st.cache_entries, 1ULL);
  ASSERT_EQ_U64(st.cache_hits, 1ULL);
  ASSERT_EQ_U64(st.cache_misses, 1ULL);

  /* Different metadata filter must get distinct cache entry */
  nbase_meta_cond_t cond_b = {"grp", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "b"}}};
  nbase_meta_filter_t flt_b = {&cond_b, 1U};
  nbase_search_opts_t opts = nbase_search_opts_default();
  opts.meta_filter = &flt_b;
  ASSERT_STATUS(nbase_search(db, v1, dim, 2U, &opts, res, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 1ULL);
  ASSERT_EQ_U64(res[0].id, 2ULL);

  /* Touch q=v1 unfiltered so flt_b becomes LRU, then add 3rd entry v3 (cap=2 evicts flt_b) */
  ASSERT_STATUS(nbase_search(db, v1, dim, 2U, NULL, res, &cnt), NBASE_OK);
  ASSERT_STATUS(nbase_search(db, v3, dim, 2U, NULL, res, &cnt), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.cache_entries, 2ULL);
  ASSERT_EQ_U64(st.cache_hits, 2ULL);
  ASSERT_EQ_U64(st.cache_misses, 3ULL);

  /* Touch v1 is still cached; flt_b was evicted */
  ASSERT_STATUS(nbase_search(db, v1, dim, 2U, NULL, res, &cnt), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.cache_hits, 3ULL);

  /* skip_cache bypasses lookup & insertion */
  opts = nbase_search_opts_default();
  opts.skip_cache = true;
  ASSERT_STATUS(nbase_search(db, v2, dim, 2U, &opts, res, &cnt), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.cache_hits, 3ULL);
  ASSERT_EQ_U64(st.cache_misses, 3ULL);

  /* Mutation invalidates cache automatically */
  float v4[3] = {0.95f, 0.0f, 0.0f};
  ASSERT_STATUS(nbase_add(db, 4ULL, v4, dim), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.cache_entries, 0ULL);

  /* Search batch multi-threaded with cache enabled works cleanly */
  ASSERT_STATUS(nbase_cache_configure(db, 8U), NBASE_OK);
  float batch_q[6] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  nbase_result_t batch_res[4];
  uint32_t batch_cnts[2] = {0U, 0U};
  nbase_search_opts_t bopts = nbase_search_opts_default();
  bopts.num_threads = 2U;
  ASSERT_STATUS(nbase_search_batch(db, batch_q, 2U, dim, 2U, &bopts, batch_res, batch_cnts), NBASE_OK);
  ASSERT_EQ_U64(batch_cnts[0], 2ULL);
  ASSERT_EQ_U64(batch_cnts[1], 2ULL);

  ASSERT_STATUS(nbase_cache_clear(db), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.cache_entries, 0ULL);

  nbase_close(db);
}

static void test_db_upsert_batch_and_get_batch(void) {
  ASSERT_TRUE(strcmp(nbase_version(), "1.21.0") == 0);

  const uint32_t dim = 3U;
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.enable_hnsw = true;
  cfg.cache_capacity = 8U;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);
  ASSERT_STATUS(nbase_save(db, dir), NBASE_OK);
  nbase_close(db);

  /* Reopen with WAL enabled and insert initial vectors */
  ASSERT_STATUS(nbase_open_wal(dir, &db), NBASE_OK);
  ASSERT_STATUS(nbase_cache_configure(db, 8U), NBASE_OK);
  uint64_t init_ids[2] = {1ULL, 2ULL};
  float init_vecs[6] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f
  };
  ASSERT_STATUS(nbase_add_batch(db, init_ids, init_vecs, 2U, dim, 2U), NBASE_OK);
  ASSERT_STATUS(nbase_wal_checkpoint(db), NBASE_OK);

  /* Populate query cache, then run mixed upsert_batch (existing 2 + new 3) */
  nbase_result_t res[2];
  uint32_t cnt = 0U;
  float q[3] = {0.0f, 1.0f, 0.0f};
  ASSERT_STATUS(nbase_search(db, q, dim, 2U, NULL, res, &cnt), NBASE_OK);
  nbase_stats_t st;
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.cache_entries, 1ULL);

  /* Duplicate ID inside upsert_batch is rejected cleanly without mutation */
  uint64_t dup_ids[2] = {5ULL, 5ULL};
  float dup_vecs[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  ASSERT_STATUS(nbase_upsert_batch(db, dup_ids, dup_vecs, 2U, dim, 0U), NBASE_ERR_INVALID_ARG);

  uint64_t up_ids[3] = {2ULL, 3ULL, 4ULL};
  float up_vecs[9] = {
      0.0f, 0.5f, 0.5f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f
  };
  ASSERT_STATUS(nbase_upsert_batch(db, up_ids, up_vecs, 3U, dim, 2U), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, 4ULL);
  ASSERT_EQ_U64(st.cache_entries, 0ULL);

  /* Simulate crash (close without save) and replay WAL */
  nbase_close(db);
  ASSERT_STATUS(nbase_open_wal(dir, &db), NBASE_OK);

  uint64_t get_ids[4] = {1ULL, 2ULL, 999ULL, 3ULL};
  float got_vecs[12];
  bool got_found[4] = {false, false, false, false};
  ASSERT_STATUS(nbase_get_batch(db, get_ids, 4U, dim, got_vecs, got_found), NBASE_OK);
  ASSERT_TRUE(got_found[0]);
  ASSERT_TRUE(got_found[1]);
  ASSERT_TRUE(!got_found[2]);
  ASSERT_TRUE(got_found[3]);
  ASSERT_NEAR(got_vecs[0], 1.0, 1e-6, 0.0);
  ASSERT_NEAR(got_vecs[4], 0.5, 1e-6, 0.0);
  ASSERT_NEAR(got_vecs[6], 0.0, 1e-6, 0.0);
  ASSERT_NEAR(got_vecs[10], 1.0, 1e-6, 0.0);

  nbase_close(db);
  test_rm_rf(dir);
}

static void test_db_list_ids_and_search_radius(void) {
  const uint32_t dim = 3U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.enable_hnsw = true;
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  uint64_t ids[5] = {50ULL, 10ULL, 40ULL, 20ULL, 30ULL};
  float vecs[15] = {
      1.0f, 0.0f, 0.0f,
      1.05f, 0.0f, 0.0f,
      1.20f, 0.0f, 0.0f,
      2.0f, 0.0f, 0.0f,
      5.0f, 0.0f, 0.0f
  };
  ASSERT_STATUS(nbase_add_batch(db, ids, vecs, 5U, dim, 2U), NBASE_OK);

  /* Metadata for filtered radius search */
  nbase_meta_field_t mk = {"grp", {.type = NBASE_META_STRING, .as = {.str = "ok"}}};
  nbase_meta_field_t mn = {"grp", {.type = NBASE_META_STRING, .as = {.str = "no"}}};
  ASSERT_STATUS(nbase_set_meta(db, 10ULL, &mk, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 50ULL, &mn, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 40ULL, &mk, 1U), NBASE_OK);

  /* Paginated deterministic sorted ID enumeration: 10, 20, 30, 40, 50 */
  uint64_t page[2] = {0ULL, 0ULL};
  uint32_t cnt = 0U;
  uint64_t total = 0ULL;
  ASSERT_STATUS(nbase_list_ids(db, 1ULL, 2U, page, &cnt, &total), NBASE_OK);
  ASSERT_EQ_U64(total, 5ULL);
  ASSERT_EQ_U64(cnt, 2ULL);
  ASSERT_EQ_U64(page[0], 20ULL);
  ASSERT_EQ_U64(page[1], 30ULL);

  ASSERT_STATUS(nbase_list_ids(db, 10ULL, 2U, page, &cnt, &total), NBASE_OK);
  ASSERT_EQ_U64(total, 5ULL);
  ASSERT_EQ_U64(cnt, 0ULL);

  /* Exact/Auto radius search around (1,0,0) within 0.25 -> ids 50 (0.0), 10 (0.05), 40 (0.20) */
  float q[3] = {1.0f, 0.0f, 0.0f};
  nbase_result_t hits[5];
  ASSERT_STATUS(nbase_search_radius(db, q, dim, 0.25f, 5U, NULL, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 3ULL);
  ASSERT_EQ_U64(hits[0].id, 50ULL);
  ASSERT_EQ_U64(hits[1].id, 10ULL);
  ASSERT_EQ_U64(hits[2].id, 40ULL);

  /* HNSW mode + metadata filter + max_results cap */
  nbase_meta_cond_t cond = {"grp", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "ok"}}};
  nbase_meta_filter_t flt = {&cond, 1U};
  nbase_search_opts_t opts = nbase_search_opts_default();
  opts.mode = NBASE_SEARCH_HNSW;
  opts.ef_search = 32U;
  opts.meta_filter = &flt;
  ASSERT_STATUS(nbase_search_radius(db, q, dim, 0.25f, 5U, &opts, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 2ULL);
  ASSERT_EQ_U64(hits[0].id, 10ULL);
  ASSERT_EQ_U64(hits[1].id, 40ULL);

  /* Negative finite max_distance allowed on inner product metric; NaN rejected */
  opts = nbase_search_opts_default();
  opts.use_metric_override = true;
  opts.metric = NBASE_METRIC_INNER_PRODUCT;
  ASSERT_STATUS(nbase_search_radius(db, q, dim, -1.1f, 5U, &opts, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 3ULL);
  ASSERT_STATUS(nbase_search_radius(db, q, dim, (float)NAN, 5U, NULL, hits, &cnt), NBASE_ERR_INVALID_ARG);

  nbase_close(db);
}

static void test_db_count_filtered_and_unfiltered(void) {
  const uint32_t dim = 3U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  uint64_t cnt = 999ULL;
  ASSERT_STATUS(nbase_count(db, NULL, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 0ULL);
  ASSERT_STATUS(nbase_count(NULL, NULL, &cnt), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_count(db, NULL, NULL), NBASE_ERR_INVALID_ARG);
  nbase_meta_filter_t bad_flt = {NULL, 1U};
  ASSERT_STATUS(nbase_count(db, &bad_flt, &cnt), NBASE_ERR_INVALID_ARG);

  uint64_t ids[4] = {1ULL, 2ULL, 3ULL, 4ULL};
  float vecs[12] = {
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
      1, 1, 0};
  ASSERT_STATUS(nbase_add_batch(db, ids, vecs, 4U, dim, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_count(db, NULL, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 4ULL);

  nbase_meta_field_t f_a[2] = {
      {"team", {.type = NBASE_META_STRING, .as = {.str = "blue"}}},
      {"score", {.type = NBASE_META_INT64, .as = {.i64 = 10}}}};
  nbase_meta_field_t f_b[2] = {
      {"team", {.type = NBASE_META_STRING, .as = {.str = "blue"}}},
      {"score", {.type = NBASE_META_INT64, .as = {.i64 = 30}}}};
  nbase_meta_field_t f_c[2] = {
      {"team", {.type = NBASE_META_STRING, .as = {.str = "red"}}},
      {"score", {.type = NBASE_META_INT64, .as = {.i64 = 50}}}};
  nbase_meta_batch_item_t mitems[3] = {
      {1ULL, f_a, 2U}, {2ULL, f_b, 2U}, {3ULL, f_c, 2U}};
  ASSERT_STATUS(nbase_set_meta_batch(db, mitems, 3U), NBASE_OK);

  nbase_meta_cond_t conds[2] = {
      {"team", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "blue"}}},
      {"score", NBASE_OP_GTE, {.type = NBASE_META_INT64, .as = {.i64 = 20}}}};
  nbase_meta_filter_t flt = {conds, 2U};
  ASSERT_STATUS(nbase_count(db, &flt, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 1ULL);

  /* Delete matching vector -> count reflects live state accurately. */
  ASSERT_STATUS(nbase_delete(db, 2ULL), NBASE_OK);
  ASSERT_STATUS(nbase_count(db, NULL, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 3ULL);
  ASSERT_STATUS(nbase_count(db, &flt, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 0ULL);

  nbase_close(db);
}

static bool fuzz_file_len(const char *path, long *out_len) {
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    return false;
  }
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return false;
  }
  long sz = ftell(fp);
  fclose(fp);
  if (sz < 0) {
    return false;
  }
  *out_len = sz;
  return true;
}

static bool fuzz_copy_file(const char *src, const char *dst, long max_bytes) {
  FILE *in = fopen(src, "rb");
  if (!in) {
    return false;
  }
  FILE *out = fopen(dst, "wb");
  if (!out) {
    fclose(in);
    return false;
  }
  char buf[4096];
  long rem = max_bytes;
  bool ok = true;
  while (max_bytes < 0L || rem > 0L) {
    size_t want = sizeof(buf);
    if (max_bytes >= 0L && (size_t)rem < want) {
      want = (size_t)rem;
    }
    size_t n = fread(buf, 1U, want, in);
    if (n == 0U) {
      break;
    }
    if (fwrite(buf, 1U, n, out) != n) {
      ok = false;
      break;
    }
    if (max_bytes >= 0L) {
      rem -= (long)n;
    }
  }
  fclose(in);
  fclose(out);
  return ok;
}

static void fuzz_verify_standalone_format(
    const char *orig_path,
    const char *work_path,
    nbase_status_t (*try_load)(const char *)) {
  long fsz = 0L;
  ASSERT_TRUE(fuzz_file_len(orig_path, &fsz));
  ASSERT_TRUE(fsz >= 128L);

  const long truncs[5] = {0L, 16L, 64L, 127L, fsz - 1L};
  for (size_t ti = 0; ti < 5; ti++) {
    ASSERT_TRUE(fuzz_copy_file(orig_path, work_path, truncs[ti]));
    ASSERT_TRUE(try_load(work_path) != NBASE_OK);
  }
  if (fsz > 140L) {
    ASSERT_TRUE(fuzz_copy_file(orig_path, work_path, 128L + (fsz - 128L) / 2L));
    ASSERT_TRUE(try_load(work_path) != NBASE_OK);
  }

  const long flips[3] = {16L, 128L < fsz ? 128L : 32L, fsz - 4L};
  for (size_t fi = 0; fi < 3; fi++) {
    ASSERT_TRUE(fuzz_copy_file(orig_path, work_path, -1L));
    ASSERT_TRUE(test_flip_file_byte(work_path, flips[fi]));
    ASSERT_TRUE(try_load(work_path) != NBASE_OK);
  }
}

static nbase_status_t fuzz_try_load_lsh(const char *path) {
  nbase_lsh_t *p = NULL;
  nbase_status_t st = nbase_lsh_load(path, &p);
  if (p != NULL) {
    nbase_lsh_destroy(p);
    return NBASE_OK;
  }
  return st;
}

static nbase_status_t fuzz_try_load_clustered(const char *path) {
  nbase_clustered_t *p = NULL;
  nbase_status_t st = nbase_clustered_load(path, &p);
  if (p != NULL) {
    nbase_clustered_destroy(p);
    return NBASE_OK;
  }
  return st;
}

static nbase_status_t fuzz_try_load_ivfpq(const char *path) {
  nbase_ivfpq_t *p = NULL;
  nbase_status_t st = nbase_ivfpq_load(path, &p);
  if (p != NULL) {
    nbase_ivfpq_destroy(p);
    return NBASE_OK;
  }
  return st;
}

static nbase_status_t fuzz_try_load_sq8(const char *path) {
  nbase_sq8_t *p = NULL;
  nbase_status_t st = nbase_sq8_load(path, &p);
  if (p != NULL) {
    nbase_sq8_destroy(p);
    return NBASE_OK;
  }
  return st;
}

static nbase_status_t fuzz_try_load_bq(const char *path) {
  nbase_bq_t *p = NULL;
  nbase_status_t st = nbase_bq_load(path, &p);
  if (p != NULL) {
    nbase_bq_destroy(p);
    return NBASE_OK;
  }
  return st;
}

static void test_binary_formats_fuzz_corruption_and_truncation(void) {
  char orig_dir[256];
  char fuzz_dir[256];
  ASSERT_TRUE(test_make_tmpdir(orig_dir, sizeof(orig_dir)));
  ASSERT_TRUE(test_make_tmpdir(fuzz_dir, sizeof(fuzz_dir)));

  const uint32_t dim = 16U;
  const uint32_t n = 32U;
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);
  test_fill_vectors(vecs, n, dim, 9911ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = (uint64_t)i + 1ULL;
  }

  /* 1. Embedded DB directory snapshot (vectors.nbv, hnsw.nbi, meta.nbm, bm25.nbbm) */
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.enable_hnsw = true;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);
  ASSERT_STATUS(nbase_add_batch(db, ids, vecs, n, dim, 1U), NBASE_OK);
  nbase_meta_field_t fld = {"tag", {.type = NBASE_META_STRING, .as = {.str = "fuzz"}}};
  ASSERT_STATUS(nbase_set_meta(db, 1ULL, &fld, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 1ULL, "fuzz test bm25 snapshot"), NBASE_OK);
  ASSERT_STATUS(nbase_save(db, orig_dir), NBASE_OK);
  nbase_close(db);

  static const char *db_files[4] = {"vectors.nbv", "hnsw.nbi", "meta.nbm", "bm25.nbbm"};
  for (size_t target = 0; target < 4; target++) {
    for (size_t c = 0; c < 4; c++) {
      char sp[300], dp[300];
      snprintf(sp, sizeof(sp), "%s/%s", orig_dir, db_files[c]);
      snprintf(dp, sizeof(dp), "%s/%s", fuzz_dir, db_files[c]);
      ASSERT_TRUE(fuzz_copy_file(sp, dp, -1L));
    }
    char tpath[300], opath[300];
    snprintf(opath, sizeof(opath), "%s/%s", orig_dir, db_files[target]);
    snprintf(tpath, sizeof(tpath), "%s/%s", fuzz_dir, db_files[target]);
    long fsz = 0L;
    ASSERT_TRUE(fuzz_file_len(opath, &fsz));

    ASSERT_TRUE(fuzz_copy_file(opath, tpath, 32L));
    nbase_db_t *bad_db = NULL;
    ASSERT_TRUE(nbase_open(fuzz_dir, &bad_db) != NBASE_OK);
    ASSERT_TRUE(bad_db == NULL);

    ASSERT_TRUE(fuzz_copy_file(opath, tpath, fsz > 1L ? fsz - 1L : 0L));
    ASSERT_TRUE(nbase_open(fuzz_dir, &bad_db) != NBASE_OK);
    ASSERT_TRUE(bad_db == NULL);

    ASSERT_TRUE(fuzz_copy_file(opath, tpath, -1L));
    ASSERT_TRUE(test_flip_file_byte(tpath, 16L));
    ASSERT_TRUE(nbase_open(fuzz_dir, &bad_db) != NBASE_OK);
    ASSERT_TRUE(bad_db == NULL);

    if (fsz > 132L) {
      ASSERT_TRUE(fuzz_copy_file(opath, tpath, -1L));
      ASSERT_TRUE(test_flip_file_byte(tpath, fsz - 4L));
      ASSERT_TRUE(nbase_open(fuzz_dir, &bad_db) != NBASE_OK);
      ASSERT_TRUE(bad_db == NULL);
    }
  }

  /* 2. LSH (.nblh) */
  char p_orig[300], p_work[300];
  nbase_lsh_config_t lcfg = nbase_lsh_config_default(dim);
  lcfg.num_tables = 4U;
  lcfg.num_bits = 6U;
  nbase_lsh_t *lsh = NULL;
  ASSERT_STATUS(nbase_lsh_create(&lcfg, &lsh), NBASE_OK);
  ASSERT_STATUS(nbase_lsh_add_batch(lsh, ids, vecs, n, dim), NBASE_OK);
  snprintf(p_orig, sizeof(p_orig), "%s/idx.nblh", orig_dir);
  snprintf(p_work, sizeof(p_work), "%s/idx.nblh", fuzz_dir);
  ASSERT_STATUS(nbase_lsh_save(lsh, p_orig), NBASE_OK);
  nbase_lsh_destroy(lsh);
  fuzz_verify_standalone_format(p_orig, p_work, fuzz_try_load_lsh);

  /* 3. Clustered (.nbcl) */
  nbase_clustered_config_t clcfg = nbase_clustered_config_default(dim);
  clcfg.target_cluster_size = 8U;
  nbase_clustered_t *cl = NULL;
  ASSERT_STATUS(nbase_clustered_create(&clcfg, &cl), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add_batch(cl, ids, vecs, n, dim), NBASE_OK);
  snprintf(p_orig, sizeof(p_orig), "%s/idx.nbcl", orig_dir);
  snprintf(p_work, sizeof(p_work), "%s/idx.nbcl", fuzz_dir);
  ASSERT_STATUS(nbase_clustered_save(cl, p_orig), NBASE_OK);
  nbase_clustered_destroy(cl);
  fuzz_verify_standalone_format(p_orig, p_work, fuzz_try_load_clustered);

  /* 4. IVF-PQ (.nbiv) */
  nbase_ivfpq_config_t icfg = nbase_ivfpq_config_default(dim, 4U, 4U);
  icfg.num_centroids = 4U;
  icfg.max_iters = 5U;
  nbase_ivfpq_t *ivf = NULL;
  ASSERT_STATUS(nbase_ivfpq_train(&icfg, vecs, n, &ivf), NBASE_OK);
  ASSERT_STATUS(nbase_ivfpq_add(ivf, ids, vecs, n), NBASE_OK);
  snprintf(p_orig, sizeof(p_orig), "%s/idx.nbiv", orig_dir);
  snprintf(p_work, sizeof(p_work), "%s/idx.nbiv", fuzz_dir);
  ASSERT_STATUS(nbase_ivfpq_save(ivf, p_orig), NBASE_OK);
  nbase_ivfpq_destroy(ivf);
  fuzz_verify_standalone_format(p_orig, p_work, fuzz_try_load_ivfpq);

  /* 5. SQ8 (.nbsq) */
  nbase_sq8_t *sq8 = NULL;
  ASSERT_STATUS(nbase_sq8_train(vecs, n, dim, &sq8), NBASE_OK);
  snprintf(p_orig, sizeof(p_orig), "%s/idx.nbsq", orig_dir);
  snprintf(p_work, sizeof(p_work), "%s/idx.nbsq", fuzz_dir);
  ASSERT_STATUS(nbase_sq8_save(sq8, p_orig), NBASE_OK);
  nbase_sq8_destroy(sq8);
  fuzz_verify_standalone_format(p_orig, p_work, fuzz_try_load_sq8);

  /* 6. BQ (.nbbq) */
  nbase_bq_t *bq = NULL;
  ASSERT_STATUS(nbase_bq_train(vecs, n, dim, &bq), NBASE_OK);
  snprintf(p_orig, sizeof(p_orig), "%s/idx.nbbq", orig_dir);
  snprintf(p_work, sizeof(p_work), "%s/idx.nbbq", fuzz_dir);
  ASSERT_STATUS(nbase_bq_save(bq, p_orig), NBASE_OK);
  nbase_bq_destroy(bq);
  fuzz_verify_standalone_format(p_orig, p_work, fuzz_try_load_bq);

  free(ids);
  free(vecs);
  test_rm_rf(orig_dir);
  test_rm_rf(fuzz_dir);
}

typedef struct db_text_batch_worker_ctx {
  const nbase_db_t *db;
} db_text_batch_worker_ctx_t;

static void *db_text_batch_reader_worker(void *raw) {
  const db_text_batch_worker_ctx_t *ctx = (const db_text_batch_worker_ctx_t *)raw;
  static const uint64_t qids[5] = {3ULL, 999ULL, 1ULL, 2ULL, 1ULL};
  for (uint32_t it = 0U; it < 200U; it++) {
    char *texts[5] = {NULL, NULL, NULL, NULL, NULL};
    bool found[5] = {false, false, false, false, false};
    if (nbase_get_text_batch(ctx->db, qids, 5U, texts, found) != NBASE_OK) {
      return (void *)(uintptr_t)1U;
    }
    bool ok = found[0] && texts[0] != NULL && strcmp(texts[0], "") == 0 &&
              !found[1] && texts[1] == NULL &&
              found[2] && texts[2] != NULL && strcmp(texts[2], "alpha vector doc") == 0 &&
              !found[3] && texts[3] == NULL &&
              found[4] && texts[4] != NULL && strcmp(texts[4], "alpha vector doc") == 0;
    nbase_free_texts(texts, 5U);
    if (!ok) {
      return (void *)(uintptr_t)1U;
    }
  }
  return NULL;
}

static void test_db_get_text_batch_and_concurrent_reads(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_config_t cfg = nbase_config_default(4U);
  cfg.seed = 4217ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float v[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_STATUS(nbase_add(db, 1ULL, v, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 2ULL, v, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 3ULL, v, 4U), NBASE_OK);

  /* DB without BM25 initialized (db->bm25 == NULL) still handles get_text_batch safely. */
  uint64_t pre_ids[2] = {1ULL, 999ULL};
  char *pre_texts[2] = {NULL, NULL};
  bool pre_found[2] = {true, true};
  ASSERT_STATUS(nbase_get_text_batch(db, pre_ids, 2U, pre_texts, pre_found), NBASE_OK);
  ASSERT_TRUE(!pre_found[0] && pre_texts[0] == NULL);
  ASSERT_TRUE(!pre_found[1] && pre_texts[1] == NULL);

  ASSERT_STATUS(nbase_get_text_batch(NULL, pre_ids, 2U, pre_texts, pre_found), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_get_text_batch(db, NULL, 2U, pre_texts, pre_found), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_get_text_batch(db, pre_ids, 2U, NULL, pre_found), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_get_text_batch(db, pre_ids, 2U, pre_texts, NULL), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_get_text_batch(db, NULL, 0U, NULL, NULL), NBASE_OK);

  ASSERT_STATUS(nbase_set_text(db, 1ULL, "alpha vector doc"), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 3ULL, ""), NBASE_OK);

  ASSERT_STATUS(nbase_save(db, dir), NBASE_OK);
  nbase_close(db);

  ASSERT_STATUS(nbase_open(dir, &db), NBASE_OK);

  uint64_t qids[5] = {3ULL, 999ULL, 1ULL, 2ULL, 1ULL};
  char *texts[5] = {NULL, NULL, NULL, NULL, NULL};
  bool found[5] = {false, false, false, false, false};
  ASSERT_STATUS(nbase_get_text_batch(db, qids, 5U, texts, found), NBASE_OK);
  ASSERT_TRUE(found[0] && texts[0] != NULL && strcmp(texts[0], "") == 0);
  ASSERT_TRUE(!found[1] && texts[1] == NULL);
  ASSERT_TRUE(found[2] && texts[2] != NULL && strcmp(texts[2], "alpha vector doc") == 0);
  ASSERT_TRUE(!found[3] && texts[3] == NULL);
  ASSERT_TRUE(found[4] && texts[4] != NULL && strcmp(texts[4], "alpha vector doc") == 0);
  nbase_free_texts(texts, 5U);

  /* Concurrent rdlock get_text_batch stress test for TSan verification. */
  db_text_batch_worker_ctx_t ctx = {db};
  pthread_t th[4];
  for (uint32_t t = 0U; t < 4U; t++) {
    ASSERT_TRUE(pthread_create(&th[t], NULL, db_text_batch_reader_worker, &ctx) == 0);
  }
  for (uint32_t t = 0U; t < 4U; t++) {
    void *ret = NULL;
    ASSERT_TRUE(pthread_join(th[t], &ret) == 0);
    ASSERT_TRUE(ret == NULL);
  }

  nbase_close(db);
  test_rm_rf(dir);
}

typedef struct db_similar_batch_ctx {
  const nbase_db_t *db;
  const uint64_t *ids;
  uint32_t qcnt;
  uint32_t k;
  nbase_search_opts_t opts;
} db_similar_batch_ctx_t;

static void *db_find_similar_batch_worker(void *arg) {
  const db_similar_batch_ctx_t *ctx = (const db_similar_batch_ctx_t *)arg;
  nbase_result_t res[16];
  uint32_t cnts[4] = {0U, 0U, 0U, 0U};
  for (int it = 0; it < 6; it++) {
    if (nbase_find_similar_batch(ctx->db, ctx->ids, ctx->qcnt, ctx->k, &ctx->opts, res, cnts) !=
        NBASE_OK) {
      return (void *)1;
    }
    for (uint32_t q = 0U; q < ctx->qcnt; q++) {
      if (cnts[q] != ctx->k) {
        return (void *)1;
      }
      for (uint32_t j = 0U; j < cnts[q]; j++) {
        size_t idx = (size_t)q * (size_t)ctx->k + (size_t)j;
        if (res[idx].id == ctx->ids[q] || !isfinite(res[idx].distance)) {
          return (void *)1;
        }
      }
    }
  }
  return NULL;
}

static void test_db_find_similar_batch_and_tsan(void) {
  const uint32_t dim = 8U;
  const uint32_t nvecs = 32U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.enable_hnsw = true;
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.seed = 2020ULL;

  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float vecs[32U * 8U];
  uint64_t all_ids[32U];
  test_fill_vectors(vecs, nvecs, dim, 991ULL);
  for (uint32_t i = 0U; i < nvecs; i++) {
    all_ids[i] = (i == 0U) ? 0ULL : (100ULL + (uint64_t)i);
    ASSERT_STATUS(nbase_add(db, all_ids[i], vecs + (size_t)i * (size_t)dim, dim), NBASE_OK);
    nbase_meta_field_t mf = {"even", {.type = NBASE_META_INT64, .as = {.i64 = (int64_t)((i % 2U) == 0U ? 1 : 0)}}};
    ASSERT_STATUS(nbase_set_meta(db, all_ids[i], &mf, 1U), NBASE_OK);
  }

  const uint64_t qids[4] = {0ULL, 105ULL, 114ULL, 131ULL};
  const uint32_t qcnt = 4U;
  const uint32_t k = 5U;
  nbase_result_t batch_1t[4U * 5U];
  nbase_result_t batch_4t[4U * 5U];
  nbase_result_t single_row[5U];
  uint32_t cnt_1t[4] = {0};
  uint32_t cnt_4t[4] = {0};

  /* Error & edge cases */
  const uint64_t missing_qids[2] = {0ULL, 999999ULL};
  ASSERT_STATUS(nbase_find_similar_batch(db, missing_qids, 2U, k, NULL, batch_1t, cnt_1t), NBASE_ERR_NOT_FOUND);
  ASSERT_STATUS(nbase_find_similar_batch(db, NULL, 0U, k, NULL, NULL, NULL), NBASE_OK);
  uint32_t zcnt[2] = {9U, 9U};
  ASSERT_STATUS(nbase_find_similar_batch(db, qids, 2U, 0U, NULL, NULL, zcnt), NBASE_OK);
  ASSERT_EQ_U64(zcnt[0], 0ULL);
  ASSERT_EQ_U64(zcnt[1], 0ULL);
  ASSERT_STATUS(nbase_find_similar_batch(db, qids, 2U, 3U, NULL, NULL, cnt_1t), NBASE_ERR_INVALID_ARG);

  /* Single-thread vs 4-thread parity & per-ID match */
  nbase_search_opts_t opts1 = nbase_search_opts_default();
  opts1.num_threads = 1U;
  nbase_search_opts_t opts4 = nbase_search_opts_default();
  opts4.num_threads = 4U;
  ASSERT_STATUS(nbase_find_similar_batch(db, qids, qcnt, k, &opts1, batch_1t, cnt_1t), NBASE_OK);
  ASSERT_STATUS(nbase_find_similar_batch(db, qids, qcnt, k, &opts4, batch_4t, cnt_4t), NBASE_OK);

  for (uint32_t q = 0U; q < qcnt; q++) {
    uint32_t scnt = 0U;
    ASSERT_STATUS(nbase_find_similar(db, qids[q], k, &opts1, single_row, &scnt), NBASE_OK);
    ASSERT_EQ_U64(cnt_1t[q], scnt);
    ASSERT_EQ_U64(cnt_4t[q], scnt);
    for (uint32_t j = 0U; j < scnt; j++) {
      size_t idx = (size_t)q * (size_t)k + (size_t)j;
      ASSERT_TRUE(batch_1t[idx].id != qids[q]);
      ASSERT_EQ_U64(batch_1t[idx].id, single_row[j].id);
      ASSERT_EQ_U64(batch_4t[idx].id, single_row[j].id);
      ASSERT_NEAR(batch_1t[idx].distance, single_row[j].distance, 1e-6, 1e-6);
      ASSERT_NEAR(batch_4t[idx].distance, single_row[j].distance, 1e-6, 1e-6);
    }
  }

  /* Filtered + metric override batch similarity */
  nbase_meta_cond_t cond = {"even", NBASE_OP_EQ, {.type = NBASE_META_INT64, .as = {.i64 = 1}}};
  nbase_meta_filter_t flt = {&cond, 1U};
  nbase_search_opts_t fopts = nbase_search_opts_default();
  fopts.meta_filter = &flt;
  fopts.metric = NBASE_METRIC_COSINE;
  fopts.use_metric_override = true;
  fopts.num_threads = 4U;
  ASSERT_STATUS(nbase_find_similar_batch(db, qids, qcnt, k, &fopts, batch_4t, cnt_4t), NBASE_OK);
  for (uint32_t q = 0U; q < qcnt; q++) {
    uint32_t scnt = 0U;
    ASSERT_STATUS(nbase_find_similar(db, qids[q], k, &fopts, single_row, &scnt), NBASE_OK);
    ASSERT_EQ_U64(cnt_4t[q], scnt);
    for (uint32_t j = 0U; j < scnt; j++) {
      size_t idx = (size_t)q * (size_t)k + (size_t)j;
      ASSERT_EQ_U64(batch_4t[idx].id, single_row[j].id);
      ASSERT_NEAR(batch_4t[idx].distance, single_row[j].distance, 1e-6, 1e-6);
      ASSERT_TRUE(batch_4t[idx].id == 0ULL || ((batch_4t[idx].id - 100ULL) % 2ULL == 0ULL));
    }
  }

  /* Concurrent TSan stress test */
  db_similar_batch_ctx_t wctx = {db, qids, qcnt, 4U, nbase_search_opts_default()};
  wctx.opts.num_threads = 2U;
  pthread_t th[4];
  for (int t = 0; t < 4; t++) {
    ASSERT_TRUE(pthread_create(&th[t], NULL, db_find_similar_batch_worker, &wctx) == 0);
  }
  for (int t = 0; t < 4; t++) {
    void *ret = NULL;
    ASSERT_TRUE(pthread_join(th[t], &ret) == 0);
    ASSERT_TRUE(ret == NULL);
  }

  nbase_close(db);
}

typedef struct db_radius_batch_tsan_ctx {
  const nbase_db_t *db;
  const float *queries;
  uint32_t qcnt;
  uint32_t dim;
  float max_distance;
  uint32_t max_results;
  nbase_search_opts_t opts;
} db_radius_batch_tsan_ctx_t;

static void *db_search_radius_batch_worker(void *raw) {
  const db_radius_batch_tsan_ctx_t *ctx = (const db_radius_batch_tsan_ctx_t *)raw;
  nbase_result_t res[24];
  uint32_t cnts[3] = {0};
  for (int it = 0; it < 6; it++) {
    if (nbase_search_radius_batch(
            ctx->db,
            ctx->queries,
            ctx->qcnt,
            ctx->dim,
            ctx->max_distance,
            ctx->max_results,
            &ctx->opts,
            res,
            cnts) != NBASE_OK) {
      return (void *)1;
    }
    for (uint32_t q = 0U; q < ctx->qcnt; q++) {
      for (uint32_t j = 0U; j < cnts[q]; j++) {
        float d = res[(size_t)q * (size_t)ctx->max_results + (size_t)j].distance;
        if (!isfinite(d) || d > ctx->max_distance) {
          return (void *)1;
        }
      }
    }
  }
  return NULL;
}

static void test_db_search_radius_batch_and_filtered_exact_tsan(void) {
  const uint32_t dim = 8U;
  const uint32_t nvecs = 40U;
  const uint32_t qcnt = 3U;
  const uint32_t max_res = 8U;

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.enable_hnsw = true;
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.seed = 2121ULL;

  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float vecs[40U * 8U];
  test_fill_vectors(vecs, nvecs, dim, 2122ULL);
  for (uint32_t i = 0U; i < nvecs; i++) {
    uint64_t id = (i == 0U) ? 0ULL : (50ULL + (uint64_t)i);
    ASSERT_STATUS(nbase_add(db, id, vecs + (size_t)i * (size_t)dim, dim), NBASE_OK);
    nbase_meta_field_t mf = {
        "tag",
        {.type = NBASE_META_STRING, .as = {.str = ((i % 2U) == 0U) ? "even" : "odd"}}};
    ASSERT_STATUS(nbase_set_meta(db, id, &mf, 1U), NBASE_OK);
  }

  float queries[3U * 8U];
  memcpy(queries, vecs, (size_t)dim * sizeof(float));
  memcpy(queries + (size_t)dim, vecs + (size_t)10U * (size_t)dim, (size_t)dim * sizeof(float));
  for (uint32_t d = 0U; d < dim; d++) {
    queries[2U * dim + d] = 100.0f + (float)d;
  }

  nbase_result_t r1[3U * 8U];
  nbase_result_t r4[3U * 8U];
  nbase_result_t rs[8U];
  uint32_t c1[3] = {0};
  uint32_t c4[3] = {0};

  /* Error & edge paths */
  ASSERT_STATUS(nbase_search_radius_batch(db, NULL, 0U, dim, 1.5f, max_res, NULL, NULL, NULL), NBASE_OK);
  uint32_t zc[2] = {5U, 5U};
  ASSERT_STATUS(nbase_search_radius_batch(db, queries, 2U, dim, 1.5f, 0U, NULL, NULL, zc), NBASE_OK);
  ASSERT_EQ_U64(zc[0], 0ULL);
  ASSERT_EQ_U64(zc[1], 0ULL);
  ASSERT_STATUS(
      nbase_search_radius_batch(db, queries, 2U, dim - 1U, 1.5f, max_res, NULL, r1, c1),
      NBASE_ERR_DIM_MISMATCH);
  ASSERT_STATUS(
      nbase_search_radius_batch(db, queries, 2U, dim, NAN, max_res, NULL, r1, c1),
      NBASE_ERR_INVALID_ARG);
  queries[dim] = NAN;
  ASSERT_STATUS(
      nbase_search_radius_batch(db, queries, 2U, dim, 1.5f, max_res, NULL, r1, c1),
      NBASE_ERR_INVALID_VECTOR);
  memcpy(queries + (size_t)dim, vecs + (size_t)10U * (size_t)dim, (size_t)dim * sizeof(float));

  /* Filtered exact mode + Manhattan override exercising 64-slot exact_flush_filtered_slots */
  nbase_meta_cond_t cond = {"tag", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "even"}}};
  nbase_meta_filter_t flt = {&cond, 1U};
  nbase_search_opts_t opts1 = nbase_search_opts_default();
  opts1.mode = NBASE_SEARCH_EXACT;
  opts1.meta_filter = &flt;
  opts1.metric = NBASE_METRIC_MANHATTAN;
  opts1.use_metric_override = true;
  opts1.num_threads = 1U;
  nbase_search_opts_t opts4 = opts1;
  opts4.num_threads = 4U;

  const float max_dist = 3.5f;
  ASSERT_STATUS(
      nbase_search_radius_batch(db, queries, qcnt, dim, max_dist, max_res, &opts1, r1, c1),
      NBASE_OK);
  ASSERT_STATUS(
      nbase_search_radius_batch(db, queries, qcnt, dim, max_dist, max_res, &opts4, r4, c4),
      NBASE_OK);
  for (uint32_t q = 0U; q < qcnt; q++) {
    uint32_t sc = 0U;
    ASSERT_STATUS(
        nbase_search_radius(
            db, queries + (size_t)q * (size_t)dim, dim, max_dist, max_res, &opts1, rs, &sc),
        NBASE_OK);
    ASSERT_EQ_U64(c1[q], sc);
    ASSERT_EQ_U64(c4[q], sc);
    for (uint32_t j = 0U; j < sc; j++) {
      size_t idx = (size_t)q * (size_t)max_res + (size_t)j;
      ASSERT_EQ_U64(r1[idx].id, rs[j].id);
      ASSERT_EQ_U64(r4[idx].id, rs[j].id);
      ASSERT_NEAR(r1[idx].distance, rs[j].distance, 1e-6, 1e-6);
      ASSERT_TRUE(r1[idx].distance <= max_dist);
      ASSERT_TRUE(r1[idx].id == 0ULL || ((r1[idx].id - 50ULL) % 2ULL == 0ULL));
    }
  }
  ASSERT_EQ_U64(c1[2], 0ULL);

  /* HNSW parallel radius batch + TSan concurrency */
  nbase_search_opts_t hopts = nbase_search_opts_default();
  hopts.mode = NBASE_SEARCH_HNSW;
  hopts.ef_search = 64U;
  hopts.num_threads = 4U;
  ASSERT_STATUS(
      nbase_search_radius_batch(db, queries, qcnt, dim, 2.0f, max_res, &hopts, r4, c4),
      NBASE_OK);
  ASSERT_TRUE(c4[0] >= 1U);

  db_radius_batch_tsan_ctx_t wctx = {db, queries, qcnt, dim, max_dist, max_res, opts1};
  wctx.opts.num_threads = 2U;
  pthread_t th[4];
  for (int t = 0; t < 4; t++) {
    ASSERT_TRUE(pthread_create(&th[t], NULL, db_search_radius_batch_worker, &wctx) == 0);
  }
  for (int t = 0; t < 4; t++) {
    void *ret = NULL;
    ASSERT_TRUE(pthread_join(th[t], &ret) == 0);
    ASSERT_TRUE(ret == NULL);
  }

  nbase_close(db);
}

void test_db_run(void) {
  RUN_TEST(test_db_upsert_with_and_without_hnsw);
  RUN_TEST(test_db_batch_search);
  RUN_TEST(test_db_persistence_mmap_and_errors);
  RUN_TEST(test_db_compact);
  RUN_TEST(test_db_batch_and_filter_delete_and_memory);
  RUN_TEST(test_db_query_lru_cache_and_invalidation);
  RUN_TEST(test_db_upsert_batch_and_get_batch);
  RUN_TEST(test_db_list_ids_and_search_radius);
  RUN_TEST(test_db_count_filtered_and_unfiltered);
  RUN_TEST(test_binary_formats_fuzz_corruption_and_truncation);
  RUN_TEST(test_db_get_text_batch_and_concurrent_reads);
  RUN_TEST(test_db_find_similar_batch_and_tsan);
  RUN_TEST(test_db_search_radius_batch_and_filtered_exact_tsan);
}
