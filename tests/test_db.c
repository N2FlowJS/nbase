#include "test_common.h"

#include <math.h>
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
  ASSERT_TRUE(strcmp(nbase_version(), "1.6.0") == 0);

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
}
