#include "test_common.h"

#include <string.h>

static bool filter_mod_10(uint64_t id, void *ctx) {
  (void)ctx;
  return (id % 10ULL) == 0ULL;
}

static void test_hnsw_recall_500x16(void) {
  const uint32_t count = 500U;
  const uint32_t dim = 16U;
  const uint32_t k = 10U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)count * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);
  test_fill_vectors(vecs, count, dim, 1234ULL);

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.seed = 42ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  for (uint32_t i = 0U; i < count; i++) {
    ids[i] = (uint64_t)i;
    ASSERT_STATUS(nbase_add(db, ids[i], vecs + (size_t)i * dim, dim), NBASE_OK);
  }

  float query[16];
  for (uint32_t q = 0U; q < 5U; q++) {
    test_fill_vectors(query, 1U, dim, 9000ULL + q);
    nbase_result_t actual[10];
    nbase_result_t expected[10];
    uint32_t act_n = 0U;
    uint32_t exp_n = 0U;
    ASSERT_STATUS(nbase_search(db, query, dim, k, NULL, actual, &act_n), NBASE_OK);
    test_brute_force(ids, vecs, NULL, count, dim, query, k, NBASE_METRIC_EUCLIDEAN, NULL, NULL, expected, &exp_n);
    double recall = test_recall_at(actual, act_n, expected, exp_n, k);
    ASSERT_TRUE(recall >= 1.0);
  }

  nbase_close(db);
  free(vecs);
  free(ids);
}

static void test_hnsw_recall_2000x64(void) {
  const uint32_t count = 2000U;
  const uint32_t dim = 64U;
  const uint32_t k = 10U;
  const uint32_t nqueries = 10U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)count * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);
  test_fill_vectors(vecs, count, dim, 5678ULL);

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.seed = 1337ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  for (uint32_t i = 0U; i < count; i++) {
    ids[i] = (uint64_t)i;
    ASSERT_STATUS(nbase_add(db, ids[i], vecs + (size_t)i * dim, dim), NBASE_OK);
  }

  float query[64];
  double sum_recall = 0.0;
  for (uint32_t q = 0U; q < nqueries; q++) {
    test_fill_vectors(query, 1U, dim, 20000ULL + q);
    nbase_result_t actual[10];
    nbase_result_t expected[10];
    uint32_t act_n = 0U;
    uint32_t exp_n = 0U;
    ASSERT_STATUS(nbase_search(db, query, dim, k, NULL, actual, &act_n), NBASE_OK);
    test_brute_force(ids, vecs, NULL, count, dim, query, k, NBASE_METRIC_EUCLIDEAN, NULL, NULL, expected, &exp_n);
    sum_recall += test_recall_at(actual, act_n, expected, exp_n, k);
  }

  double avg_recall = sum_recall / (double)nqueries;
  ASSERT_TRUE(avg_recall >= 0.95);

  nbase_close(db);
  free(vecs);
  free(ids);
}

static void test_hnsw_ef_search_dial(void) {
  const uint32_t count = 2000U;
  const uint32_t dim = 64U;
  const uint32_t k = 10U;
  const uint32_t nqueries = 10U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)count * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);
  test_fill_vectors(vecs, count, dim, 31415ULL);

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.hnsw_m = 8U;
  cfg.hnsw_ef_construction = 64U;
  cfg.hnsw_ef_search = 10U;
  cfg.seed = 27182ULL;

  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);
  for (uint32_t i = 0U; i < count; i++) {
    ids[i] = (uint64_t)i;
    ASSERT_STATUS(nbase_add(db, ids[i], vecs + (size_t)i * dim, dim), NBASE_OK);
  }

  float query[64];
  double sum_low = 0.0;
  double sum_high = 0.0;
  for (uint32_t q = 0U; q < nqueries; q++) {
    test_fill_vectors(query, 1U, dim, 40000ULL + q);
    nbase_result_t expected[10];
    nbase_result_t act_low[10];
    nbase_result_t act_high[10];
    uint32_t exp_n = 0U;
    uint32_t low_n = 0U;
    uint32_t high_n = 0U;
    test_brute_force(ids, vecs, NULL, count, dim, query, k, NBASE_METRIC_EUCLIDEAN, NULL, NULL, expected, &exp_n);

    nbase_search_opts_t opts = nbase_search_opts_default();
    opts.ef_search = 1U;
    ASSERT_STATUS(nbase_search(db, query, dim, k, &opts, act_low, &low_n), NBASE_OK);
    opts.ef_search = 200U;
    ASSERT_STATUS(nbase_search(db, query, dim, k, &opts, act_high, &high_n), NBASE_OK);

    sum_low += test_recall_at(act_low, low_n, expected, exp_n, k);
    sum_high += test_recall_at(act_high, high_n, expected, exp_n, k);
  }

  double avg_low = sum_low / (double)nqueries;
  double avg_high = sum_high / (double)nqueries;
  ASSERT_TRUE(avg_high > avg_low);
  ASSERT_TRUE(avg_high >= 0.95);

  nbase_close(db);
  free(vecs);
  free(ids);
}

static void test_hnsw_metric_override_and_id_zero(void) {
  const uint32_t count = 800U;
  const uint32_t dim = 32U;
  const uint32_t k = 10U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)count * sizeof(uint64_t));
  ASSERT_TRUE(vecs != NULL && ids != NULL);
  test_fill_vectors(vecs, count, dim, 8888ULL);

  /* Scale vector norms so Euclidean and Cosine top-k diverge sharply. */
  for (uint32_t i = 0U; i < count; i++) {
    ids[i] = (uint64_t)i;
    float scale = 0.2f + (float)(i % 15U) * 1.5f;
    for (uint32_t d = 0U; d < dim; d++) {
      vecs[(size_t)i * dim + d] *= scale;
    }
  }

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.seed = 999ULL;

  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);
  for (uint32_t i = 0U; i < count; i++) {
    ASSERT_STATUS(nbase_add(db, ids[i], vecs + (size_t)i * dim, dim), NBASE_OK);
  }

  /* Verify id == 0 is returned when queried with its exact vector. */
  nbase_result_t zero_hit[1];
  uint32_t zero_n = 0U;
  ASSERT_STATUS(nbase_search(db, vecs, dim, 1U, NULL, zero_hit, &zero_n), NBASE_OK);
  ASSERT_EQ_U64(zero_n, 1ULL);
  ASSERT_EQ_U64(zero_hit[0].id, 0ULL);
  ASSERT_NEAR(zero_hit[0].distance, 0.0, 1e-6, 0.0);

  /* Per-search cosine override on Euclidean-built index. */
  float query[32];
  test_fill_vectors(query, 1U, dim, 7777ULL);

  nbase_search_opts_t opts = nbase_search_opts_default();
  opts.mode = NBASE_SEARCH_HNSW;
  opts.use_metric_override = true;
  opts.metric = NBASE_METRIC_COSINE;
  opts.ef_search = 150U;

  nbase_result_t actual[10];
  nbase_result_t expected[10];
  uint32_t act_n = 0U;
  uint32_t exp_n = 0U;
  ASSERT_STATUS(nbase_search(db, query, dim, k, &opts, actual, &act_n), NBASE_OK);
  test_brute_force(ids, vecs, NULL, count, dim, query, k, NBASE_METRIC_COSINE, NULL, NULL, expected, &exp_n);

  double recall = test_recall_at(actual, act_n, expected, exp_n, k);
  ASSERT_TRUE(recall >= 0.90);
  for (uint32_t i = 0U; i < act_n; i++) {
    double expected_cos = test_ref_distance(NBASE_METRIC_COSINE, query, vecs + (size_t)actual[i].id * dim, dim);
    ASSERT_NEAR(actual[i].distance, expected_cos, 1e-5, 1e-4);
  }

  nbase_close(db);
  free(vecs);
  free(ids);
}

static void test_hnsw_tombstones_rebuild_filter_and_seed(void) {
  const uint32_t count = 600U;
  const uint32_t dim = 16U;
  const uint32_t k = 10U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  uint64_t *ids = malloc((size_t)count * sizeof(uint64_t));
  bool *deleted = calloc(count, sizeof(bool));
  ASSERT_TRUE(vecs != NULL && ids != NULL && deleted != NULL);
  test_fill_vectors(vecs, count, dim, 2222ULL);

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.seed = 2026ULL;

  nbase_db_t *db1 = NULL;
  nbase_db_t *db2 = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db1), NBASE_OK);
  ASSERT_STATUS(nbase_create(&cfg, &db2), NBASE_OK);

  for (uint32_t i = 0U; i < count; i++) {
    ids[i] = (uint64_t)i;
    ASSERT_STATUS(nbase_add(db1, ids[i], vecs + (size_t)i * dim, dim), NBASE_OK);
    ASSERT_STATUS(nbase_add(db2, ids[i], vecs + (size_t)i * dim, dim), NBASE_OK);
  }

  /* Deterministic reproducibility across identical seeded builds. */
  nbase_stats_t st1, st2;
  ASSERT_STATUS(nbase_stats(db1, &st1), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db2, &st2), NBASE_OK);
  ASSERT_EQ_I64(st1.hnsw_max_level, st2.hnsw_max_level);

  float query[16];
  test_fill_vectors(query, 1U, dim, 4444ULL);
  nbase_result_t r1[10], r2[10];
  uint32_t n1 = 0U, n2 = 0U;
  ASSERT_STATUS(nbase_search(db1, query, dim, k, NULL, r1, &n1), NBASE_OK);
  ASSERT_STATUS(nbase_search(db2, query, dim, k, NULL, r2, &n2), NBASE_OK);
  ASSERT_EQ_U64(n1, n2);
  for (uint32_t i = 0U; i < n1; i++) {
    ASSERT_EQ_U64(r1[i].id, r2[i].id);
  }
  nbase_close(db2);

  /* Filter callback selects only 10% of vectors (id % 10 == 0). */
  nbase_search_opts_t fopts = nbase_search_opts_default();
  fopts.filter = filter_mod_10;
  fopts.ef_search = 100U;
  ASSERT_STATUS(nbase_search(db1, query, dim, k, &fopts, r1, &n1), NBASE_OK);
  ASSERT_EQ_U64(n1, k);
  for (uint32_t i = 0U; i < n1; i++) {
    ASSERT_TRUE((r1[i].id % 10ULL) == 0ULL);
  }

  /* Delete nearest hits (including id 0 and several top matches) -> never returned, even after save/open. */
  nbase_result_t top_before[10];
  uint32_t tb_n = 0U;
  ASSERT_STATUS(nbase_search(db1, vecs, dim, 5U, NULL, top_before, &tb_n), NBASE_OK);
  for (uint32_t i = 0U; i < tb_n; i++) {
    deleted[top_before[i].id] = true;
    ASSERT_STATUS(nbase_delete(db1, top_before[i].id), NBASE_OK);
  }

  char dir[128];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  ASSERT_STATUS(nbase_save(db1, dir), NBASE_OK);
  nbase_close(db1);

  ASSERT_STATUS(nbase_open(dir, &db1), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db1, &st1), NBASE_OK);
  ASSERT_EQ_U64(st1.hnsw_deleted_nodes, tb_n);
  ASSERT_EQ_U64(st1.hnsw_active_nodes, count - tb_n);

  nbase_result_t after[10];
  uint32_t after_n = 0U;
  ASSERT_STATUS(nbase_search(db1, vecs, dim, k, NULL, after, &after_n), NBASE_OK);
  ASSERT_EQ_U64(after_n, k);
  for (uint32_t i = 0U; i < after_n; i++) {
    ASSERT_TRUE(!deleted[after[i].id]);
  }

  /* Rebuild clears tombstones and maintains high recall. */
  ASSERT_STATUS(nbase_rebuild_hnsw(db1), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db1, &st1), NBASE_OK);
  ASSERT_EQ_U64(st1.hnsw_deleted_nodes, 0ULL);
  ASSERT_EQ_U64(st1.hnsw_active_nodes, count - tb_n);

  nbase_result_t expected[10];
  uint32_t exp_n = 0U;
  ASSERT_STATUS(nbase_search(db1, query, dim, k, NULL, after, &after_n), NBASE_OK);
  test_brute_force(ids, vecs, deleted, count, dim, query, k, NBASE_METRIC_EUCLIDEAN, NULL, NULL, expected, &exp_n);
  ASSERT_TRUE(test_recall_at(after, after_n, expected, exp_n, k) >= 0.95);

  nbase_close(db1);
  test_rm_rf(dir);
  free(vecs);
  free(ids);
  free(deleted);
}

void test_hnsw_run(void) {
  RUN_TEST(test_hnsw_recall_500x16);
  RUN_TEST(test_hnsw_recall_2000x64);
  RUN_TEST(test_hnsw_ef_search_dial);
  RUN_TEST(test_hnsw_metric_override_and_id_zero);
  RUN_TEST(test_hnsw_tombstones_rebuild_filter_and_seed);
}
