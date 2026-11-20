#include "test_common.h"

#include <stdio.h>
#include <string.h>

static bool keep_even_id(uint64_t id, void *ctx) {
  (void)ctx;
  return (id % 2ULL) == 0ULL;
}

static nbase_status_t flip_file_byte(const char *path, long offset) {
  FILE *fp = fopen(path, "r+b");
  if (!fp) {
    return NBASE_ERR_IO;
  }
  if (fseek(fp, offset, SEEK_SET) != 0) {
    fclose(fp);
    return NBASE_ERR_IO;
  }
  int c = fgetc(fp);
  if (c == EOF) {
    fclose(fp);
    return NBASE_ERR_IO;
  }
  if (fseek(fp, offset, SEEK_SET) != 0) {
    fclose(fp);
    return NBASE_ERR_IO;
  }
  unsigned char flipped = (unsigned char)(c ^ 0xFF);
  if (fputc(flipped, fp) == EOF) {
    fclose(fp);
    return NBASE_ERR_IO;
  }
  fclose(fp);
  return NBASE_OK;
}

static void test_store_boundary_ids_and_validation(void) {
  nbase_config_t cfg = nbase_config_default(4U);
  cfg.enable_hnsw = false;

  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float v0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  float vmax[4] = {0.0f, 1.0f, 0.0f, 0.0f};
  float out[4] = {0};

  ASSERT_STATUS(nbase_add(db, 0ULL, v0, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, UINT64_MAX, vmax, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 0ULL, v0, 4U), NBASE_ERR_ALREADY_EXISTS);
  ASSERT_STATUS(nbase_add(db, UINT64_MAX, vmax, 4U), NBASE_ERR_ALREADY_EXISTS);

  ASSERT_TRUE(nbase_contains(db, 0ULL));
  ASSERT_TRUE(nbase_contains(db, UINT64_MAX));
  ASSERT_TRUE(!nbase_contains(db, 42ULL));

  ASSERT_STATUS(nbase_get(db, 0ULL, out, 4U), NBASE_OK);
  ASSERT_NEAR(out[0], 1.0, 1e-6, 0.0);
  ASSERT_STATUS(nbase_get(db, UINT64_MAX, out, 4U), NBASE_OK);
  ASSERT_NEAR(out[1], 1.0, 1e-6, 0.0);

  float vnan[4] = {1.0f, NAN, 0.0f, 0.0f};
  float vinf[4] = {1.0f, INFINITY, 0.0f, 0.0f};
  ASSERT_STATUS(nbase_add(db, 5ULL, vnan, 4U), NBASE_ERR_INVALID_VECTOR);
  ASSERT_STATUS(nbase_add(db, 6ULL, vinf, 4U), NBASE_ERR_INVALID_VECTOR);

  nbase_result_t res[4];
  uint32_t count = 0U;
  ASSERT_STATUS(nbase_search(db, v0, 4U, 2U, NULL, res, &count), NBASE_OK);
  ASSERT_EQ_U64(count, 2ULL);
  ASSERT_EQ_U64(res[0].id, 0ULL);
  ASSERT_EQ_U64(res[1].id, UINT64_MAX);

  ASSERT_STATUS(nbase_delete(db, 0ULL), NBASE_OK);
  ASSERT_STATUS(nbase_delete(db, 0ULL), NBASE_ERR_NOT_FOUND);
  ASSERT_TRUE(!nbase_contains(db, 0ULL));
  ASSERT_STATUS(nbase_get(db, 0ULL, out, 4U), NBASE_ERR_NOT_FOUND);

  ASSERT_STATUS(nbase_search(db, v0, 4U, 2U, NULL, res, &count), NBASE_OK);
  ASSERT_EQ_U64(count, 1ULL);
  ASSERT_EQ_U64(res[0].id, UINT64_MAX);

  nbase_close(db);
}

static void test_store_exact_oracle_and_filter(void) {
  const uint32_t count = 300U;
  const uint32_t dim = 16U;
  const uint32_t k = 10U;

  float *vecs = malloc((size_t)count * (size_t)dim * sizeof(float));
  uint64_t *ids = malloc((size_t)count * sizeof(uint64_t));
  bool *deleted = calloc(count, sizeof(bool));
  ASSERT_TRUE(vecs != NULL && ids != NULL && deleted != NULL);

  test_fill_vectors(vecs, count, dim, 777ULL);

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  for (uint32_t i = 0U; i < count; i++) {
    ids[i] = (uint64_t)i;
    ASSERT_STATUS(nbase_add(db, ids[i], vecs + (size_t)i * dim, dim), NBASE_OK);
  }

  for (uint32_t i = 0U; i < count; i += 17U) {
    deleted[i] = true;
    ASSERT_STATUS(nbase_delete(db, ids[i]), NBASE_OK);
  }

  float query[16];
  test_fill_vectors(query, 1U, dim, 999ULL);

  nbase_result_t actual[10];
  nbase_result_t expected[10];
  uint32_t act_n = 0U;
  uint32_t exp_n = 0U;

  ASSERT_STATUS(nbase_search(db, query, dim, k, NULL, actual, &act_n), NBASE_OK);
  test_brute_force(ids, vecs, deleted, count, dim, query, k, NBASE_METRIC_EUCLIDEAN, NULL, NULL, expected, &exp_n);
  ASSERT_EQ_U64(act_n, exp_n);
  for (uint32_t i = 0U; i < exp_n; i++) {
    ASSERT_EQ_U64(actual[i].id, expected[i].id);
    ASSERT_NEAR(actual[i].distance, expected[i].distance, 1e-5, 1e-4);
  }

  /* Filter callback + cosine metric override. */
  nbase_search_opts_t opts = nbase_search_opts_default();
  opts.mode = NBASE_SEARCH_EXACT;
  opts.use_metric_override = true;
  opts.metric = NBASE_METRIC_COSINE;
  opts.filter = keep_even_id;

  ASSERT_STATUS(nbase_search(db, query, dim, k, &opts, actual, &act_n), NBASE_OK);
  test_brute_force(ids, vecs, deleted, count, dim, query, k, NBASE_METRIC_COSINE, keep_even_id, NULL, expected, &exp_n);
  ASSERT_EQ_U64(act_n, exp_n);
  for (uint32_t i = 0U; i < exp_n; i++) {
    ASSERT_TRUE((actual[i].id % 2ULL) == 0ULL);
    ASSERT_EQ_U64(actual[i].id, expected[i].id);
  }

  nbase_close(db);
  free(vecs);
  free(ids);
  free(deleted);
}

static void test_store_mmap_empty_and_crc_corruption(void) {
  char dir[128];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  char vec_path[256];
  snprintf(vec_path, sizeof(vec_path), "%s/vectors.nbv", dir);

  /* Empty DB save/open/mmap round trip. */
  nbase_config_t cfg = nbase_config_default(7U);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);
  ASSERT_STATUS(nbase_save(db, dir), NBASE_OK);
  nbase_close(db);

  db = NULL;
  ASSERT_STATUS(nbase_open_mmap(dir, &db), NBASE_OK);
  nbase_stats_t st;
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_TRUE(st.mmap_backed);
  ASSERT_EQ_U64(st.active_vectors, 0ULL);
  nbase_close(db);

  /* Odd dimension (7 floats = 28 bytes) verifies 8-byte alignment of ids_offset in mmap mode. */
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);
  float vecs[21];
  test_fill_vectors(vecs, 3U, 7U, 55ULL);
  ASSERT_STATUS(nbase_add(db, 10ULL, vecs, 7U), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 20ULL, vecs + 7U, 7U), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 30ULL, vecs + 14U, 7U), NBASE_OK);
  ASSERT_STATUS(nbase_delete(db, 20ULL), NBASE_OK);
  ASSERT_STATUS(nbase_save(db, dir), NBASE_OK);
  nbase_close(db);

  ASSERT_STATUS(nbase_open_mmap(dir, &db), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_TRUE(st.mmap_backed);
  ASSERT_EQ_U64(st.active_vectors, 2ULL);
  ASSERT_EQ_U64(st.tombstones, 1ULL);
  ASSERT_TRUE(nbase_contains(db, 10ULL));
  ASSERT_TRUE(!nbase_contains(db, 20ULL));
  ASSERT_STATUS(nbase_add(db, 40ULL, vecs, 7U), NBASE_ERR_UNSUPPORTED);
  ASSERT_STATUS(nbase_delete(db, 10ULL), NBASE_ERR_UNSUPPORTED);
  ASSERT_STATUS(nbase_upsert(db, 10ULL, vecs, 7U), NBASE_ERR_UNSUPPORTED);

  nbase_result_t res[2];
  uint32_t out_n = 0U;
  ASSERT_STATUS(nbase_search(db, vecs + 14U, 7U, 2U, NULL, res, &out_n), NBASE_OK);
  ASSERT_EQ_U64(out_n, 2ULL);
  ASSERT_EQ_U64(res[0].id, 30ULL);
  nbase_close(db);

  /* Corrupt payload byte -> both open and open_mmap reject with NBASE_ERR_CORRUPT. */
  ASSERT_STATUS(flip_file_byte(vec_path, 132L), NBASE_OK);
  ASSERT_STATUS(nbase_open(dir, &db), NBASE_ERR_CORRUPT);
  ASSERT_STATUS(nbase_open_mmap(dir, &db), NBASE_ERR_CORRUPT);

  /* Restore payload byte and corrupt header byte -> rejected as corrupt. */
  ASSERT_STATUS(flip_file_byte(vec_path, 132L), NBASE_OK);
  ASSERT_STATUS(flip_file_byte(vec_path, 12L), NBASE_OK);
  ASSERT_STATUS(nbase_open(dir, &db), NBASE_ERR_CORRUPT);
  ASSERT_STATUS(nbase_open_mmap(dir, &db), NBASE_ERR_CORRUPT);

  test_rm_rf(dir);
}

void test_store_run(void) {
  RUN_TEST(test_store_boundary_ids_and_validation);
  RUN_TEST(test_store_exact_oracle_and_filter);
  RUN_TEST(test_store_mmap_empty_and_crc_corruption);
}
