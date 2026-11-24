#include "test_common.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

static void test_clustered_online_assign_search_and_recluster(void) {
  nbase_clustered_config_t cfg = nbase_clustered_config_default(4U);
  cfg.target_cluster_size = 4U;
  cfg.max_clusters = 16U;
  cfg.distance_threshold = 0.35f;
  cfg.seed = 2025ULL;

  nbase_clustered_t *cl = NULL;
  ASSERT_STATUS(nbase_clustered_create(&cfg, &cl), NBASE_OK);

  /* Two separated clusters around (0,0,0,0) and (5,5,0,0) */
  float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float a1[4] = {0.1f, 0.0f, 0.0f, 0.0f};
  float a2[4] = {0.0f, 0.1f, 0.0f, 0.0f};
  float b0[4] = {5.0f, 5.0f, 0.0f, 0.0f};
  float b1[4] = {5.1f, 5.0f, 0.0f, 0.0f};
  float c0[4] = {10.0f, 0.0f, 0.0f, 0.0f};

  ASSERT_STATUS(nbase_clustered_add(cl, 1ULL, a0, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 2ULL, a1, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 3ULL, a2, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 4ULL, b0, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 5ULL, b1, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 6ULL, c0, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 1ULL, a0, 4U), NBASE_ERR_ALREADY_EXISTS);

  nbase_clustered_stats_t st;
  ASSERT_STATUS(nbase_clustered_stats(cl, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, 6ULL);
  ASSERT_EQ_U64(st.cluster_count, 3ULL);
  ASSERT_TRUE(st.memory_bytes > 0ULL);

  nbase_result_t hits[3];
  uint32_t cnt = 0U;
  ASSERT_STATUS(nbase_clustered_search(cl, b0, 4U, 2U, 1U, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 2ULL);
  ASSERT_EQ_U64(hits[0].id, 4ULL);
  ASSERT_EQ_U64(hits[1].id, 5ULL);

  /* Upsert moves id 6 from c0 near a0 and delete removes id 5 */
  float a3[4] = {0.05f, 0.05f, 0.0f, 0.0f};
  ASSERT_STATUS(nbase_clustered_upsert(cl, 6ULL, a3, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_delete(cl, 5ULL), NBASE_OK);
  ASSERT_TRUE(!nbase_clustered_contains(cl, 5ULL));

  float out_v[4] = {0};
  ASSERT_STATUS(nbase_clustered_get(cl, 6ULL, out_v, 4U), NBASE_OK);
  ASSERT_NEAR(out_v[0], 0.05, 1e-6, 0.0);

  /* K-Means recluster into 2 clusters */
  ASSERT_STATUS(nbase_clustered_recluster(cl, 2U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_stats(cl, &st), NBASE_OK);
  ASSERT_EQ_U64(st.cluster_count, 2ULL);
  ASSERT_EQ_U64(st.active_vectors, 5ULL);

  ASSERT_STATUS(nbase_clustered_search(cl, a0, 4U, 3U, 0U, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 3ULL);
  ASSERT_EQ_U64(hits[0].id, 1ULL);

  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  char path[300];
  snprintf(path, sizeof(path), "%s/index.nbcl", dir);
  ASSERT_STATUS(nbase_clustered_save(cl, path), NBASE_OK);
  nbase_clustered_destroy(cl);

  nbase_clustered_t *loaded = NULL;
  ASSERT_STATUS(nbase_clustered_load(path, &loaded), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_stats(loaded, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, 5ULL);
  ASSERT_EQ_U64(st.cluster_count, 2ULL);
  ASSERT_STATUS(nbase_clustered_search(loaded, b0, 4U, 1U, 1U, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 1ULL);
  ASSERT_EQ_U64(hits[0].id, 4ULL);
  nbase_clustered_destroy(loaded);

  ASSERT_TRUE(test_flip_file_byte(path, 20L));
  ASSERT_STATUS(nbase_clustered_load(path, &loaded), NBASE_ERR_CORRUPT);

  test_rm_rf(dir);
}

static bool cl_even_filter(uint64_t id, void *ctx) {
  (void)ctx;
  return (id % 2ULL) == 0ULL;
}

typedef struct cl_mt_arg {
  nbase_clustered_t *cl;
  int err;
} cl_mt_arg_t;

static void *cl_reader_thread(void *raw) {
  cl_mt_arg_t *arg = (cl_mt_arg_t *)raw;
  float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  nbase_result_t hits[4];
  for (int iter = 0; iter < 50; iter++) {
    uint32_t cnt = 0U;
    nbase_status_t st = nbase_clustered_search(arg->cl, q, 4U, 4U, 0U, hits, &cnt);
    if (st != NBASE_OK) {
      arg->err = 1;
      break;
    }
    nbase_clustered_stats_t st_info;
    if (nbase_clustered_stats(arg->cl, &st_info) != NBASE_OK) {
      arg->err = 1;
      break;
    }
  }
  return NULL;
}

static void test_clustered_batch_filtered_and_concurrent(void) {
  nbase_clustered_config_t cfg = nbase_clustered_config_default(4U);
  cfg.target_cluster_size = 4U;
  cfg.max_clusters = 16U;
  cfg.distance_threshold = 0.5f;

  nbase_clustered_t *cl = NULL;
  ASSERT_STATUS(nbase_clustered_create(&cfg, &cl), NBASE_OK);

  uint64_t ids[6] = {10ULL, 11ULL, 12ULL, 13ULL, 14ULL, 15ULL};
  float vecs[24] = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.99f, 0.01f, 0.0f, 0.0f,
      0.98f, 0.02f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.99f, 0.01f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f
  };
  ASSERT_STATUS(nbase_clustered_add_batch(cl, ids, vecs, 6U, 4U), NBASE_OK);

  /* Duplicate ID inside add_batch fails cleanly without inserting partial batch */
  uint64_t bad_ids[2] = {100ULL, 100ULL};
  ASSERT_STATUS(nbase_clustered_add_batch(cl, bad_ids, vecs, 2U, 4U), NBASE_ERR_ALREADY_EXISTS);
  ASSERT_TRUE(!nbase_clustered_contains(cl, 100ULL));

  /* Filtered search selects only even IDs */
  float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  nbase_result_t hits[4];
  uint32_t cnt = 0U;
  ASSERT_STATUS(
      nbase_clustered_search_filtered(cl, q, 4U, 4U, 0U, cl_even_filter, NULL, hits, &cnt),
      NBASE_OK);
  ASSERT_EQ_U64(cnt, 3ULL);
  for (uint32_t i = 0U; i < cnt; i++) {
    ASSERT_TRUE((hits[i].id % 2ULL) == 0ULL);
  }
  ASSERT_EQ_U64(hits[0].id, 10ULL);

  /* Batch delete removes found IDs and ignores missing IDs */
  uint64_t del_ids[3] = {11ULL, 13ULL, 999ULL};
  uint32_t deleted = 0U;
  ASSERT_STATUS(nbase_clustered_delete_batch(cl, del_ids, 3U, &deleted), NBASE_OK);
  ASSERT_EQ_U64(deleted, 2ULL);
  ASSERT_TRUE(!nbase_clustered_contains(cl, 11ULL));
  ASSERT_TRUE(!nbase_clustered_contains(cl, 13ULL));
  ASSERT_TRUE(nbase_clustered_contains(cl, 12ULL));

  /* Concurrent readers + writer verify rwlock under TSan */
  pthread_t readers[2];
  cl_mt_arg_t args[2] = {{cl, 0}, {cl, 0}};
  for (int t = 0; t < 2; t++) {
    ASSERT_TRUE(pthread_create(&readers[t], NULL, cl_reader_thread, &args[t]) == 0);
  }
  for (uint64_t wid = 1000ULL; wid < 1030ULL; wid++) {
    float wv[4] = {(float)(wid % 5ULL), 0.1f, 0.0f, 0.0f};
    ASSERT_STATUS(nbase_clustered_upsert(cl, wid, wv, 4U), NBASE_OK);
  }
  for (int t = 0; t < 2; t++) {
    pthread_join(readers[t], NULL);
    ASSERT_EQ_I64(args[t].err, 0);
  }

  nbase_clustered_destroy(cl);
}

void test_clustered_run(void) {
  RUN_TEST(test_clustered_online_assign_search_and_recluster);
  RUN_TEST(test_clustered_batch_filtered_and_concurrent);
}
