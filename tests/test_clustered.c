#include "test_common.h"

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

void test_clustered_run(void) {
  RUN_TEST(test_clustered_online_assign_search_and_recluster);
}
