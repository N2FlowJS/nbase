#include "test_common.h"

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

void test_lsh_run(void) {
  RUN_TEST(test_lsh_cosine_and_multiprobe);
  RUN_TEST(test_lsh_euclidean_save_load_and_errors);
}
