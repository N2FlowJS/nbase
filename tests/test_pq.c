#include "test_common.h"

#include <stdio.h>
#include <string.h>

static void test_kmeans_synthetic_clusters_and_edge_cases(void) {
  const uint32_t count = 400U;
  const uint32_t dim = 8U;
  const uint32_t k = 4U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  float *noise = malloc((size_t)count * dim * sizeof(float));
  float *c1 = malloc((size_t)k * dim * sizeof(float));
  float *c2 = malloc((size_t)k * dim * sizeof(float));
  uint32_t *a1 = malloc((size_t)count * sizeof(uint32_t));
  uint32_t *a2 = malloc((size_t)count * sizeof(uint32_t));
  ASSERT_TRUE(vecs != NULL && noise != NULL && c1 != NULL && c2 != NULL && a1 != NULL && a2 != NULL);

  test_fill_vectors(noise, count, dim, 1111ULL);
  static const float centers[4][2] = {{-5.0f, -5.0f}, {-5.0f, 5.0f}, {5.0f, -5.0f}, {5.0f, 5.0f}};
  for (uint32_t i = 0U; i < count; i++) {
    uint32_t grp = i % k;
    for (uint32_t d = 0U; d < dim; d++) {
      float base = (d < 2U) ? centers[grp][d] : 0.0f;
      vecs[(size_t)i * dim + d] = base + (noise[(size_t)i * dim + d] - 0.5f) * 0.4f;
    }
  }

  ASSERT_STATUS(nbase_kmeans_train(vecs, count, dim, k, 30U, 99ULL, c1, a1), NBASE_OK);
  ASSERT_STATUS(nbase_kmeans_train(vecs, count, dim, k, 30U, 99ULL, c2, a2), NBASE_OK);

  /* Deterministic reproducibility with identical seed. */
  for (uint32_t i = 0U; i < count; i++) {
    ASSERT_EQ_U64(a1[i], a2[i]);
  }
  for (uint32_t i = 0U; i < k * dim; i++) {
    ASSERT_NEAR(c1[i], c2[i], 1e-6, 0.0);
  }

  /* Every true center is closely recovered by one learned centroid. */
  for (uint32_t grp = 0U; grp < k; grp++) {
    double best = 1e9;
    for (uint32_t c = 0U; c < k; c++) {
      double dx = (double)c1[c * dim] - (double)centers[grp][0];
      double dy = (double)c1[c * dim + 1U] - (double)centers[grp][1];
      double dist = sqrt(dx * dx + dy * dy);
      if (dist < best) {
        best = dist;
      }
    }
    ASSERT_TRUE(best < 0.25);
  }

  /* Duplicate points with k == count must recover without NaN. */
  float dup[4 * 4] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  float dup_c[4 * 4];
  ASSERT_STATUS(nbase_kmeans_train(dup, 4U, 4U, 4U, 10U, 1ULL, dup_c, NULL), NBASE_OK);
  for (uint32_t i = 0U; i < 16U; i++) {
    ASSERT_TRUE(isfinite(dup_c[i]));
  }

  free(vecs);
  free(noise);
  free(c1);
  free(c2);
  free(a1);
  free(a2);
}

typedef struct pq_hit {
  uint32_t idx;
  float dist;
} pq_hit_t;

static int cmp_pq_hit(const void *pa, const void *pb) {
  const pq_hit_t *a = (const pq_hit_t *)pa;
  const pq_hit_t *b = (const pq_hit_t *)pb;
  if (a->dist < b->dist) {
    return -1;
  }
  if (a->dist > b->dist) {
    return 1;
  }
  return (a->idx < b->idx) ? -1 : (a->idx > b->idx);
}

static void test_pq_encode_decode_adc_and_persistence(void) {
  const uint32_t count = 500U;
  const uint32_t dim = 32U;
  const uint32_t m = 8U;
  const uint32_t ks = 64U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  float *noise = malloc((size_t)count * dim * sizeof(float));
  uint8_t *codes = malloc((size_t)count * m * sizeof(uint8_t));
  float *decoded = malloc((size_t)count * dim * sizeof(float));
  float *lut = malloc((size_t)m * ks * sizeof(float));
  ASSERT_TRUE(vecs != NULL && noise != NULL && codes != NULL && decoded != NULL && lut != NULL);

  float *anchors = malloc((size_t)100U * dim * sizeof(float));
  ASSERT_TRUE(anchors != NULL);
  test_fill_vectors(anchors, 100U, dim, 3333ULL);
  test_fill_vectors(noise, count, dim, 2222ULL);
  for (uint32_t i = 0U; i < count; i++) {
    uint32_t cluster = i % 100U;
    for (uint32_t d = 0U; d < dim; d++) {
      float anchor = anchors[(size_t)cluster * dim + d] * 3.0f;
      vecs[(size_t)i * dim + d] = anchor + (noise[(size_t)i * dim + d] - 0.5f) * 0.25f;
    }
  }
  free(anchors);

  nbase_pq_t *pq = NULL;
  ASSERT_STATUS(nbase_pq_train(vecs, count, dim, m, ks, 20U, 777ULL, &pq), NBASE_OK);
  ASSERT_EQ_U64(nbase_pq_dim(pq), dim);
  ASSERT_EQ_U64(nbase_pq_num_subvectors(pq), m);
  ASSERT_EQ_U64(nbase_pq_num_centroids(pq), ks);

  ASSERT_STATUS(nbase_pq_encode(pq, vecs, count, codes), NBASE_OK);
  ASSERT_STATUS(nbase_pq_decode(pq, codes, count, decoded), NBASE_OK);

  double mse = 0.0;
  for (size_t i = 0U; i < (size_t)count * dim; i++) {
    ASSERT_TRUE(isfinite(decoded[i]));
    double diff = (double)vecs[i] - (double)decoded[i];
    mse += diff * diff;
  }
  mse /= (double)((size_t)count * dim);
  ASSERT_TRUE(mse < 0.05);

  /* ADC distance must match exact squared L2 between query and decoded vector. */
  const float *query = vecs + 17U * dim;
  ASSERT_STATUS(nbase_pq_build_lut(pq, query, lut), NBASE_OK);
  for (uint32_t i = 0U; i < 20U; i++) {
    float adc = nbase_pq_adc_distance(pq, lut, codes + (size_t)i * m);
    float exact_dec = nbase_distance(NBASE_METRIC_SQUARED_EUCLIDEAN, query, decoded + (size_t)i * dim, dim);
    ASSERT_NEAR(adc, exact_dec, 1e-3, 1e-4);
  }

  /* ADC top-10 recall against exact squared L2 on the dataset. */
  pq_hit_t *adc_hits = malloc(count * sizeof(pq_hit_t));
  pq_hit_t *exact_hits = malloc(count * sizeof(pq_hit_t));
  ASSERT_TRUE(adc_hits != NULL && exact_hits != NULL);
  for (uint32_t i = 0U; i < count; i++) {
    adc_hits[i].idx = i;
    adc_hits[i].dist = nbase_pq_adc_distance(pq, lut, codes + (size_t)i * m);
    exact_hits[i].idx = i;
    exact_hits[i].dist = nbase_distance(NBASE_METRIC_SQUARED_EUCLIDEAN, query, vecs + (size_t)i * dim, dim);
  }
  qsort(adc_hits, count, sizeof(pq_hit_t), cmp_pq_hit);
  qsort(exact_hits, count, sizeof(pq_hit_t), cmp_pq_hit);

  uint32_t overlap = 0U;
  for (uint32_t i = 0U; i < 10U; i++) {
    for (uint32_t j = 0U; j < 10U; j++) {
      if (exact_hits[i].idx == adc_hits[j].idx) {
        overlap++;
        break;
      }
    }
  }
  ASSERT_TRUE((double)overlap / 10.0 >= 0.70);

  /* Save/load round trip + CRC32 corruption detection. */
  char dir[128];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  char pq_path[256];
  snprintf(pq_path, sizeof(pq_path), "%s/codebook.nbq", dir);
  ASSERT_STATUS(nbase_pq_save(pq, pq_path), NBASE_OK);
  nbase_pq_destroy(pq);
  pq = NULL;

  ASSERT_STATUS(nbase_pq_load(pq_path, &pq), NBASE_OK);
  float *lut2 = malloc((size_t)m * ks * sizeof(float));
  ASSERT_TRUE(lut2 != NULL);
  ASSERT_STATUS(nbase_pq_build_lut(pq, query, lut2), NBASE_OK);
  for (uint32_t i = 0U; i < m * ks; i++) {
    ASSERT_NEAR(lut2[i], lut[i], 1e-6, 0.0);
  }
  free(lut2);
  nbase_pq_destroy(pq);

  ASSERT_TRUE(test_flip_file_byte(pq_path, 160L));
  ASSERT_STATUS(nbase_pq_load(pq_path, &pq), NBASE_ERR_CORRUPT);

  test_rm_rf(dir);
  free(adc_hits);
  free(exact_hits);
  free(vecs);
  free(noise);
  free(codes);
  free(decoded);
  free(lut);
}

static void test_pq_adc_distance_batch(void) {
  const uint32_t dim = 16U;
  const uint32_t m = 8U;
  const uint32_t count = 300U;
  float *vecs = malloc((size_t)count * dim * sizeof(float));
  ASSERT_TRUE(vecs != NULL);
  test_fill_vectors(vecs, count, dim, 2026ULL);

  /* Test fast num_centroids == 256 path AND general num_centroids < 256 path. */
  static const uint32_t ks_cases[2] = {256U, 32U};
  for (uint32_t ci = 0U; ci < 2U; ci++) {
    uint32_t ks = ks_cases[ci];
    nbase_pq_t *pq = NULL;
    ASSERT_STATUS(nbase_pq_train(vecs, count, dim, m, ks, 8U, 42ULL + ci, &pq), NBASE_OK);

    uint8_t *codes = malloc((size_t)count * m);
    float *lut = malloc((size_t)m * ks * sizeof(float));
    float *batch_dists = malloc((size_t)count * sizeof(float));
    ASSERT_TRUE(codes != NULL && lut != NULL && batch_dists != NULL);

    ASSERT_STATUS(nbase_pq_encode(pq, vecs, count, codes), NBASE_OK);
    ASSERT_STATUS(nbase_pq_build_lut(pq, vecs, lut), NBASE_OK);
    static const uint32_t sub_counts[7] = {0U, 1U, 2U, 3U, 4U, 5U, 7U};
    for (uint32_t si = 0U; si < 7U; si++) {
      uint32_t sc = sub_counts[si];
      ASSERT_STATUS(nbase_pq_adc_distance_batch(pq, lut, codes, sc, batch_dists), NBASE_OK);
      for (uint32_t i = 0U; i < sc; i++) {
        float single = nbase_pq_adc_distance(pq, lut, codes + (size_t)i * m);
        ASSERT_NEAR(batch_dists[i], single, 1e-6, 0.0);
      }
    }
    ASSERT_STATUS(nbase_pq_adc_distance_batch(pq, lut, codes, count, batch_dists), NBASE_OK);

    for (uint32_t i = 0U; i < count; i++) {
      float single = nbase_pq_adc_distance(pq, lut, codes + (size_t)i * m);
      ASSERT_NEAR(batch_dists[i], single, 1e-6, 0.0);
    }

    if (ks < 256U) {
      uint8_t bad_code[8] = {0};
      bad_code[3] = (uint8_t)ks;
      float out_d = 0.0f;
      ASSERT_STATUS(nbase_pq_adc_distance_batch(pq, lut, bad_code, 1U, &out_d), NBASE_ERR_INVALID_ARG);
    }
    ASSERT_STATUS(nbase_pq_adc_distance_batch(pq, lut, NULL, 1U, batch_dists), NBASE_ERR_INVALID_ARG);

    free(codes);
    free(lut);
    free(batch_dists);
    nbase_pq_destroy(pq);
  }

  free(vecs);
}

static void test_pq_train_parallel_determinism_tsan(void) {
  const uint32_t dim = 32U;
  const uint32_t m = 8U;
  const uint32_t ks = 64U;
  const uint32_t count = 400U; /* count * ks = 25600 >= 16384 triggers MT PQ subvector training */

  float *vecs = malloc((size_t)count * (size_t)dim * sizeof(float));
  uint8_t *codes1 = malloc((size_t)count * (size_t)m);
  uint8_t *codes2 = malloc((size_t)count * (size_t)m);
  float *lut1 = malloc((size_t)m * (size_t)ks * sizeof(float));
  float *lut2 = malloc((size_t)m * (size_t)ks * sizeof(float));
  ASSERT_TRUE(vecs != NULL && codes1 != NULL && codes2 != NULL && lut1 != NULL && lut2 != NULL);

  test_fill_vectors(vecs, count, dim, 5150ULL);

  nbase_pq_t *pq1 = NULL;
  nbase_pq_t *pq2 = NULL;
  ASSERT_STATUS(nbase_pq_train(vecs, count, dim, m, ks, 10U, 777ULL, &pq1), NBASE_OK);
  ASSERT_STATUS(nbase_pq_train(vecs, count, dim, m, ks, 10U, 777ULL, &pq2), NBASE_OK);

  ASSERT_STATUS(nbase_pq_encode(pq1, vecs, count, codes1), NBASE_OK);
  ASSERT_STATUS(nbase_pq_encode(pq2, vecs, count, codes2), NBASE_OK);
  ASSERT_TRUE(memcmp(codes1, codes2, (size_t)count * (size_t)m) == 0);

  ASSERT_STATUS(nbase_pq_build_lut(pq1, vecs, lut1), NBASE_OK);
  ASSERT_STATUS(nbase_pq_build_lut(pq2, vecs, lut2), NBASE_OK);
  for (uint32_t i = 0U; i < m * ks; i++) {
    ASSERT_NEAR(lut1[i], lut2[i], 0.0, 0.0);
  }
  ASSERT_NEAR(nbase_pq_adc_distance(pq1, lut1, codes1), 0.5, 0.5, 1.0);

  nbase_pq_destroy(pq1);
  nbase_pq_destroy(pq2);
  free(lut2);
  free(lut1);
  free(codes2);
  free(codes1);
  free(vecs);
}

void test_pq_run(void) {
  RUN_TEST(test_kmeans_synthetic_clusters_and_edge_cases);
  RUN_TEST(test_pq_encode_decode_adc_and_persistence);
  RUN_TEST(test_pq_adc_distance_batch);
  RUN_TEST(test_pq_train_parallel_determinism_tsan);
}
