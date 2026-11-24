#include "test_common.h"

#include <math.h>
#include <stdio.h>

static void test_sq8_train_encode_decode_and_distance(void) {
  const uint32_t dim = 32U;
  const uint32_t n = 256U;
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  uint8_t *codes = malloc((size_t)n * dim);
  float *decoded = malloc((size_t)n * dim * sizeof(float));
  ASSERT_TRUE(vecs != NULL && codes != NULL && decoded != NULL);

  test_fill_vectors(vecs, n, dim, 7001ULL);
  for (size_t i = 0U; i < (size_t)n * dim; i++) {
    vecs[i] = vecs[i] * 10.0f - 5.0f;
  }

  nbase_sq8_t *sq8 = NULL;
  ASSERT_STATUS(nbase_sq8_train(vecs, n, dim, &sq8), NBASE_OK);
  ASSERT_EQ_U64(nbase_sq8_dim(sq8), dim);

  ASSERT_STATUS(nbase_sq8_encode(sq8, vecs, n, codes), NBASE_OK);
  ASSERT_STATUS(nbase_sq8_decode(sq8, codes, n, decoded), NBASE_OK);

  /* Quantization max per-coordinate step is span / 255 (~10 / 255 < 0.04, half-step < 0.025). */
  double max_err = 0.0;
  for (size_t i = 0U; i < (size_t)n * dim; i++) {
    double e = fabs((double)vecs[i] - (double)decoded[i]);
    if (e > max_err) {
      max_err = e;
    }
  }
  ASSERT_TRUE(max_err < 0.03);

  float q[32];
  test_fill_vectors(q, 1U, dim, 7002ULL);
  for (uint32_t d = 0U; d < dim; d++) {
    q[d] = q[d] * 10.0f - 5.0f;
  }

  uint32_t best_exact = 0U;
  double best_exact_d = 1e18;
  uint32_t best_asym = 0U;
  double best_asym_d = 1e18;
  for (uint32_t i = 0U; i < n; i++) {
    double ed = test_ref_distance(NBASE_METRIC_SQUARED_EUCLIDEAN, q, vecs + (size_t)i * dim, dim);
    if (ed < best_exact_d) {
      best_exact_d = ed;
      best_exact = i;
    }
    double ad = (double)nbase_sq8_l2_sq_asymmetric(sq8, q, codes + (size_t)i * dim);
    if (ad < best_asym_d) {
      best_asym_d = ad;
      best_asym = i;
    }
  }
  ASSERT_EQ_U64(best_asym, best_exact);

  float sym00 = nbase_sq8_l2_sq_symmetric(sq8, codes, codes);
  ASSERT_TRUE(fabsf(sym00) < 1e-6f);
  float sym01 = nbase_sq8_l2_sq_symmetric(sq8, codes, codes + dim);
  float asym_d = nbase_sq8_l2_sq_asymmetric(sq8, decoded, codes + dim);
  ASSERT_TRUE(fabsf(sym01 - asym_d) < 1e-3f);

  nbase_sq8_destroy(sq8);
  free(vecs);
  free(codes);
  free(decoded);
}

static void test_sq8_persistence_and_corruption(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  char path[300];
  snprintf(path, sizeof(path), "%s/quant.nbsq", dir);

  const uint32_t dim = 8U;
  float vecs[16] = {
      0.0f, 1.0f, 2.0f, 3.0f, -1.0f, 4.0f, 5.0f, 6.0f,
      10.0f, 1.0f, 6.0f, 7.0f, 3.0f, 8.0f, 5.0f, 2.0f,
  };
  nbase_sq8_t *sq8 = NULL;
  ASSERT_STATUS(nbase_sq8_train(vecs, 2U, dim, &sq8), NBASE_OK);
  ASSERT_STATUS(nbase_sq8_save(sq8, path), NBASE_OK);

  uint8_t c1[8];
  ASSERT_STATUS(nbase_sq8_encode(sq8, vecs, 1U, c1), NBASE_OK);
  nbase_sq8_destroy(sq8);

  nbase_sq8_t *loaded = NULL;
  ASSERT_STATUS(nbase_sq8_load(path, &loaded), NBASE_OK);
  ASSERT_EQ_U64(nbase_sq8_dim(loaded), dim);

  uint8_t c2[8];
  ASSERT_STATUS(nbase_sq8_encode(loaded, vecs, 1U, c2), NBASE_OK);
  for (uint32_t d = 0U; d < dim; d++) {
    ASSERT_EQ_U64(c1[d], c2[d]);
  }
  nbase_sq8_destroy(loaded);

  ASSERT_TRUE(test_flip_file_byte(path, 132L));
  ASSERT_STATUS(nbase_sq8_load(path, &loaded), NBASE_ERR_CORRUPT);

  test_rm_rf(dir);
}

static void test_sq8_l2_sq_asymmetric_batch(void) {
  /* Cover scalar tail, 8-dim AVX2 chunk, 16-dim AVX2 unroll, and heap q_shifted (>512). */
  static const uint32_t dims[7] = {1U, 7U, 13U, 16U, 64U, 128U, 600U};
  const uint32_t count = 40U;

  for (uint32_t di = 0U; di < 7U; di++) {
    uint32_t dim = dims[di];
    float *vecs = malloc((size_t)count * dim * sizeof(float));
    uint8_t *codes = malloc((size_t)count * dim);
    float *batch_d = malloc((size_t)count * sizeof(float));
    float *query = malloc((size_t)dim * sizeof(float));
    ASSERT_TRUE(vecs != NULL && codes != NULL && batch_d != NULL && query != NULL);

    test_fill_vectors(vecs, count, dim, 9001ULL + di);
    test_fill_vectors(query, 1U, dim, 9999ULL + di);

    nbase_sq8_t *sq8 = NULL;
    ASSERT_STATUS(nbase_sq8_train(vecs, count, dim, &sq8), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_encode(sq8, vecs, count, codes), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_l2_sq_asymmetric_batch(sq8, query, codes, 0U, batch_d), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_l2_sq_asymmetric_batch(sq8, query, codes, count, batch_d), NBASE_OK);

    for (uint32_t i = 0U; i < count; i++) {
      float single = nbase_sq8_l2_sq_asymmetric(sq8, query, codes + (size_t)i * dim);
      ASSERT_NEAR(batch_d[i], single, 1e-4, 1e-4);
    }

    ASSERT_STATUS(nbase_sq8_l2_sq_asymmetric_batch(sq8, query, NULL, 1U, batch_d), NBASE_ERR_INVALID_ARG);
    query[0] = (float)NAN;
    ASSERT_STATUS(nbase_sq8_l2_sq_asymmetric_batch(sq8, query, codes, count, batch_d), NBASE_ERR_INVALID_VECTOR);

    nbase_sq8_destroy(sq8);
    free(vecs);
    free(codes);
    free(batch_d);
    free(query);
  }
}

static void test_sq8_l2_sq_symmetric_batch(void) {
  static const uint32_t dims[7] = {1U, 7U, 13U, 16U, 64U, 128U, 600U};
  const uint32_t count = 40U;

  for (uint32_t di = 0U; di < 7U; di++) {
    uint32_t dim = dims[di];
    float *vecs = malloc((size_t)count * dim * sizeof(float));
    uint8_t *codes = malloc((size_t)count * dim);
    float *batch_d = malloc((size_t)count * sizeof(float));
    ASSERT_TRUE(vecs != NULL && codes != NULL && batch_d != NULL);

    test_fill_vectors(vecs, count, dim, 8101ULL + di);

    nbase_sq8_t *sq8 = NULL;
    ASSERT_STATUS(nbase_sq8_train(vecs, count, dim, &sq8), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_encode(sq8, vecs, count, codes), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_l2_sq_symmetric_batch(sq8, codes, codes, 0U, batch_d), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_l2_sq_symmetric_batch(sq8, codes, codes, count, batch_d), NBASE_OK);

    ASSERT_NEAR(batch_d[0], 0.0, 1e-6, 0.0);
    for (uint32_t i = 0U; i < count; i++) {
      float single = nbase_sq8_l2_sq_symmetric(sq8, codes, codes + (size_t)i * dim);
      ASSERT_NEAR(batch_d[i], single, 1e-4, 1e-4);
    }

    ASSERT_STATUS(nbase_sq8_l2_sq_symmetric_batch(NULL, codes, codes, 1U, batch_d), NBASE_ERR_INVALID_ARG);
    ASSERT_STATUS(nbase_sq8_l2_sq_symmetric_batch(sq8, NULL, codes, 1U, batch_d), NBASE_ERR_INVALID_ARG);
    ASSERT_STATUS(nbase_sq8_l2_sq_symmetric_batch(sq8, codes, NULL, 1U, batch_d), NBASE_ERR_INVALID_ARG);
    ASSERT_STATUS(nbase_sq8_l2_sq_symmetric_batch(sq8, codes, codes, 1U, NULL), NBASE_ERR_INVALID_ARG);

    nbase_sq8_destroy(sq8);
    free(vecs);
    free(codes);
    free(batch_d);
  }
}

void test_sq8_run(void) {
  RUN_TEST(test_sq8_train_encode_decode_and_distance);
  RUN_TEST(test_sq8_persistence_and_corruption);
  RUN_TEST(test_sq8_l2_sq_asymmetric_batch);
  RUN_TEST(test_sq8_l2_sq_symmetric_batch);
}
