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
  /* Cover scalar tail, 8-dim AVX2 chunk, 16-dim AVX2 unroll, 4-row unroll tails, and heap q_shifted (>512). */
  static const uint32_t dims[9] = {1U, 7U, 13U, 16U, 17U, 35U, 64U, 128U, 600U};
  static const uint32_t sub_counts[8] = {0U, 1U, 2U, 3U, 4U, 5U, 7U, 13U};
  const uint32_t count = 40U;

  for (uint32_t di = 0U; di < 9U; di++) {
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
    for (uint32_t si = 0U; si < 8U; si++) {
      uint32_t sc = sub_counts[si];
      ASSERT_STATUS(nbase_sq8_l2_sq_asymmetric_batch(sq8, query, codes, sc, batch_d), NBASE_OK);
      for (uint32_t i = 0U; i < sc; i++) {
        float single = nbase_sq8_l2_sq_asymmetric(sq8, query, codes + (size_t)i * dim);
        ASSERT_NEAR(batch_d[i], single, 1e-4, 1e-4);
      }
    }
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
  static const uint32_t dims[9] = {1U, 7U, 13U, 16U, 17U, 35U, 64U, 128U, 600U};
  static const uint32_t sub_counts[8] = {0U, 1U, 2U, 3U, 4U, 5U, 7U, 13U};
  const uint32_t count = 40U;

  for (uint32_t di = 0U; di < 9U; di++) {
    uint32_t dim = dims[di];
    float *vecs = malloc((size_t)count * dim * sizeof(float));
    uint8_t *codes = malloc((size_t)count * dim);
    float *batch_d = malloc((size_t)count * sizeof(float));
    ASSERT_TRUE(vecs != NULL && codes != NULL && batch_d != NULL);

    test_fill_vectors(vecs, count, dim, 8101ULL + di);

    nbase_sq8_t *sq8 = NULL;
    ASSERT_STATUS(nbase_sq8_train(vecs, count, dim, &sq8), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_encode(sq8, vecs, count, codes), NBASE_OK);
    for (uint32_t si = 0U; si < 8U; si++) {
      uint32_t sc = sub_counts[si];
      ASSERT_STATUS(nbase_sq8_l2_sq_symmetric_batch(sq8, codes, codes, sc, batch_d), NBASE_OK);
      for (uint32_t i = 0U; i < sc; i++) {
        float single = nbase_sq8_l2_sq_symmetric(sq8, codes, codes + (size_t)i * dim);
        ASSERT_NEAR(batch_d[i], single, 1e-4, 1e-4);
      }
    }
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

static void test_sq8_avx2_decode_dims_and_accuracy(void) {
  static const uint32_t dims[6] = {1U, 7U, 8U, 17U, 35U, 64U};
  const uint32_t count = 64U;

  for (size_t di = 0U; di < 6U; di++) {
    uint32_t dim = dims[di];
    float *vecs = malloc((size_t)count * (size_t)dim * sizeof(float));
    uint8_t *codes = malloc((size_t)count * (size_t)dim);
    float *decoded = malloc((size_t)count * (size_t)dim * sizeof(float));
    ASSERT_TRUE(vecs != NULL && codes != NULL && decoded != NULL);
    test_fill_vectors(vecs, count, dim, 5500ULL + (uint64_t)dim);

    nbase_sq8_t *sq8 = NULL;
    ASSERT_STATUS(nbase_sq8_train(vecs, count, dim, &sq8), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_encode(sq8, vecs, count, codes), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_decode(sq8, codes, count, decoded), NBASE_OK);

    for (size_t idx = 0U; idx < (size_t)count * (size_t)dim; idx++) {
      ASSERT_NEAR(decoded[idx], vecs[idx], 0.015, 0.015);
    }

    nbase_sq8_destroy(sq8);
    free(decoded);
    free(codes);
    free(vecs);
  }
}

static void test_sq8_avx2_encode_exact_edge_cases(void) {
  static const uint32_t dims[7] = {1U, 7U, 8U, 9U, 17U, 35U, 64U};
  static const float edges[9] = {
      0.0f,
      0.49999997f,
      0.5f,
      1.5f,
      254.49998f,
      254.5f,
      -3.25f,
      300.0f,
      42.25f,
  };
  const uint32_t batch_cnt = 10U;

  for (size_t di = 0U; di < 7U; di++) {
    uint32_t dim = dims[di];
    float train[2U * 64U];
    for (uint32_t d = 0U; d < dim; d++) {
      if (dim >= 8U && d == 5U) {
        /* Constant dim -> inv_scale == 0.0f */
        train[d] = 42.0f;
        train[dim + d] = 42.0f;
      } else {
        /* Exact span 255 -> scale=1, inv_scale=1 */
        train[d] = 0.0f;
        train[dim + d] = 255.0f;
      }
    }

    nbase_sq8_t *sq8 = NULL;
    ASSERT_STATUS(nbase_sq8_train(train, 2U, dim, &sq8), NBASE_OK);

    float *vecs = malloc((size_t)batch_cnt * (size_t)dim * sizeof(float));
    uint8_t *codes = malloc((size_t)batch_cnt * (size_t)dim);
    ASSERT_TRUE(vecs != NULL && codes != NULL);

    for (uint32_t i = 0U; i < batch_cnt; i++) {
      for (uint32_t d = 0U; d < dim; d++) {
        vecs[(size_t)i * dim + d] = edges[(i * 3U + d) % 9U];
      }
    }

    ASSERT_STATUS(nbase_sq8_encode(sq8, vecs, batch_cnt, codes), NBASE_OK);
    for (uint32_t i = 0U; i < batch_cnt; i++) {
      for (uint32_t d = 0U; d < dim; d++) {
        uint8_t expected = 0U;
        if (!(dim >= 8U && d == 5U)) {
          float q = roundf(vecs[(size_t)i * dim + d]);
          if (q <= 0.0f) {
            expected = 0U;
          } else if (q >= 255.0f) {
            expected = 255U;
          } else {
            expected = (uint8_t)q;
          }
        }
        ASSERT_EQ_U64(codes[(size_t)i * dim + d], expected);
      }
    }

    nbase_sq8_destroy(sq8);
    free(codes);
    free(vecs);
  }
}

static void test_sq8_avx2_single_and_sym_batch_accuracy(void) {
  static const uint32_t dims[12] = {1U, 7U, 8U, 15U, 16U, 17U, 31U, 32U, 33U, 64U, 128U, 131U};
  static const uint32_t counts[6] = {1U, 2U, 3U, 4U, 5U, 17U};
  const uint32_t max_cnt = 17U;

  for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
    uint32_t dim = dims[di];
    float *vecs = malloc((size_t)max_cnt * (size_t)dim * sizeof(float));
    float *query = malloc((size_t)dim * sizeof(float));
    float *decoded = malloc((size_t)max_cnt * (size_t)dim * sizeof(float));
    uint8_t *codes = malloc((size_t)max_cnt * (size_t)dim);
    uint8_t *qcode = malloc((size_t)dim);
    float *asym_out = malloc((size_t)max_cnt * sizeof(float));
    float *sym_out = malloc((size_t)max_cnt * sizeof(float));
    ASSERT_TRUE(
        vecs != NULL && query != NULL && decoded != NULL && codes != NULL && qcode != NULL &&
        asym_out != NULL && sym_out != NULL);

    test_fill_vectors(vecs, max_cnt, dim, 7701ULL + (uint64_t)di);
    test_fill_vectors(query, 1U, dim, 7901ULL + (uint64_t)di);
    for (size_t i = 0; i < (size_t)max_cnt * (size_t)dim; i++) {
      vecs[i] *= 50.0f;
    }
    for (uint32_t d = 0U; d < dim; d++) {
      query[d] *= 50.0f;
    }

    nbase_sq8_t *sq8 = NULL;
    ASSERT_STATUS(nbase_sq8_train(vecs, max_cnt, dim, &sq8), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_encode(sq8, vecs, max_cnt, codes), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_encode(sq8, query, 1U, qcode), NBASE_OK);
    ASSERT_STATUS(nbase_sq8_decode(sq8, codes, max_cnt, decoded), NBASE_OK);

    for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ci++) {
      uint32_t cnt = counts[ci];
      ASSERT_STATUS(nbase_sq8_l2_sq_asymmetric_batch(sq8, query, codes, cnt, asym_out), NBASE_OK);
      ASSERT_STATUS(nbase_sq8_l2_sq_symmetric_batch(sq8, codes, codes, cnt, sym_out), NBASE_OK);
      ASSERT_NEAR(sym_out[0], 0.0, 1e-6, 0.0);

      for (uint32_t i = 0U; i < cnt; i++) {
        float asym_single = nbase_sq8_l2_sq_asymmetric(sq8, query, codes + (size_t)i * dim);
        float sym_single = nbase_sq8_l2_sq_symmetric(sq8, codes, codes + (size_t)i * dim);
        double ref_asym = 0.0;
        double ref_sym = 0.0;
        for (uint32_t d = 0U; d < dim; d++) {
          double da = (double)query[d] - (double)decoded[(size_t)i * dim + d];
          double ds = (double)decoded[d] - (double)decoded[(size_t)i * dim + d];
          ref_asym += da * da;
          ref_sym += ds * ds;
        }
        ASSERT_NEAR(asym_single, asym_out[i], 1e-3, 1e-4);
        ASSERT_NEAR(asym_single, ref_asym, 1e-3, 1e-4);
        ASSERT_NEAR(sym_single, sym_out[i], 1e-3, 1e-4);
        ASSERT_NEAR(sym_single, ref_sym, 1e-3, 1e-4);
      }
    }

    nbase_sq8_destroy(sq8);
    free(sym_out);
    free(asym_out);
    free(qcode);
    free(codes);
    free(decoded);
    free(query);
    free(vecs);
  }
}

void test_sq8_run(void) {
  RUN_TEST(test_sq8_train_encode_decode_and_distance);
  RUN_TEST(test_sq8_persistence_and_corruption);
  RUN_TEST(test_sq8_l2_sq_asymmetric_batch);
  RUN_TEST(test_sq8_l2_sq_symmetric_batch);
  RUN_TEST(test_sq8_avx2_decode_dims_and_accuracy);
  RUN_TEST(test_sq8_avx2_encode_exact_edge_cases);
  RUN_TEST(test_sq8_avx2_single_and_sym_batch_accuracy);
}
