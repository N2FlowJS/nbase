#include "test_common.h"
#include "nbase_internal.h"

#include <string.h>

static void test_all_metrics_across_dims(void) {
  static const uint32_t dims[] = {1U, 3U, 7U, 8U, 9U, 15U, 16U, 17U, 64U, 128U, 1536U};
  static const nbase_metric_t metrics[] = {
      NBASE_METRIC_EUCLIDEAN,
      NBASE_METRIC_SQUARED_EUCLIDEAN,
      NBASE_METRIC_COSINE,
      NBASE_METRIC_INNER_PRODUCT,
      NBASE_METRIC_MANHATTAN,
      NBASE_METRIC_CHEBYSHEV,
      NBASE_METRIC_HAMMING,
  };

  float *a = malloc(1536U * sizeof(float));
  float *b = malloc(1536U * sizeof(float));
  ASSERT_TRUE(a != NULL && b != NULL);

  test_fill_vectors(a, 1U, 1536U, 101ULL);
  test_fill_vectors(b, 1U, 1536U, 202ULL);
  /* Set some exact equal lanes so Hamming tests both matches and mismatches. */
  for (uint32_t i = 0U; i < 1536U; i += 3U) {
    b[i] = a[i];
  }

  for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
    uint32_t dim = dims[di];
    for (size_t mi = 0; mi < sizeof(metrics) / sizeof(metrics[0]); mi++) {
      nbase_metric_t metric = metrics[mi];
      float simd_val = nbase_distance(metric, a, b, dim);
      float scalar_val = nbase_distance_scalar(metric, a, b, dim);
      double ref_val = test_ref_distance(metric, a, b, dim);
      ASSERT_NEAR(simd_val, scalar_val, 1e-4, 1e-4);
      ASSERT_NEAR(simd_val, ref_val, 1e-4, 1e-4);
    }
  }

  free(a);
  free(b);
}

static void test_distance_edge_cases(void) {
  float zero[16];
  float unit[16];
  float neg_unit[16];
  memset(zero, 0, sizeof(zero));
  for (uint32_t i = 0U; i < 16U; i++) {
    unit[i] = (i == 0U) ? 1.0f : 0.0f;
    neg_unit[i] = (i == 0U) ? -1.0f : 0.0f;
  }

  ASSERT_NEAR(nbase_distance(NBASE_METRIC_COSINE, zero, unit, 16U), 1.0, 1e-6, 0.0);
  ASSERT_NEAR(nbase_distance(NBASE_METRIC_COSINE, zero, zero, 16U), 1.0, 1e-6, 0.0);
  ASSERT_NEAR(nbase_distance(NBASE_METRIC_COSINE, unit, unit, 16U), 0.0, 1e-6, 0.0);
  ASSERT_NEAR(nbase_distance(NBASE_METRIC_COSINE, unit, neg_unit, 16U), 2.0, 1e-6, 0.0);

  ASSERT_NEAR(nbase_distance(NBASE_METRIC_EUCLIDEAN, unit, unit, 16U), 0.0, 1e-6, 0.0);
  ASSERT_NEAR(nbase_distance(NBASE_METRIC_SQUARED_EUCLIDEAN, unit, unit, 16U), 0.0, 1e-6, 0.0);
  ASSERT_NEAR(nbase_distance(NBASE_METRIC_MANHATTAN, unit, unit, 16U), 0.0, 1e-6, 0.0);
  ASSERT_NEAR(nbase_distance(NBASE_METRIC_CHEBYSHEV, unit, unit, 16U), 0.0, 1e-6, 0.0);
  ASSERT_NEAR(nbase_distance(NBASE_METRIC_HAMMING, unit, unit, 16U), 0.0, 1e-6, 0.0);
  ASSERT_NEAR(nbase_distance(NBASE_METRIC_INNER_PRODUCT, unit, neg_unit, 16U), 1.0, 1e-6, 0.0);
}

static void test_distance_batch_all_metrics_and_dims(void) {
  static const uint32_t dims[] = {1U, 7U, 8U, 15U, 16U, 64U, 128U};
  static const nbase_metric_t metrics[] = {
      NBASE_METRIC_EUCLIDEAN,
      NBASE_METRIC_SQUARED_EUCLIDEAN,
      NBASE_METRIC_COSINE,
      NBASE_METRIC_INNER_PRODUCT,
      NBASE_METRIC_MANHATTAN,
      NBASE_METRIC_CHEBYSHEV,
      NBASE_METRIC_HAMMING,
  };
  const uint32_t count = 37U;
  float *query = malloc(128U * sizeof(float));
  float *vecs = malloc((size_t)count * 128U * sizeof(float));
  float *dists = malloc((size_t)count * sizeof(float));
  float *zero_q = calloc(128U, sizeof(float));
  ASSERT_TRUE(query != NULL && vecs != NULL && dists != NULL && zero_q != NULL);

  test_fill_vectors(query, 1U, 128U, 501ULL);
  test_fill_vectors(vecs, count, 128U, 602ULL);
  for (uint32_t n = 0U; n < count; n++) {
    for (uint32_t i = 0U; i < 128U; i += 4U) {
      vecs[(size_t)n * 128U + i] = query[i];
    }
  }

  for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
    uint32_t dim = dims[di];
    for (size_t mi = 0; mi < sizeof(metrics) / sizeof(metrics[0]); mi++) {
      nbase_metric_t metric = metrics[mi];
      ASSERT_STATUS(nbase_distance_batch(metric, query, vecs, count, dim, dists), NBASE_OK);
      for (uint32_t n = 0U; n < count; n++) {
        float expected = nbase_distance_scalar(metric, query, vecs + (size_t)n * (size_t)dim, dim);
        ASSERT_NEAR(dists[n], expected, 1e-4, 1e-4);
      }
    }
    /* Zero query vector edge case for Cosine distance */
    ASSERT_STATUS(nbase_distance_batch(NBASE_METRIC_COSINE, zero_q, vecs, count, dim, dists), NBASE_OK);
    for (uint32_t n = 0U; n < count; n++) {
      ASSERT_NEAR(dists[n], 1.0, 1e-6, 0.0);
    }
  }

  /* Validation & count == 0 edge cases */
  ASSERT_STATUS(nbase_distance_batch(NBASE_METRIC_EUCLIDEAN, NULL, NULL, 0U, 8U, NULL), NBASE_OK);
  ASSERT_STATUS(nbase_distance_batch(NBASE_METRIC_EUCLIDEAN, query, vecs, 4U, 0U, dists), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_distance_batch((nbase_metric_t)99, query, vecs, 4U, 8U, dists), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_distance_batch(NBASE_METRIC_EUCLIDEAN, NULL, vecs, 4U, 8U, dists), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_distance_batch(NBASE_METRIC_EUCLIDEAN, query, NULL, 4U, 8U, dists), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_distance_batch(NBASE_METRIC_EUCLIDEAN, query, vecs, 4U, 8U, NULL), NBASE_ERR_INVALID_ARG);

  free(zero_q);
  free(dists);
  free(vecs);
  free(query);
}

static void test_distance_batch_unrolled_edges(void) {
  static const uint32_t dims[] = {1U, 3U, 7U, 8U, 9U, 13U, 15U, 16U, 17U, 31U, 32U, 33U, 64U, 128U, 129U};
  static const uint32_t counts[] = {0U, 1U, 2U, 3U, 4U, 5U, 17U};
  static const nbase_metric_t metrics[] = {
      NBASE_METRIC_EUCLIDEAN,
      NBASE_METRIC_SQUARED_EUCLIDEAN,
      NBASE_METRIC_COSINE,
      NBASE_METRIC_INNER_PRODUCT,
      NBASE_METRIC_MANHATTAN,
      NBASE_METRIC_CHEBYSHEV,
      NBASE_METRIC_HAMMING,
  };

  float *query = malloc(129U * sizeof(float));
  float *vecs = malloc(17U * 129U * sizeof(float));
  float *dists = malloc(17U * sizeof(float));
  ASSERT_TRUE(query != NULL && vecs != NULL && dists != NULL);
  test_fill_vectors(query, 1U, 129U, 711ULL);
  test_fill_vectors(vecs, 17U, 129U, 722ULL);
  /* Match every 3rd lane so Hamming covers mixed equal and distinct floats across dims. */
  for (uint32_t n = 0U; n < 17U; n++) {
    for (uint32_t i = 0U; i < 129U; i += 3U) {
      vecs[(size_t)n * 129U + i] = query[i];
    }
  }
  /* Include a zero-norm candidate in slot 2 to verify Cosine denom == 0 edge case. */
  memset(vecs + 2U * 129U, 0, 129U * sizeof(float));

  for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
    uint32_t dim = dims[di];
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ci++) {
      uint32_t count = counts[ci];
      for (size_t mi = 0; mi < sizeof(metrics) / sizeof(metrics[0]); mi++) {
        nbase_metric_t metric = metrics[mi];
        ASSERT_STATUS(nbase_distance_batch(metric, query, vecs, count, dim, dists), NBASE_OK);
        for (uint32_t n = 0U; n < count; n++) {
          float expected = nbase_distance_scalar(metric, query, vecs + (size_t)n * (size_t)dim, dim);
          ASSERT_NEAR(dists[n], expected, 1e-4, 1e-4);
        }
      }
    }
  }

  free(dists);
  free(vecs);
  free(query);
}

static void test_hamming_avx2_and_signed_zero_edges(void) {
  static const uint32_t dims[] = {1U, 7U, 8U, 15U, 16U, 17U, 31U, 32U, 33U, 64U, 128U, 131U};
  static const uint32_t counts[] = {1U, 2U, 3U, 4U, 5U, 17U};
  const uint32_t max_dim = 131U;
  const uint32_t max_cnt = 17U;

  float *q = malloc((size_t)max_dim * sizeof(float));
  float *vecs = malloc((size_t)max_cnt * (size_t)max_dim * sizeof(float));
  float *dists = malloc((size_t)max_cnt * sizeof(float));
  ASSERT_TRUE(q != NULL && vecs != NULL && dists != NULL);

  test_fill_vectors(q, 1U, max_dim, 9101ULL);
  test_fill_vectors(vecs, max_cnt, max_dim, 9202ULL);
  for (uint32_t d = 0U; d < max_dim; d++) {
    if ((d % 3U) == 0U) {
      q[d] = 0.0f;
    }
  }
  for (uint32_t n = 0U; n < max_cnt; n++) {
    for (uint32_t d = 0U; d < max_dim; d++) {
      if ((d % 3U) == 0U) {
        /* IEEE +0.0f == -0.0f must count as equal in Hamming. */
        vecs[(size_t)n * max_dim + d] = (n % 2U == 0U) ? -0.0f : 0.0f;
      } else if (((n + d) % 5U) == 0U) {
        vecs[(size_t)n * max_dim + d] = q[d];
      }
    }
  }

  for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
    uint32_t dim = dims[di];
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ci++) {
      uint32_t cnt = counts[ci];
      ASSERT_STATUS(nbase_distance_batch(NBASE_METRIC_HAMMING, q, vecs, cnt, dim, dists), NBASE_OK);
      for (uint32_t n = 0U; n < cnt; n++) {
        const float *row = vecs + (size_t)n * (size_t)dim;
        float scalar = nbase_distance_scalar(NBASE_METRIC_HAMMING, q, row, dim);
        float single = nbase_distance(NBASE_METRIC_HAMMING, q, row, dim);
        ASSERT_NEAR(single, scalar, 1e-6, 0.0);
        ASSERT_NEAR(dists[n], scalar, 1e-6, 0.0);
      }
    }
  }

  free(dists);
  free(vecs);
  free(q);
}

static void test_distance_indexed_batch_all_metrics_and_dims(void) {
  static const uint32_t dims[] = {1U, 7U, 8U, 15U, 16U, 33U, 64U, 128U};
  static const uint32_t counts[] = {0U, 1U, 3U, 4U, 5U, 17U, 64U};
  static const nbase_metric_t metrics[] = {
      NBASE_METRIC_EUCLIDEAN,
      NBASE_METRIC_SQUARED_EUCLIDEAN,
      NBASE_METRIC_COSINE,
      NBASE_METRIC_INNER_PRODUCT,
      NBASE_METRIC_MANHATTAN,
      NBASE_METRIC_CHEBYSHEV,
      NBASE_METRIC_HAMMING,
  };
  const uint32_t pool_n = 80U;
  const uint32_t max_dim = 128U;
  const uint32_t max_cnt = 64U;

  float *q = malloc((size_t)max_dim * sizeof(float));
  float *zero_q = calloc((size_t)max_dim, sizeof(float));
  float *pool = malloc((size_t)pool_n * (size_t)max_dim * sizeof(float));
  float *dists = malloc((size_t)max_cnt * sizeof(float));
  uint32_t *slots = malloc((size_t)max_cnt * sizeof(uint32_t));
  ASSERT_TRUE(q && zero_q && pool && dists && slots);

  test_fill_vectors(q, 1U, max_dim, 7101ULL);
  test_fill_vectors(pool, pool_n, max_dim, 7202ULL);
  for (uint32_t n = 0U; n < pool_n; n++) {
    for (uint32_t d = 0U; d < max_dim; d++) {
      if (((n + d) % 4U) == 0U) {
        pool[(size_t)n * max_dim + d] = q[d];
      }
    }
  }
  for (uint32_t i = 0U; i < max_cnt; i++) {
    /* Mix strided, reversed, and duplicate slot indices across pool. */
    slots[i] = (i % 7U == 0U) ? 13U : ((pool_n - 1U) - ((i * 5U) % pool_n));
  }

  for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
    uint32_t dim = dims[di];
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ci++) {
      uint32_t cnt = counts[ci];
      for (size_t mi = 0; mi < sizeof(metrics) / sizeof(metrics[0]); mi++) {
        nbase_metric_t metric = metrics[mi];
        nbase_distance_indexed_batch(metric, q, pool, slots, cnt, dim, dists);
        for (uint32_t i = 0U; i < cnt; i++) {
          double ref = test_ref_distance(metric, q, pool + (size_t)slots[i] * (size_t)dim, dim);
          ASSERT_NEAR((double)dists[i], ref, 1e-5, 1e-5);
        }
      }
      /* Zero-norm cosine query edge must return 1.0f for every indexed slot. */
      nbase_distance_indexed_batch(NBASE_METRIC_COSINE, zero_q, pool, slots, cnt, dim, dists);
      for (uint32_t i = 0U; i < cnt; i++) {
        ASSERT_NEAR((double)dists[i], 1.0, 1e-6, 0.0);
      }
    }
  }

  free(slots);
  free(dists);
  free(pool);
  free(zero_q);
  free(q);
}

void test_distance_run(void) {
  RUN_TEST(test_all_metrics_across_dims);
  RUN_TEST(test_distance_edge_cases);
  RUN_TEST(test_distance_batch_all_metrics_and_dims);
  RUN_TEST(test_distance_batch_unrolled_edges);
  RUN_TEST(test_hamming_avx2_and_signed_zero_edges);
  RUN_TEST(test_distance_indexed_batch_all_metrics_and_dims);
}
