#include "test_common.h"

#include <stdio.h>

static void test_bq_train_encode_hamming_and_persistence(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  char path[300];
  snprintf(path, sizeof(path), "%s/quant.nbbq", dir);

  const uint32_t dim = 130U; /* Exercises > 2 uint64_t words (3 words = 192 bits). */
  const uint32_t n = 64U;
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  ASSERT_TRUE(vecs != NULL);
  test_fill_vectors(vecs, n, dim, 6100ULL);
  for (size_t i = 0U; i < (size_t)n * dim; i++) {
    vecs[i] = vecs[i] * 2.0f - 1.0f;
  }

  nbase_bq_t *bq = NULL;
  ASSERT_STATUS(nbase_bq_train(vecs, n, dim, &bq), NBASE_OK);
  ASSERT_EQ_U64(nbase_bq_dim(bq), dim);
  ASSERT_EQ_U64(nbase_bq_words_per_vec(bq), 3ULL);

  uint64_t *words = malloc((size_t)n * 3U * sizeof(uint64_t));
  ASSERT_TRUE(words != NULL);
  ASSERT_STATUS(nbase_bq_encode(bq, vecs, n, words), NBASE_OK);

  /* Identical vector has Hamming distance 0; negated vector differs on most dims. */
  ASSERT_EQ_U64(nbase_bq_hamming(words, words, 3U), 0ULL);

  float neg_vec[130];
  for (uint32_t d = 0U; d < dim; d++) {
    neg_vec[d] = -vecs[d];
  }
  uint64_t neg_words[3];
  ASSERT_STATUS(nbase_bq_encode(bq, neg_vec, 1U, neg_words), NBASE_OK);
  uint32_t d_neg = nbase_bq_hamming(words, neg_words, 3U);
  ASSERT_TRUE(d_neg > 60U && d_neg <= dim);

  ASSERT_STATUS(nbase_bq_save(bq, path), NBASE_OK);
  nbase_bq_destroy(bq);

  nbase_bq_t *loaded = NULL;
  ASSERT_STATUS(nbase_bq_load(path, &loaded), NBASE_OK);
  ASSERT_EQ_U64(nbase_bq_dim(loaded), dim);
  uint64_t reload_w[3];
  ASSERT_STATUS(nbase_bq_encode(loaded, vecs, 1U, reload_w), NBASE_OK);
  ASSERT_EQ_U64(nbase_bq_hamming(words, reload_w, 3U), 0ULL);
  nbase_bq_destroy(loaded);

  ASSERT_TRUE(test_flip_file_byte(path, 132L));
  ASSERT_STATUS(nbase_bq_load(path, &loaded), NBASE_ERR_CORRUPT);

  free(words);
  free(vecs);
  test_rm_rf(dir);
}

static void test_bq_hamming_batch(void) {
  static const uint32_t dims[4] = {64U, 128U, 256U, 300U};
  const uint32_t n = 48U;

  for (uint32_t di = 0U; di < 4U; di++) {
    uint32_t dim = dims[di];
    float *vecs = malloc((size_t)n * dim * sizeof(float));
    float *query = malloc((size_t)dim * sizeof(float));
    ASSERT_TRUE(vecs != NULL && query != NULL);
    test_fill_vectors(vecs, n, dim, 6500ULL + di);
    test_fill_vectors(query, 1U, dim, 6800ULL + di);

    nbase_bq_t *bq = NULL;
    ASSERT_STATUS(nbase_bq_train(vecs, n, dim, &bq), NBASE_OK);
    uint32_t wpv = nbase_bq_words_per_vec(bq);
    uint64_t *codes = malloc((size_t)n * wpv * sizeof(uint64_t));
    uint64_t *qwords = malloc((size_t)wpv * sizeof(uint64_t));
    uint32_t *out = malloc((size_t)n * sizeof(uint32_t));
    ASSERT_TRUE(codes != NULL && qwords != NULL && out != NULL);

    ASSERT_STATUS(nbase_bq_encode(bq, vecs, n, codes), NBASE_OK);
    ASSERT_STATUS(nbase_bq_encode(bq, query, 1U, qwords), NBASE_OK);
    ASSERT_STATUS(nbase_bq_hamming_batch(qwords, codes, 0U, wpv, out), NBASE_OK);
    ASSERT_STATUS(nbase_bq_hamming_batch(qwords, codes, n, wpv, out), NBASE_OK);

    for (uint32_t i = 0U; i < n; i++) {
      uint32_t exact = nbase_bq_hamming(qwords, codes + (size_t)i * wpv, wpv);
      ASSERT_EQ_U64(out[i], exact);
    }

    ASSERT_STATUS(nbase_bq_hamming_batch(NULL, NULL, n, 0U, out), NBASE_OK);
    ASSERT_EQ_U64(out[0], 0ULL);
    ASSERT_STATUS(nbase_bq_hamming_batch(NULL, codes, n, wpv, out), NBASE_ERR_INVALID_ARG);

    free(codes);
    free(qwords);
    free(out);
    nbase_bq_destroy(bq);
    free(vecs);
    free(query);
  }
}

static void test_bq_avx2_encode_and_popcnt_batch_edges(void) {
  static const uint32_t dims[10] = {1U, 7U, 8U, 15U, 16U, 63U, 64U, 65U, 130U, 256U};
  static const uint32_t sub_counts[8] = {0U, 1U, 2U, 3U, 4U, 5U, 7U, 13U};
  const uint32_t n = 20U;

  for (uint32_t di = 0U; di < 10U; di++) {
    uint32_t dim = dims[di];
    float *vecs = malloc((size_t)n * dim * sizeof(float));
    ASSERT_TRUE(vecs != NULL);
    test_fill_vectors(vecs, n, dim, 7100ULL + di);

    nbase_bq_t *bq = NULL;
    ASSERT_STATUS(nbase_bq_train(vecs, n, dim, &bq), NBASE_OK);
    uint32_t wpv = nbase_bq_words_per_vec(bq);
    uint64_t *codes = malloc((size_t)n * wpv * sizeof(uint64_t));
    uint32_t *out = malloc((size_t)n * sizeof(uint32_t));
    ASSERT_TRUE(codes != NULL && out != NULL);

    ASSERT_STATUS(nbase_bq_encode(bq, vecs, n, codes), NBASE_OK);
    for (uint32_t i = 0U; i < n; i++) {
      for (uint32_t d = 0U; d < dim; d++) {
        uint64_t single_word = 0ULL;
        uint32_t bit = (codes[(size_t)i * wpv + (d >> 6U)] >> (d & 63U)) & 1ULL;
        (void)single_word;
        double sum = 0.0;
        for (uint32_t k = 0U; k < n; k++) {
          sum += (double)vecs[(size_t)k * dim + d];
        }
        float thresh = (float)(sum / (double)n);
        uint32_t expected_bit = (vecs[(size_t)i * dim + d] >= thresh) ? 1U : 0U;
        ASSERT_EQ_U64((uint64_t)bit, (uint64_t)expected_bit);
      }
    }

    for (uint32_t si = 0U; si < 8U; si++) {
      uint32_t sc = sub_counts[si];
      ASSERT_STATUS(nbase_bq_hamming_batch(codes, codes, sc, wpv, out), NBASE_OK);
      for (uint32_t i = 0U; i < sc; i++) {
        uint32_t exact = nbase_bq_hamming(codes, codes + (size_t)i * wpv, wpv);
        ASSERT_EQ_U64(out[i], exact);
      }
    }

    free(out);
    free(codes);
    nbase_bq_destroy(bq);
    free(vecs);
  }
}

void test_bq_run(void) {
  RUN_TEST(test_bq_train_encode_hamming_and_persistence);
  RUN_TEST(test_bq_hamming_batch);
  RUN_TEST(test_bq_avx2_encode_and_popcnt_batch_edges);
}
