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

void test_bq_run(void) {
  RUN_TEST(test_bq_train_encode_hamming_and_persistence);
}
