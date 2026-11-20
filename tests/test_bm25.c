#include "test_common.h"

#include <stdio.h>
#include <string.h>

static bool filter_even_ids(uint64_t id, void *ctx) {
  (void)ctx;
  return (id % 2ULL) == 0ULL;
}

static void test_bm25_index_search_delete_and_persistence(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  char path[300];
  snprintf(path, sizeof(path), "%s/text.nbbm", dir);

  nbase_bm25_t *bm25 = NULL;
  ASSERT_STATUS(nbase_bm25_create(1.5f, 0.75f, &bm25), NBASE_OK);
  ASSERT_EQ_U64(nbase_bm25_doc_count(bm25), 0ULL);

  ASSERT_STATUS(nbase_bm25_index(bm25, 1ULL, "Pure C11 vector database with AVX2 SIMD acceleration"), NBASE_OK);
  ASSERT_STATUS(nbase_bm25_index(bm25, 2ULL, "Hybrid BM25 full text search with vector embeddings"), NBASE_OK);
  ASSERT_STATUS(nbase_bm25_index(bm25, 3ULL, "Vector database search engine supporting HNSW and IVF PQ vector"), NBASE_OK);
  /* Upsert id 1 with stronger match for vector search */
  ASSERT_STATUS(nbase_bm25_index(bm25, 1ULL, "Pure C11 vector database with AVX2 SIMD HNSW vector search"), NBASE_OK);
  ASSERT_EQ_U64(nbase_bm25_doc_count(bm25), 3ULL);

  size_t got_len = 0U;
  char doc_buf[128];
  ASSERT_STATUS(nbase_bm25_get(bm25, 2ULL, NULL, 0U, &got_len), NBASE_OK);
  ASSERT_EQ_U64(got_len, strlen("Hybrid BM25 full text search with vector embeddings"));
  ASSERT_STATUS(nbase_bm25_get(bm25, 2ULL, doc_buf, sizeof(doc_buf), NULL), NBASE_OK);
  ASSERT_TRUE(strcmp(doc_buf, "Hybrid BM25 full text search with vector embeddings") == 0);
  ASSERT_STATUS(nbase_bm25_get(bm25, 99ULL, doc_buf, sizeof(doc_buf), NULL), NBASE_ERR_NOT_FOUND);

  nbase_scored_id_t hits[5];
  uint32_t cnt = 0U;
  ASSERT_STATUS(nbase_bm25_search(bm25, "avx2 hnsw", 5U, hits, &cnt), NBASE_OK);
  ASSERT_TRUE(cnt >= 1U);
  ASSERT_EQ_U64(hits[0].id, 1ULL);

  ASSERT_STATUS(nbase_bm25_search_filtered(bm25, "vector", 5U, filter_even_ids, NULL, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 1ULL);
  ASSERT_EQ_U64(hits[0].id, 2ULL);

  ASSERT_STATUS(nbase_bm25_save(bm25, path), NBASE_OK);
  ASSERT_STATUS(nbase_bm25_delete(bm25, 1ULL), NBASE_OK);
  ASSERT_STATUS(nbase_bm25_delete(bm25, 1ULL), NBASE_ERR_NOT_FOUND);
  ASSERT_EQ_U64(nbase_bm25_doc_count(bm25), 2ULL);

  cnt = 0U;
  ASSERT_STATUS(nbase_bm25_search(bm25, "avx2", 5U, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 0ULL);
  nbase_bm25_destroy(bm25);

  nbase_bm25_t *loaded = NULL;
  ASSERT_STATUS(nbase_bm25_load(path, &loaded), NBASE_OK);
  ASSERT_EQ_U64(nbase_bm25_doc_count(loaded), 3ULL);
  ASSERT_STATUS(nbase_bm25_search(loaded, "avx2 hnsw", 5U, hits, &cnt), NBASE_OK);
  ASSERT_TRUE(cnt >= 1U && hits[0].id == 1ULL);
  nbase_bm25_destroy(loaded);

  ASSERT_TRUE(test_flip_file_byte(path, 140L));
  ASSERT_STATUS(nbase_bm25_load(path, &loaded), NBASE_ERR_CORRUPT);

  test_rm_rf(dir);
}

static void test_hybrid_rrf_fusion(void) {
  nbase_result_t vec_res[3] = {{10ULL, 0.1f}, {20ULL, 0.2f}, {30ULL, 0.3f}};
  nbase_scored_id_t txt_res[3] = {{30ULL, 9.0f}, {20ULL, 5.0f}, {40ULL, 2.0f}};
  nbase_scored_id_t fused[4];
  uint32_t cnt = 0U;

  /* alpha = 1.0 -> strictly vector order (10, 20, 30) */
  ASSERT_STATUS(nbase_hybrid_rrf(vec_res, 3U, txt_res, 3U, 1.0f, 60.0f, 4U, fused, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 4ULL);
  ASSERT_EQ_U64(fused[0].id, 10ULL);
  ASSERT_EQ_U64(fused[1].id, 20ULL);
  ASSERT_EQ_U64(fused[2].id, 30ULL);

  /* alpha = 0.0 -> strictly text order (30, 20, 40) */
  ASSERT_STATUS(nbase_hybrid_rrf(vec_res, 3U, txt_res, 3U, 0.0f, 60.0f, 4U, fused, &cnt), NBASE_OK);
  ASSERT_EQ_U64(fused[0].id, 30ULL);
  ASSERT_EQ_U64(fused[1].id, 20ULL);
  ASSERT_EQ_U64(fused[2].id, 40ULL);

  /* Balanced alpha = 0.5 -> id 20 & 30 appear in both lists and outscore single-list ids 10 & 40. */
  ASSERT_STATUS(nbase_hybrid_rrf(vec_res, 3U, txt_res, 3U, 0.5f, 60.0f, 4U, fused, &cnt), NBASE_OK);
  ASSERT_TRUE(fused[0].id == 20ULL || fused[0].id == 30ULL);
  ASSERT_TRUE(fused[1].id == 20ULL || fused[1].id == 30ULL);

  ASSERT_STATUS(nbase_hybrid_rrf(vec_res, 3U, txt_res, 3U, 1.5f, 60.0f, 4U, fused, &cnt), NBASE_ERR_INVALID_ARG);
}

void test_bm25_run(void) {
  RUN_TEST(test_bm25_index_search_delete_and_persistence);
  RUN_TEST(test_hybrid_rrf_fusion);
}
