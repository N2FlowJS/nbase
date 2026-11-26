#include "test_common.h"

#include <pthread.h>
#include <stdatomic.h>
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

typedef struct bm25_tsan_ctx {
  nbase_bm25_t *bm25;
  atomic_bool stop;
  atomic_int errors;
} bm25_tsan_ctx_t;

static void *bm25_reader_fn(void *raw) {
  bm25_tsan_ctx_t *ctx = (bm25_tsan_ctx_t *)raw;
  static const char *qs[3] = {"vector simd", "bm25 hybrid", "compact engine"};
  nbase_scored_id_t batch_res[3U * 5U];
  uint32_t batch_cnts[3];
  uint64_t ids[16];
  for (uint32_t i = 0U; i < 30U && !atomic_load_explicit(&ctx->stop, memory_order_relaxed); i++) {
    nbase_scored_id_t hits[5];
    uint32_t got = 0U;
    if (nbase_bm25_search(ctx->bm25, "vector simd", 5U, hits, &got) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    if (nbase_bm25_search_batch(ctx->bm25, qs, 3U, 5U, 2U, batch_res, batch_cnts) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    uint32_t lcnt = 0U;
    uint64_t ltot = 0ULL;
    if (nbase_bm25_list_ids(ctx->bm25, 0ULL, 16U, ids, &lcnt, &ltot) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    (void)nbase_bm25_contains(ctx->bm25, 10ULL);
    (void)nbase_bm25_doc_count(ctx->bm25);
  }
  return NULL;
}

static void test_bm25_crud_batch_compact_and_tsan(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  char path[300];
  snprintf(path, sizeof(path), "%s/compact.nbbm", dir);

  nbase_bm25_t *bm25 = NULL;
  ASSERT_STATUS(nbase_bm25_create(1.5f, 0.75f, &bm25), NBASE_OK);

  static const uint64_t init_ids[5] = {50ULL, 10ULL, 40ULL, 20ULL, 30ULL};
  static const char *init_texts[5] = {
      "vector search database with avx2 simd",
      "hybrid bm25 full text search engine",
      "scalable partitioned vector storage engine",
      "ivfpq product quantization compressed search",
      "ephemeral_unique_token vector storage test",
  };
  ASSERT_STATUS(nbase_bm25_index_batch(bm25, init_ids, init_texts, 5U), NBASE_OK);
  ASSERT_EQ_U64(nbase_bm25_doc_count(bm25), 5ULL);
  ASSERT_TRUE(nbase_bm25_contains(bm25, 50ULL));
  ASSERT_TRUE(!nbase_bm25_contains(bm25, 999ULL));
  ASSERT_TRUE(!nbase_bm25_contains(NULL, 10ULL));

  /* Error checks */
  static const char *bad_texts[1] = {NULL};
  ASSERT_STATUS(nbase_bm25_index_batch(bm25, init_ids, bad_texts, 1U), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_bm25_compact(NULL), NBASE_ERR_INVALID_ARG);

  /* Upsert via index_batch + sorted list_ids pagination */
  static const uint64_t up_ids[2] = {30ULL, 60ULL};
  static const char *up_texts[2] = {
      "vector search replacement text for thirty",
      "hnsw graph index with avx2 vector distance",
  };
  ASSERT_STATUS(nbase_bm25_index_batch(bm25, up_ids, up_texts, 2U), NBASE_OK);
  ASSERT_EQ_U64(nbase_bm25_doc_count(bm25), 6ULL);

  uint64_t page[4];
  uint32_t got = 0U;
  uint64_t total = 0ULL;
  ASSERT_STATUS(nbase_bm25_list_ids(bm25, 1ULL, 3U, page, &got, &total), NBASE_OK);
  ASSERT_EQ_U64(total, 6ULL);
  ASSERT_EQ_U64(got, 3ULL);
  ASSERT_EQ_U64(page[0], 20ULL);
  ASSERT_EQ_U64(page[1], 30ULL);
  ASSERT_EQ_U64(page[2], 40ULL);

  ASSERT_STATUS(nbase_bm25_list_ids(bm25, 20ULL, 4U, page, &got, &total), NBASE_OK);
  ASSERT_EQ_U64(total, 6ULL);
  ASSERT_EQ_U64(got, 0ULL);

  /* Delete and compact reclaims tombstones & dead terms */
  ASSERT_STATUS(nbase_bm25_delete(bm25, 20ULL), NBASE_OK);
  ASSERT_TRUE(!nbase_bm25_contains(bm25, 20ULL));
  ASSERT_STATUS(nbase_bm25_compact(bm25), NBASE_OK);
  ASSERT_EQ_U64(nbase_bm25_doc_count(bm25), 5ULL);

  nbase_scored_id_t dead_hits[5];
  uint32_t dead_cnt = 0U;
  ASSERT_STATUS(nbase_bm25_search(bm25, "ephemeral_unique_token", 5U, dead_hits, &dead_cnt), NBASE_OK);
  ASSERT_EQ_U64(dead_cnt, 0ULL);

  /* Multi-threaded search_batch parity vs serial nbase_bm25_search */
  static const char *queries[6] = {
      "vector simd",
      "hybrid full text",
      "avx2 hnsw vector",
      "replacement thirty",
      "nonexistent_term_xyz",
      "engine",
  };
  nbase_scored_id_t batch_res[6U * 5U];
  uint32_t batch_cnts[6];
  ASSERT_STATUS(nbase_bm25_search_batch(bm25, queries, 0U, 5U, 4U, NULL, NULL), NBASE_OK);
  ASSERT_STATUS(nbase_bm25_search_batch(bm25, queries, 6U, 5U, 4U, batch_res, batch_cnts), NBASE_OK);
  for (uint32_t q = 0U; q < 6U; q++) {
    nbase_scored_id_t serial_res[5];
    uint32_t serial_cnt = 0U;
    ASSERT_STATUS(nbase_bm25_search(bm25, queries[q], 5U, serial_res, &serial_cnt), NBASE_OK);
    ASSERT_EQ_U64(batch_cnts[q], serial_cnt);
    for (uint32_t j = 0U; j < serial_cnt; j++) {
      ASSERT_EQ_U64(batch_res[(size_t)q * 5U + j].id, serial_res[j].id);
      ASSERT_NEAR(batch_res[(size_t)q * 5U + j].score, serial_res[j].score, 1e-6, 0.0);
    }
  }
  ASSERT_STATUS(nbase_bm25_search_batch(bm25, bad_texts, 1U, 5U, 2U, batch_res, batch_cnts), NBASE_ERR_INVALID_ARG);

  /* Persistence round trip after compact */
  ASSERT_STATUS(nbase_bm25_save(bm25, path), NBASE_OK);
  nbase_bm25_destroy(bm25);
  bm25 = NULL;
  ASSERT_STATUS(nbase_bm25_load(path, &bm25), NBASE_OK);
  ASSERT_EQ_U64(nbase_bm25_doc_count(bm25), 5ULL);
  ASSERT_TRUE(nbase_bm25_contains(bm25, 60ULL));

  /* Concurrent TSan readers + writer */
  bm25_tsan_ctx_t ctx = {bm25, false, 0};
  pthread_t r1, r2;
  ASSERT_TRUE(pthread_create(&r1, NULL, bm25_reader_fn, &ctx) == 0);
  ASSERT_TRUE(pthread_create(&r2, NULL, bm25_reader_fn, &ctx) == 0);
  for (uint32_t round = 0U; round < 15U; round++) {
    uint64_t tid = 1000ULL + (uint64_t)round;
    ASSERT_STATUS(nbase_bm25_index(bm25, tid, "concurrent vector simd bm25 compact test"), NBASE_OK);
    if ((round % 2U) == 1U) {
      ASSERT_STATUS(nbase_bm25_delete(bm25, tid), NBASE_OK);
    }
    if ((round % 5U) == 4U) {
      ASSERT_STATUS(nbase_bm25_compact(bm25), NBASE_OK);
    }
  }
  atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
  pthread_join(r1, NULL);
  pthread_join(r2, NULL);
  ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);

  nbase_bm25_destroy(bm25);
  test_rm_rf(dir);
}

static void test_bm25_delete_batch_and_errors(void) {
  nbase_bm25_t *bm25 = NULL;
  ASSERT_STATUS(nbase_bm25_create(1.5f, 0.75f, &bm25), NBASE_OK);

  for (uint64_t id = 1ULL; id <= 8ULL; id++) {
    ASSERT_STATUS(nbase_bm25_index(bm25, id, "vector similarity engine batch delete bm25"), NBASE_OK);
  }
  ASSERT_EQ_U64(nbase_bm25_doc_count(bm25), 8ULL);

  uint32_t del_cnt = 99U;
  ASSERT_STATUS(nbase_bm25_delete_batch(bm25, NULL, 0U, &del_cnt), NBASE_OK);
  ASSERT_EQ_U64(del_cnt, 0ULL);
  ASSERT_STATUS(nbase_bm25_delete_batch(NULL, (const uint64_t[]){1ULL}, 1U, &del_cnt), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_bm25_delete_batch(bm25, NULL, 1U, &del_cnt), NBASE_ERR_INVALID_ARG);

  uint64_t del_ids[5] = {2ULL, 999ULL, 2ULL, 5ULL, 7ULL};
  ASSERT_STATUS(nbase_bm25_delete_batch(bm25, del_ids, 5U, &del_cnt), NBASE_OK);
  ASSERT_EQ_U64(del_cnt, 3ULL);
  ASSERT_EQ_U64(nbase_bm25_doc_count(bm25), 5ULL);
  ASSERT_TRUE(!nbase_bm25_contains(bm25, 2ULL));
  ASSERT_TRUE(!nbase_bm25_contains(bm25, 5ULL));
  ASSERT_TRUE(!nbase_bm25_contains(bm25, 7ULL));
  ASSERT_TRUE(nbase_bm25_contains(bm25, 1ULL));
  ASSERT_TRUE(nbase_bm25_contains(bm25, 8ULL));

  nbase_scored_id_t res[8];
  uint32_t res_cnt = 0U;
  ASSERT_STATUS(nbase_bm25_search(bm25, "vector batch delete", 8U, res, &res_cnt), NBASE_OK);
  ASSERT_EQ_U64(res_cnt, 5ULL);
  for (uint32_t i = 0U; i < res_cnt; i++) {
    ASSERT_TRUE(res[i].id != 2ULL && res[i].id != 5ULL && res[i].id != 7ULL);
  }

  nbase_bm25_destroy(bm25);
}

typedef struct bm25_v16_tsan_ctx {
  nbase_bm25_t *bm25;
  atomic_bool stop;
  atomic_int errors;
} bm25_v16_tsan_ctx_t;

static void *bm25_v16_reader_fn(void *raw) {
  bm25_v16_tsan_ctx_t *ctx = (bm25_v16_tsan_ctx_t *)raw;
  static const char *const qs[4] = {
      "vector",
      "simd bm25",
      "hnsw compact",
      "nonexistent_term_v16",
  };
  uint64_t qids[3] = {1ULL, 2ULL, 999ULL};
  char *texts[3] = {NULL, NULL, NULL};
  bool found[3] = {false, false, false};
  nbase_scored_id_t bres[4U * 5U];
  uint32_t bcnts[4];

  while (!atomic_load_explicit(&ctx->stop, memory_order_relaxed)) {
    nbase_bm25_stats_t st;
    if (nbase_bm25_stats(ctx->bm25, &st) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    if (nbase_bm25_get_batch(ctx->bm25, qids, 3U, texts, found) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    nbase_free_texts(texts, 3U);
    if (nbase_bm25_search_batch(ctx->bm25, qs, 4U, 5U, 2U, bres, bcnts) != NBASE_OK) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
  }
  return NULL;
}

static void test_bm25_stats_get_batch_and_search_scratch_tsan(void) {
  nbase_bm25_t *bm25 = NULL;
  ASSERT_STATUS(nbase_bm25_create(1.5f, 0.75f, &bm25), NBASE_OK);

  nbase_bm25_stats_t st;
  ASSERT_STATUS(nbase_bm25_stats(NULL, &st), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_bm25_stats(bm25, NULL), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_bm25_stats(bm25, &st), NBASE_OK);
  ASSERT_EQ_U64(st.doc_count, 0ULL);
  ASSERT_EQ_U64(st.tombstones, 0ULL);
  ASSERT_EQ_U64(st.unique_terms, 0ULL);
  ASSERT_EQ_U64(st.total_postings, 0ULL);
  ASSERT_NEAR(st.avg_doc_len, 0.0, 1e-9, 0.0);
  ASSERT_TRUE(st.memory_bytes > 0ULL);

  static const uint64_t doc_ids[6] = {1ULL, 2ULL, 3ULL, 4ULL, 5ULL, 6ULL};
  static const char *const doc_texts[6] = {
      "vector similarity simd kernel",
      "bm25 inverted index ranking",
      "hnsw compact wal durability",
      "vector search avx2 kernel",
      "hybrid rrf fusion search",
      "vector bm25 graph rerank",
  };
  ASSERT_STATUS(nbase_bm25_index_batch(bm25, doc_ids, doc_texts, 6U), NBASE_OK);
  ASSERT_STATUS(nbase_bm25_stats(bm25, &st), NBASE_OK);
  ASSERT_EQ_U64(st.doc_count, 6ULL);
  ASSERT_EQ_U64(st.tombstones, 0ULL);
  ASSERT_TRUE(st.unique_terms >= 15ULL);
  ASSERT_TRUE(st.total_postings >= 20ULL);
  ASSERT_NEAR(st.avg_doc_len, 4.0, 1e-6, 0.0);
  uint64_t mem_before_del = st.memory_bytes;

  ASSERT_STATUS(nbase_bm25_delete(bm25, 3ULL), NBASE_OK);
  ASSERT_STATUS(nbase_bm25_delete(bm25, 5ULL), NBASE_OK);
  ASSERT_STATUS(nbase_bm25_stats(bm25, &st), NBASE_OK);
  ASSERT_EQ_U64(st.doc_count, 4ULL);
  ASSERT_EQ_U64(st.tombstones, 2ULL);

  ASSERT_STATUS(nbase_bm25_compact(bm25), NBASE_OK);
  ASSERT_STATUS(nbase_bm25_stats(bm25, &st), NBASE_OK);
  ASSERT_EQ_U64(st.doc_count, 4ULL);
  ASSERT_EQ_U64(st.tombstones, 0ULL);
  ASSERT_TRUE(st.memory_bytes <= mem_before_del);

  /* get_batch validation & mixed existing/missing/duplicate IDs */
  ASSERT_STATUS(nbase_bm25_get_batch(bm25, NULL, 0U, NULL, NULL), NBASE_OK);
  nbase_free_texts(NULL, 3U);
  uint64_t qids[4] = {1ULL, 3ULL, 6ULL, 1ULL};
  char *got_texts[4] = {NULL, NULL, NULL, NULL};
  bool got_found[4] = {false, true, false, false};
  ASSERT_STATUS(nbase_bm25_get_batch(NULL, qids, 4U, got_texts, got_found), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_bm25_get_batch(bm25, qids, 4U, NULL, got_found), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_bm25_get_batch(bm25, qids, 4U, got_texts, NULL), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_bm25_get_batch(bm25, qids, 4U, got_texts, got_found), NBASE_OK);
  ASSERT_TRUE(got_found[0] == true && strcmp(got_texts[0], doc_texts[0]) == 0);
  ASSERT_TRUE(got_found[1] == false && got_texts[1] == NULL);
  ASSERT_TRUE(got_found[2] == true && strcmp(got_texts[2], doc_texts[5]) == 0);
  ASSERT_TRUE(got_found[3] == true && strcmp(got_texts[3], doc_texts[0]) == 0);
  nbase_free_texts(got_texts, 4U);

  /* 64 mixed single-term (fast path) and multi-term batch parity vs serial */
  const char *queries[64];
  for (uint32_t q = 0U; q < 64U; q++) {
    switch (q % 4U) {
      case 0U:
        queries[q] = "vector";
        break;
      case 1U:
        queries[q] = "bm25";
        break;
      case 2U:
        queries[q] = "vector simd avx2";
        break;
      default:
        queries[q] = "missing_token_only";
        break;
    }
  }
  nbase_scored_id_t *bres = malloc(64U * 4U * sizeof(nbase_scored_id_t));
  uint32_t *bcnts = malloc(64U * sizeof(uint32_t));
  ASSERT_TRUE(bres != NULL && bcnts != NULL);
  ASSERT_STATUS(nbase_bm25_search_batch(bm25, queries, 64U, 4U, 4U, bres, bcnts), NBASE_OK);
  for (uint32_t q = 0U; q < 64U; q++) {
    nbase_scored_id_t sres[4];
    uint32_t scnt = 0U;
    ASSERT_STATUS(nbase_bm25_search(bm25, queries[q], 4U, sres, &scnt), NBASE_OK);
    ASSERT_EQ_U64(bcnts[q], scnt);
    for (uint32_t j = 0U; j < scnt; j++) {
      ASSERT_EQ_U64(bres[(size_t)q * 4U + j].id, sres[j].id);
      ASSERT_NEAR(bres[(size_t)q * 4U + j].score, sres[j].score, 1e-6, 0.0);
    }
  }
  free(bcnts);
  free(bres);

  /* Concurrent TSan readers (stats + get_batch + search_batch) vs index/delete */
  bm25_v16_tsan_ctx_t ctx = {bm25, false, 0};
  pthread_t r1, r2;
  ASSERT_TRUE(pthread_create(&r1, NULL, bm25_v16_reader_fn, &ctx) == 0);
  ASSERT_TRUE(pthread_create(&r2, NULL, bm25_v16_reader_fn, &ctx) == 0);
  for (uint32_t step = 0U; step < 12U; step++) {
    uint64_t tid = 900ULL + (step % 4ULL);
    ASSERT_STATUS(nbase_bm25_index(bm25, tid, "concurrent vector simd bm25 scratch"), NBASE_OK);
    if ((step % 3U) == 2U) {
      nbase_status_t dst = nbase_bm25_delete(bm25, tid);
      ASSERT_TRUE(dst == NBASE_OK || dst == NBASE_ERR_NOT_FOUND);
    }
  }
  atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
  pthread_join(r1, NULL);
  pthread_join(r2, NULL);
  ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);

  nbase_bm25_destroy(bm25);
}

void test_bm25_run(void) {
  RUN_TEST(test_bm25_index_search_delete_and_persistence);
  RUN_TEST(test_hybrid_rrf_fusion);
  RUN_TEST(test_bm25_crud_batch_compact_and_tsan);
  RUN_TEST(test_bm25_delete_batch_and_errors);
  RUN_TEST(test_bm25_stats_get_batch_and_search_scratch_tsan);
}
