#include "test_common.h"

#include <stdio.h>
#include <string.h>

static void test_db_text_and_hybrid_search(void) {
  nbase_config_t cfg = nbase_config_default(4U);
  cfg.seed = 4242ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  float v2[4] = {0.9f, 0.1f, 0.0f, 0.0f};
  float v3[4] = {0.0f, 0.0f, 1.0f, 0.0f};
  float v2_new[4] = {0.85f, 0.15f, 0.0f, 0.0f};

  ASSERT_STATUS(nbase_add(db, 1ULL, v1, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 2ULL, v2, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 3ULL, v3, 4U), NBASE_OK);

  nbase_meta_field_t m1[1] = {{"tag", {.type = NBASE_META_STRING, .as = {.str = "ml"}}}};
  nbase_meta_field_t m2[1] = {{"tag", {.type = NBASE_META_STRING, .as = {.str = "sys"}}}};
  nbase_meta_field_t m3[1] = {{"tag", {.type = NBASE_META_STRING, .as = {.str = "sys"}}}};
  ASSERT_STATUS(nbase_set_meta(db, 1ULL, m1, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 2ULL, m2, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 3ULL, m3, 1U), NBASE_OK);

  /* Cannot set text for missing vector */
  ASSERT_STATUS(nbase_set_text(db, 999ULL, "missing"), NBASE_ERR_NOT_FOUND);

  ASSERT_STATUS(nbase_set_text(db, 1ULL, "python notebook neural network"), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 2ULL, "pure c11 avx2 hnsw vector engine"), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 3ULL, "c11 avx2 simd quantization engine engine"), NBASE_OK);

  /* Size probe & buffer retrieval */
  size_t tlen = 0U;
  ASSERT_STATUS(nbase_get_text(db, 2ULL, NULL, 0U, &tlen), NBASE_OK);
  ASSERT_EQ_U64(tlen, strlen("pure c11 avx2 hnsw vector engine"));
  char tiny[8];
  ASSERT_STATUS(nbase_get_text(db, 2ULL, tiny, sizeof(tiny), NULL), NBASE_ERR_INVALID_ARG);
  char full[128];
  ASSERT_STATUS(nbase_get_text(db, 2ULL, full, sizeof(full), &tlen), NBASE_OK);
  ASSERT_TRUE(strcmp(full, "pure c11 avx2 hnsw vector engine") == 0);

  nbase_stats_t st;
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.text_docs, 3ULL);

  /* Upserting a vector preserves associated text */
  ASSERT_STATUS(nbase_upsert(db, 2ULL, v2_new, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_get_text(db, 2ULL, full, sizeof(full), NULL), NBASE_OK);
  ASSERT_TRUE(strcmp(full, "pure c11 avx2 hnsw vector engine") == 0);

  /* Filtered BM25 search via DB */
  nbase_meta_cond_t cond = {"tag", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "sys"}}};
  nbase_meta_filter_t mflt = {&cond, 1U};
  nbase_scored_id_t thits[4];
  uint32_t tcnt = 0U;
  ASSERT_STATUS(nbase_search_text(db, "avx2 engine", 4U, &mflt, thits, &tcnt), NBASE_OK);
  ASSERT_EQ_U64(tcnt, 2ULL);

  /* Combined Hybrid RRF search (vector + text) favors id 2 which is close in vector space AND matches text */
  float qvec[4] = {0.9f, 0.1f, 0.0f, 0.0f};
  nbase_scored_id_t hhits[4];
  uint32_t hcnt = 0U;
  ASSERT_STATUS(nbase_search_hybrid(db, qvec, 4U, "avx2 hnsw c11", 3U, 0.5f, 60.0f, NULL, hhits, &hcnt), NBASE_OK);
  ASSERT_EQ_U64(hcnt, 3ULL);
  ASSERT_EQ_U64(hhits[0].id, 2ULL);

  /* Vector-only and text-only modes + metadata filter */
  ASSERT_STATUS(nbase_search_hybrid(db, qvec, 4U, NULL, 2U, 0.5f, 60.0f, NULL, hhits, &hcnt), NBASE_OK);
  ASSERT_EQ_U64(hcnt, 2ULL);
  ASSERT_STATUS(nbase_search_hybrid(db, NULL, 0U, "neural network", 2U, 0.5f, 60.0f, NULL, hhits, &hcnt), NBASE_OK);
  ASSERT_EQ_U64(hcnt, 1ULL);
  ASSERT_EQ_U64(hhits[0].id, 1ULL);

  nbase_search_opts_t sopts = nbase_search_opts_default();
  sopts.meta_filter = &mflt;
  ASSERT_STATUS(nbase_search_hybrid(db, qvec, 4U, "neural avx2", 3U, 0.5f, 60.0f, &sopts, hhits, &hcnt), NBASE_OK);
  for (uint32_t i = 0U; i < hcnt; i++) {
    ASSERT_TRUE(hhits[i].id == 2ULL || hhits[i].id == 3ULL);
  }

  /* Invalid input validation */
  ASSERT_STATUS(nbase_search_hybrid(db, NULL, 0U, NULL, 3U, 0.5f, 60.0f, NULL, hhits, &hcnt), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_search_hybrid(db, qvec, 3U, "avx2", 3U, 0.5f, 60.0f, NULL, hhits, &hcnt), NBASE_ERR_DIM_MISMATCH);

  /* Delete text directly and delete vector cascade */
  ASSERT_STATUS(nbase_delete_text(db, 1ULL), NBASE_OK);
  ASSERT_STATUS(nbase_get_text(db, 1ULL, full, sizeof(full), NULL), NBASE_ERR_NOT_FOUND);
  ASSERT_STATUS(nbase_delete(db, 2ULL), NBASE_OK);
  ASSERT_STATUS(nbase_stats(db, &st), NBASE_OK);
  ASSERT_EQ_U64(st.text_docs, 1ULL);

  nbase_close(db);
}

static void test_db_text_persistence_and_wal_recovery(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_config_t cfg = nbase_config_default(3U);
  cfg.seed = 777ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);
  ASSERT_STATUS(nbase_wal_enable(db, dir), NBASE_OK);

  float v1[3] = {1.0f, 0.0f, 0.0f};
  float v2[3] = {0.0f, 1.0f, 0.0f};
  ASSERT_STATUS(nbase_add(db, 10ULL, v1, 3U), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 20ULL, v2, 3U), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 10ULL, "alpha wal recovery text"), NBASE_OK);
  ASSERT_STATUS(nbase_set_text(db, 20ULL, "beta transient text"), NBASE_OK);
  ASSERT_STATUS(nbase_delete_text(db, 20ULL), NBASE_OK);
  nbase_close(db);

  /* Reopen with WAL replay (without explicit save after set_text) */
  nbase_db_t *replayed = NULL;
  ASSERT_STATUS(nbase_open_wal(dir, &replayed), NBASE_OK);
  char buf[128];
  ASSERT_STATUS(nbase_get_text(replayed, 10ULL, buf, sizeof(buf), NULL), NBASE_OK);
  ASSERT_TRUE(strcmp(buf, "alpha wal recovery text") == 0);
  ASSERT_STATUS(nbase_get_text(replayed, 20ULL, buf, sizeof(buf), NULL), NBASE_ERR_NOT_FOUND);

  ASSERT_STATUS(nbase_set_text(replayed, 20ULL, "gamma hybrid document"), NBASE_OK);
  ASSERT_STATUS(nbase_wal_checkpoint(replayed), NBASE_OK);
  nbase_close(replayed);

  /* Reopen read-only / snapshot open from bm25.nbbm */
  nbase_db_t *reopened = NULL;
  ASSERT_STATUS(nbase_open(dir, &reopened), NBASE_OK);
  nbase_stats_t st;
  ASSERT_STATUS(nbase_stats(reopened, &st), NBASE_OK);
  ASSERT_EQ_U64(st.text_docs, 2ULL);

  nbase_scored_id_t hits[2];
  uint32_t cnt = 0U;
  ASSERT_STATUS(nbase_search_hybrid(reopened, v2, 3U, "gamma hybrid", 2U, 0.5f, 60.0f, NULL, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 2ULL);
  ASSERT_EQ_U64(hits[0].id, 20ULL);
  nbase_close(reopened);

  /* Corrupt bm25.nbbm detection */
  char bm_path[300];
  snprintf(bm_path, sizeof(bm_path), "%s/bm25.nbbm", dir);
  ASSERT_TRUE(test_flip_file_byte(bm_path, 4L));
  ASSERT_STATUS(nbase_open(dir, &reopened), NBASE_ERR_CORRUPT);

  test_rm_rf(dir);
}

void test_hybrid_run(void) {
  RUN_TEST(test_db_text_and_hybrid_search);
  RUN_TEST(test_db_text_persistence_and_wal_recovery);
}
