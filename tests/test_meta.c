#include "test_common.h"

#include <stdio.h>
#include <string.h>

static bool id_above_10(uint64_t id, void *ctx) {
  (void)ctx;
  return id > 10ULL;
}

static void test_meta_crud_and_upsert_preservation(void) {
  nbase_config_t cfg = nbase_config_default(4U);
  cfg.seed = 123ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float v1[4] = {1, 0, 0, 0};
  float v2[4] = {0, 1, 0, 0};
  ASSERT_STATUS(nbase_add(db, 1ULL, v1, 4U), NBASE_OK);

  nbase_meta_field_t initial[4] = {
      {"count", {.type = NBASE_META_INT64, .as = {.i64 = 42}}},
      {"score", {.type = NBASE_META_DOUBLE, .as = {.f64 = 3.5}}},
      {"active", {.type = NBASE_META_BOOL, .as = {.b = true}}},
      {"tag", {.type = NBASE_META_STRING, .as = {.str = "alpha-doc"}}},
  };
  ASSERT_STATUS(nbase_set_meta(db, 1ULL, initial, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 999ULL, initial, 4U), NBASE_ERR_NOT_FOUND);

  nbase_meta_value_t val;
  char sbuf[32];
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "count", &val, NULL, 0U), NBASE_OK);
  ASSERT_EQ_I64(val.as.i64, 42LL);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "score", &val, NULL, 0U), NBASE_OK);
  ASSERT_NEAR(val.as.f64, 3.5, 1e-9, 0.0);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "active", &val, NULL, 0U), NBASE_OK);
  ASSERT_TRUE(val.as.b);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "tag", &val, sbuf, 4U), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "tag", &val, sbuf, sizeof(sbuf)), NBASE_OK);
  ASSERT_TRUE(strcmp(val.as.str, "alpha-doc") == 0);

  /* update_meta updates specified fields while keeping others intact. */
  nbase_meta_field_t upd[2] = {
      {"count", {.type = NBASE_META_INT64, .as = {.i64 = 99}}},
      {"extra", {.type = NBASE_META_STRING, .as = {.str = "kept"}}},
  };
  ASSERT_STATUS(nbase_update_meta(db, 1ULL, upd, 2U), NBASE_OK);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "count", &val, NULL, 0U), NBASE_OK);
  ASSERT_EQ_I64(val.as.i64, 99LL);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "tag", &val, sbuf, sizeof(sbuf)), NBASE_OK);
  ASSERT_TRUE(strcmp(val.as.str, "alpha-doc") == 0);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "extra", &val, sbuf, sizeof(sbuf)), NBASE_OK);
  ASSERT_TRUE(strcmp(val.as.str, "kept") == 0);

  /* upsert must preserve metadata on the updated vector. */
  ASSERT_STATUS(nbase_upsert(db, 1ULL, v2, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "count", &val, NULL, 0U), NBASE_OK);
  ASSERT_EQ_I64(val.as.i64, 99LL);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "tag", &val, sbuf, sizeof(sbuf)), NBASE_OK);
  ASSERT_TRUE(strcmp(val.as.str, "alpha-doc") == 0);

  /* delete_meta removes metadata; delete removes vector + metadata. */
  ASSERT_STATUS(nbase_delete_meta(db, 1ULL), NBASE_OK);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "count", &val, NULL, 0U), NBASE_ERR_NOT_FOUND);

  ASSERT_STATUS(nbase_set_meta(db, 1ULL, initial, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_delete(db, 1ULL), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 1ULL, v1, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "count", &val, NULL, 0U), NBASE_ERR_NOT_FOUND);

  nbase_close(db);
}

static void test_meta_filtering_operators_and_persistence(void) {
  const uint32_t count = 80U;
  const uint32_t dim = 8U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  ASSERT_TRUE(vecs != NULL);
  test_fill_vectors(vecs, count, dim, 3333ULL);

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.seed = 4444ULL;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  for (uint32_t i = 0U; i < count; i++) {
    ASSERT_STATUS(nbase_add(db, (uint64_t)i, vecs + (size_t)i * dim, dim), NBASE_OK);
    if (i == 0U) {
      /* String "100" must NOT satisfy numeric GT 50 (strict type bracketing). */
      nbase_meta_field_t f = {"rank", {.type = NBASE_META_STRING, .as = {.str = "100"}}};
      ASSERT_STATUS(nbase_set_meta(db, 0ULL, &f, 1U), NBASE_OK);
      continue;
    }
    const char *cat = (i % 3U == 0U) ? "engineering-ml" : "product-ops";
    nbase_meta_field_t fields[4] = {
        {"rank", {.type = NBASE_META_INT64, .as = {.i64 = (int64_t)i}}},
        {"weight", {.type = NBASE_META_DOUBLE, .as = {.f64 = (double)i * 0.5}}},
        {"verified", {.type = NBASE_META_BOOL, .as = {.b = (i % 2U == 0U)}}},
        {"category", {.type = NBASE_META_STRING, .as = {.str = cat}}},
    };
    ASSERT_STATUS(nbase_set_meta(db, (uint64_t)i, fields, 4U), NBASE_OK);
  }

  /* Query with combined meta_filter + callback filter on BOTH EXACT and HNSW. */
  nbase_meta_cond_t conds[5] = {
      {"rank", NBASE_OP_GT, {.type = NBASE_META_INT64, .as = {.i64 = 50}}},
      {"weight", NBASE_OP_LTE, {.type = NBASE_META_DOUBLE, .as = {.f64 = 38.0}}},
      {"verified", NBASE_OP_EQ, {.type = NBASE_META_BOOL, .as = {.b = true}}},
      {"category", NBASE_OP_CONTAINS, {.type = NBASE_META_STRING, .as = {.str = "ml"}}},
      {"category", NBASE_OP_EXISTS, {.type = NBASE_META_BOOL, .as = {.b = false}}},
  };
  nbase_meta_filter_t mfilter = {conds, 5U};

  nbase_search_opts_t opts = nbase_search_opts_default();
  opts.meta_filter = &mfilter;
  opts.filter = id_above_10;
  opts.ef_search = 100U;

  float query[8];
  test_fill_vectors(query, 1U, dim, 5555ULL);

  nbase_result_t exact_res[10], hnsw_res[10];
  uint32_t exact_n = 0U, hnsw_n = 0U;

  opts.mode = NBASE_SEARCH_EXACT;
  ASSERT_STATUS(nbase_search(db, query, dim, 10U, &opts, exact_res, &exact_n), NBASE_OK);

  opts.mode = NBASE_SEARCH_HNSW;
  ASSERT_STATUS(nbase_search(db, query, dim, 10U, &opts, hnsw_res, &hnsw_n), NBASE_OK);

  /* Expected matching IDs: > 50, <= 76, even, divisible by 3 -> 54, 60, 66, 72. */
  ASSERT_EQ_U64(exact_n, 4ULL);
  ASSERT_EQ_U64(hnsw_n, 4ULL);
  for (uint32_t i = 0U; i < exact_n; i++) {
    ASSERT_EQ_U64(hnsw_res[i].id, exact_res[i].id);
    ASSERT_TRUE(exact_res[i].id == 54ULL || exact_res[i].id == 60ULL || exact_res[i].id == 66ULL || exact_res[i].id == 72ULL);
  }

  /* Verify missing field does NOT match NBASE_OP_NE. */
  nbase_meta_cond_t ne_missing = {"nonexistent", NBASE_OP_NE, {.type = NBASE_META_STRING, .as = {.str = "x"}}};
  nbase_meta_filter_t ne_filter = {&ne_missing, 1U};
  opts.mode = NBASE_SEARCH_EXACT;
  opts.meta_filter = &ne_filter;
  opts.filter = NULL;
  ASSERT_STATUS(nbase_search(db, query, dim, 10U, &opts, exact_res, &exact_n), NBASE_OK);
  ASSERT_EQ_U64(exact_n, 0ULL);

  /* Save -> open_mmap -> verify metadata survives and corruption is caught. */
  char dir[128];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  ASSERT_STATUS(nbase_save(db, dir), NBASE_OK);
  nbase_close(db);

  ASSERT_STATUS(nbase_open_mmap(dir, &db), NBASE_OK);
  opts.meta_filter = &mfilter;
  opts.filter = id_above_10;
  opts.mode = NBASE_SEARCH_HNSW;
  ASSERT_STATUS(nbase_search(db, query, dim, 10U, &opts, hnsw_res, &hnsw_n), NBASE_OK);
  ASSERT_EQ_U64(hnsw_n, 4ULL);

  nbase_meta_value_t out_v;
  char sbuf[64];
  ASSERT_STATUS(nbase_get_meta(db, 60ULL, "category", &out_v, sbuf, sizeof(sbuf)), NBASE_OK);
  ASSERT_TRUE(strcmp(out_v.as.str, "engineering-ml") == 0);
  ASSERT_STATUS(nbase_set_meta(db, 60ULL, NULL, 0U), NBASE_ERR_UNSUPPORTED);
  nbase_close(db);

  char meta_path[256];
  snprintf(meta_path, sizeof(meta_path), "%s/meta.nbm", dir);
  ASSERT_TRUE(test_flip_file_byte(meta_path, 140L));
  ASSERT_STATUS(nbase_open(dir, &db), NBASE_ERR_CORRUPT);

  test_rm_rf(dir);
  free(vecs);
}

typedef struct meta_count_ctx {
  uint32_t seen;
  bool found_cat;
} meta_count_ctx_t;

static bool count_meta_cb(const char *key, const nbase_meta_value_t *val, void *raw) {
  meta_count_ctx_t *ctx = (meta_count_ctx_t *)raw;
  ctx->seen++;
  if (strcmp(key, "cat") == 0 && val->type == NBASE_META_STRING && strcmp(val->as.str, "ml") == 0) {
    ctx->found_cat = true;
  }
  return true;
}

static void test_meta_foreach_query_and_find_similar(void) {
  const uint32_t dim = 4U;
  nbase_config_t cfg = nbase_config_default(dim);
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  float v2[4] = {0.9f, 0.1f, 0.0f, 0.0f};
  float v3[4] = {0.0f, 1.0f, 0.0f, 0.0f};
  ASSERT_STATUS(nbase_add(db, 30ULL, v3, dim), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 10ULL, v1, dim), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 20ULL, v2, dim), NBASE_OK);

  nbase_meta_field_t m1[2] = {
      {"cat", {.type = NBASE_META_STRING, .as = {.str = "ml"}}},
      {"score", {.type = NBASE_META_INT64, .as = {.i64 = 95}}},
  };
  nbase_meta_field_t m2[2] = {
      {"cat", {.type = NBASE_META_STRING, .as = {.str = "ml"}}},
      {"score", {.type = NBASE_META_INT64, .as = {.i64 = 75}}},
  };
  ASSERT_STATUS(nbase_set_meta(db, 10ULL, m1, 2U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 20ULL, m2, 2U), NBASE_OK);

  meta_count_ctx_t mctx = {0U, false};
  ASSERT_STATUS(nbase_foreach_meta(db, 10ULL, count_meta_cb, &mctx), NBASE_OK);
  ASSERT_EQ_U64(mctx.seen, 2ULL);
  ASSERT_TRUE(mctx.found_cat);
  ASSERT_STATUS(nbase_foreach_meta(db, 999ULL, count_meta_cb, &mctx), NBASE_ERR_NOT_FOUND);

  nbase_meta_cond_t cond = {"cat", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "ml"}}};
  nbase_meta_filter_t flt = {&cond, 1U};
  uint64_t qids[4];
  uint32_t qcnt = 0U;
  ASSERT_STATUS(nbase_query_meta(db, &flt, 4U, qids, &qcnt), NBASE_OK);
  ASSERT_EQ_U64(qcnt, 2ULL);
  ASSERT_EQ_U64(qids[0], 10ULL);
  ASSERT_EQ_U64(qids[1], 20ULL);

  nbase_result_t sim[2];
  uint32_t scnt = 0U;
  ASSERT_STATUS(nbase_find_similar(db, 10ULL, 2U, NULL, sim, &scnt), NBASE_OK);
  ASSERT_EQ_U64(scnt, 2ULL);
  ASSERT_EQ_U64(sim[0].id, 20ULL);
  ASSERT_EQ_U64(sim[1].id, 30ULL);
  ASSERT_STATUS(nbase_find_similar(db, 999ULL, 2U, NULL, sim, &scnt), NBASE_ERR_NOT_FOUND);

  nbase_close(db);
}

static void test_meta_and_text_batch_atomicity_and_edge_cases(void) {
  const uint32_t dim = 3U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.cache_capacity = 16U;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  uint64_t ids[3] = {1ULL, 2ULL, 3ULL};
  float vecs[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  ASSERT_STATUS(nbase_add_batch(db, ids, vecs, 3U, dim, 1U), NBASE_OK);

  /* Empty batches succeed cleanly. */
  ASSERT_STATUS(nbase_set_meta_batch(db, NULL, 0U), NBASE_OK);
  ASSERT_STATUS(nbase_update_meta_batch(db, NULL, 0U), NBASE_OK);
  ASSERT_STATUS(nbase_set_text_batch(db, NULL, NULL, 0U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta_batch(db, NULL, 1U), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_set_text_batch(db, ids, NULL, 1U), NBASE_ERR_INVALID_ARG);

  nbase_meta_field_t f1[2] = {
      {"dept", {.type = NBASE_META_STRING, .as = {.str = "eng"}}},
      {"score", {.type = NBASE_META_INT64, .as = {.i64 = 10}}}};
  nbase_meta_field_t f2[2] = {
      {"dept", {.type = NBASE_META_STRING, .as = {.str = "ops"}}},
      {"score", {.type = NBASE_META_INT64, .as = {.i64 = 20}}}};
  nbase_meta_field_t f1_dup[2] = {
      {"dept", {.type = NBASE_META_STRING, .as = {.str = "ml"}}},
      {"score", {.type = NBASE_META_INT64, .as = {.i64 = 30}}}};

  /* Atomicity: if any ID in the batch is missing, zero items mutate. */
  nbase_meta_batch_item_t missing_batch[2] = {
      {1ULL, f1, 2U},
      {999ULL, f2, 2U}};
  ASSERT_STATUS(nbase_set_meta_batch(db, missing_batch, 2U), NBASE_ERR_NOT_FOUND);
  nbase_meta_value_t mv;
  char sbuf[32];
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "dept", &mv, sbuf, sizeof(sbuf)), NBASE_ERR_NOT_FOUND);

  /* Batch set with duplicate ID overwrite preserves last write. */
  nbase_meta_batch_item_t valid_batch[3] = {
      {1ULL, f1, 2U},
      {2ULL, f2, 2U},
      {1ULL, f1_dup, 2U}};
  ASSERT_STATUS(nbase_set_meta_batch(db, valid_batch, 3U), NBASE_OK);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "dept", &mv, sbuf, sizeof(sbuf)), NBASE_OK);
  ASSERT_TRUE(strcmp(mv.as.str, "ml") == 0);
  ASSERT_STATUS(nbase_get_meta(db, 2ULL, "score", &mv, NULL, 0U), NBASE_OK);
  ASSERT_EQ_U64((uint64_t)mv.as.i64, 20ULL);

  /* Populate LRU search cache with filtered query, then verify batch update invalidates cache. */
  nbase_meta_cond_t cond = {"dept", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "ops"}}};
  nbase_meta_filter_t flt = {&cond, 1U};
  nbase_search_opts_t sopts = nbase_search_opts_default();
  sopts.meta_filter = &flt;
  nbase_result_t hits[3];
  uint32_t hit_cnt = 0U;
  ASSERT_STATUS(nbase_search(db, vecs, dim, 3U, &sopts, hits, &hit_cnt), NBASE_OK);
  ASSERT_EQ_U64(hit_cnt, 1ULL);
  ASSERT_EQ_U64(hits[0].id, 2ULL);

  nbase_meta_field_t promote = {"dept", {.type = NBASE_META_STRING, .as = {.str = "ops"}}};
  nbase_meta_batch_item_t upd[1] = {{1ULL, &promote, 1U}};
  ASSERT_STATUS(nbase_update_meta_batch(db, upd, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_get_meta(db, 1ULL, "score", &mv, NULL, 0U), NBASE_OK);
  ASSERT_EQ_U64((uint64_t)mv.as.i64, 30ULL); /* Existing field preserved on update! */

  ASSERT_STATUS(nbase_search(db, vecs, dim, 3U, &sopts, hits, &hit_cnt), NBASE_OK);
  ASSERT_EQ_U64(hit_cnt, 2ULL);

  /* Batch text indexing atomicity + BM25 query. */
  uint64_t bad_tids[2] = {1ULL, 888ULL};
  const char *bad_texts[2] = {"avx2 kernel", "missing"};
  ASSERT_STATUS(nbase_set_text_batch(db, bad_tids, bad_texts, 2U), NBASE_ERR_NOT_FOUND);
  size_t tlen = 0U;
  ASSERT_STATUS(nbase_get_text(db, 1ULL, NULL, 0U, &tlen), NBASE_ERR_NOT_FOUND);

  uint64_t tids[3] = {1ULL, 2ULL, 3ULL};
  const char *texts[3] = {
      "vector engine c11 avx2 search",
      "distributed storage raft consensus",
      "graph rerank hybrid bm25 search"};
  ASSERT_STATUS(nbase_set_text_batch(db, tids, texts, 3U), NBASE_OK);

  nbase_scored_id_t thits[3];
  uint32_t tcnt = 0U;
  ASSERT_STATUS(nbase_search_text(db, "avx2 search", 3U, &flt, thits, &tcnt), NBASE_OK);
  ASSERT_EQ_U64(tcnt, 1ULL);
  ASSERT_EQ_U64(thits[0].id, 1ULL);

  nbase_close(db);
}

void test_meta_run(void) {
  RUN_TEST(test_meta_crud_and_upsert_preservation);
  RUN_TEST(test_meta_filtering_operators_and_persistence);
  RUN_TEST(test_meta_foreach_query_and_find_similar);
  RUN_TEST(test_meta_and_text_batch_atomicity_and_edge_cases);
}
