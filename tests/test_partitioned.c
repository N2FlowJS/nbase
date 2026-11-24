#include "test_common.h"

#include <stdio.h>
#include <string.h>

static void test_pdb_durable_eviction_and_partition_ids(void) {
  char dir[128];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_pdb_config_t cfg = nbase_pdb_config_default(4U);
  cfg.partition_capacity = 2U;
  cfg.max_active_partitions = 1U;
  cfg.seed = 4242ULL;

  nbase_pdb_t *pdb = NULL;
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  float vecs[6 * 4];
  uint64_t ids[6] = {0ULL, 1ULL, 2ULL, 3ULL, 4ULL, 5ULL};
  test_fill_vectors(vecs, 6U, 4U, 888ULL);

  for (uint32_t i = 0U; i < 6U; i++) {
    ASSERT_STATUS(nbase_pdb_add(pdb, ids[i], vecs + (size_t)i * 4U, 4U), NBASE_OK);
    char label[16];
    snprintf(label, sizeof(label), "vec-%u", i);
    nbase_meta_field_t f[2] = {
        {"label", {.type = NBASE_META_STRING, .as = {.str = label}}},
        {"seq", {.type = NBASE_META_INT64, .as = {.i64 = (int64_t)i}}},
    };
    ASSERT_STATUS(nbase_pdb_set_meta(pdb, ids[i], f, 2U), NBASE_OK);
  }
  ASSERT_STATUS(nbase_pdb_add(pdb, 0ULL, vecs, 4U), NBASE_ERR_ALREADY_EXISTS);

  nbase_pdb_stats_t st;
  ASSERT_STATUS(nbase_pdb_stats(pdb, &st), NBASE_OK);
  ASSERT_EQ_U64(st.total_vectors, 6ULL);
  ASSERT_EQ_U64(st.total_partitions, 3ULL);
  ASSERT_EQ_U64(st.loaded_partitions, 1ULL);
  ASSERT_TRUE(st.eviction_count >= 2ULL);

  /* Search explicit evicted partition 0 even though partition 2 is currently active. */
  uint32_t target_pid = 0U;
  nbase_pdb_search_opts_t p_opts = nbase_pdb_search_opts_default();
  p_opts.partition_ids = &target_pid;
  p_opts.partition_id_count = 1U;

  nbase_result_t p0_hits[4];
  uint32_t p0_n = 0U;
  ASSERT_STATUS(nbase_pdb_search(pdb, vecs, 4U, 4U, &p_opts, p0_hits, &p0_n), NBASE_OK);
  ASSERT_EQ_U64(p0_n, 2ULL);
  ASSERT_EQ_U64(p0_hits[0].id, 0ULL);

  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);

  /* Reopen from disk -> all 6 of 6 vectors and metadata must survive eviction & restart. */
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_stats(pdb, &st), NBASE_OK);
  ASSERT_EQ_U64(st.total_vectors, 6ULL);
  ASSERT_EQ_U64(st.total_partitions, 3ULL);

  for (uint32_t i = 0U; i < 6U; i++) {
    ASSERT_TRUE(nbase_pdb_contains(pdb, ids[i]));
    float out_v[4] = {0};
    ASSERT_STATUS(nbase_pdb_get(pdb, ids[i], out_v, 4U), NBASE_OK);
    for (uint32_t d = 0U; d < 4U; d++) {
      ASSERT_NEAR(out_v[d], vecs[i * 4U + d], 1e-6, 0.0);
    }
    nbase_meta_value_t mv;
    char sbuf[32];
    ASSERT_STATUS(nbase_pdb_get_meta(pdb, ids[i], "label", &mv, sbuf, sizeof(sbuf)), NBASE_OK);
    char expected_lbl[16];
    snprintf(expected_lbl, sizeof(expected_lbl), "vec-%u", i);
    ASSERT_TRUE(strcmp(mv.as.str, expected_lbl) == 0);
  }

  /* Cross-partition search (HNSW and EXACT) matches brute-force oracle across all 3 partitions. */
  float query[4] = {0.25f, 0.5f, 0.75f, 0.1f};
  nbase_pdb_search_opts_t all_opts = nbase_pdb_search_opts_default();
  all_opts.search_all_partitions = true;
  all_opts.search.mode = NBASE_SEARCH_EXACT;

  nbase_result_t actual[6], expected[6];
  uint32_t act_n = 0U, exp_n = 0U;
  ASSERT_STATUS(nbase_pdb_search(pdb, query, 4U, 6U, &all_opts, actual, &act_n), NBASE_OK);
  test_brute_force(ids, vecs, NULL, 6U, 4U, query, 6U, NBASE_METRIC_EUCLIDEAN, NULL, NULL, expected, &exp_n);
  ASSERT_EQ_U64(act_n, exp_n);
  for (uint32_t i = 0U; i < exp_n; i++) {
    ASSERT_EQ_U64(actual[i].id, expected[i].id);
  }

  all_opts.search.mode = NBASE_SEARCH_HNSW;
  ASSERT_STATUS(nbase_pdb_search(pdb, query, 4U, 6U, &all_opts, actual, &act_n), NBASE_OK);
  ASSERT_TRUE(test_recall_at(actual, act_n, expected, exp_n, 4U) >= 0.99);

  /* Upsert & delete across evicted partitions. */
  float updated[4] = {9.0f, 9.0f, 9.0f, 9.0f};
  ASSERT_STATUS(nbase_pdb_upsert(pdb, 0ULL, updated, 4U), NBASE_OK);
  float got[4] = {0};
  ASSERT_STATUS(nbase_pdb_get(pdb, 0ULL, got, 4U), NBASE_OK);
  ASSERT_NEAR(got[0], 9.0, 1e-6, 0.0);

  ASSERT_STATUS(nbase_pdb_delete(pdb, 1ULL), NBASE_OK);
  ASSERT_TRUE(!nbase_pdb_contains(pdb, 1ULL));
  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);

  /* Corrupt manifest.nbp -> open rejects with NBASE_ERR_CORRUPT. */
  char mpath[256];
  snprintf(mpath, sizeof(mpath), "%s/manifest.nbp", dir);
  ASSERT_TRUE(test_flip_file_byte(mpath, 16L));
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_ERR_CORRUPT);

  test_rm_rf(dir);
}

static void test_pdb_batch_text_hybrid_and_cache_stats(void) {
  char dir[128];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_pdb_config_t cfg = nbase_pdb_config_default(4U);
  cfg.partition_capacity = 2U;
  cfg.max_active_partitions = 1U;
  cfg.seed = 3131ULL;

  nbase_pdb_t *pdb = NULL;
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  uint64_t ids[4] = {10ULL, 20ULL, 30ULL, 40ULL};
  float vecs[4 * 4] = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.95f, 0.05f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.88f, 0.12f, 0.0f, 0.0f,
  };
  ASSERT_STATUS(nbase_pdb_add_batch(pdb, ids, vecs, 4U, 4U, 2U), NBASE_OK);

  nbase_meta_field_t mf_sys = {"group", {.type = NBASE_META_STRING, .as = {.str = "sys"}}};
  nbase_meta_field_t mf_ml = {"group", {.type = NBASE_META_STRING, .as = {.str = "ml"}}};
  ASSERT_STATUS(nbase_pdb_set_meta(pdb, 10ULL, &mf_sys, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_set_meta(pdb, 20ULL, &mf_ml, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_set_meta(pdb, 40ULL, &mf_sys, 1U), NBASE_OK);

  ASSERT_STATUS(nbase_pdb_set_text(pdb, 10ULL, "python machine learning model"), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_set_text(pdb, 20ULL, "pure c11 avx2 hybrid vector engine"), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_set_text(pdb, 40ULL, "avx2 simd quantization engine"), NBASE_OK);

  char tbuf[128];
  size_t tlen = 0U;
  ASSERT_STATUS(nbase_pdb_get_text(pdb, 20ULL, tbuf, sizeof(tbuf), &tlen), NBASE_OK);
  ASSERT_TRUE(strcmp(tbuf, "pure c11 avx2 hybrid vector engine") == 0);

  /* Cross-partition metadata query */
  nbase_meta_cond_t cond = {"group", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "sys"}}};
  nbase_meta_filter_t mflt = {&cond, 1U};
  uint64_t qids[4];
  uint32_t qcnt = 0U;
  ASSERT_STATUS(nbase_pdb_query_meta(pdb, &mflt, 4U, qids, &qcnt), NBASE_OK);
  ASSERT_EQ_U64(qcnt, 2ULL);
  ASSERT_EQ_U64(qids[0], 10ULL);
  ASSERT_EQ_U64(qids[1], 40ULL);

  /* Cross-partition text and hybrid search */
  nbase_scored_id_t thits[4];
  uint32_t thit_n = 0U;
  ASSERT_STATUS(nbase_pdb_search_text(pdb, "avx2 engine", 4U, NULL, thits, &thit_n), NBASE_OK);
  ASSERT_EQ_U64(thit_n, 2ULL);

  float qv[4] = {0.9f, 0.1f, 0.0f, 0.0f};
  nbase_scored_id_t hhits[4];
  uint32_t hhit_n = 0U;
  ASSERT_STATUS(nbase_pdb_search_hybrid(pdb, qv, 4U, "avx2 hybrid", 4U, 0.5f, 60.0f, NULL, hhits, &hhit_n), NBASE_OK);
  ASSERT_TRUE(hhit_n >= 2U);

  /* Cross-partition find_similar excludes source id */
  nbase_result_t sim[3];
  uint32_t sim_n = 0U;
  ASSERT_STATUS(nbase_pdb_find_similar(pdb, 10ULL, 2U, NULL, sim, &sim_n), NBASE_OK);
  ASSERT_EQ_U64(sim_n, 2ULL);
  ASSERT_TRUE(sim[0].id != 10ULL && sim[1].id != 10ULL);

  /* Compact, rebuild HNSW, and check cache_hits/cache_misses/memory_bytes */
  ASSERT_STATUS(nbase_pdb_compact(pdb), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_rebuild_hnsw_mt(pdb, 2U), NBASE_OK);

  ASSERT_STATUS(nbase_pdb_delete_meta(pdb, 40ULL), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_delete_text(pdb, 40ULL), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_get_text(pdb, 40ULL, tbuf, sizeof(tbuf), NULL), NBASE_ERR_NOT_FOUND);

  nbase_pdb_stats_t st;
  ASSERT_STATUS(nbase_pdb_stats(pdb, &st), NBASE_OK);
  ASSERT_TRUE(st.cache_hits > 0ULL);
  ASSERT_TRUE(st.cache_misses > 0ULL);
  ASSERT_TRUE(st.memory_bytes > 0ULL);

  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);
  test_rm_rf(dir);
}

static void test_pdb_delete_batch_filter_search_batch_and_graph(void) {
  char dir[128];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_pdb_config_t cfg = nbase_pdb_config_default(3U);
  cfg.partition_capacity = 2U;
  cfg.max_active_partitions = 1U;
  cfg.enable_hnsw = false;
  nbase_pdb_t *pdb = NULL;
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  uint64_t ids[5] = {1ULL, 2ULL, 3ULL, 4ULL, 5ULL};
  float vecs[5 * 3] = {
      1.0f, 0.0f, 0.0f,
      0.95f, 0.05f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.96f, 0.04f,
      0.0f, 0.0f, 1.0f
  };
  ASSERT_STATUS(nbase_pdb_add_batch(pdb, ids, vecs, 5U, 3U, 2U), NBASE_OK);

  for (uint32_t i = 0U; i < 5U; i++) {
    nbase_meta_field_t fld = {"tag", {.type = NBASE_META_STRING, .as = {.str = (i >= 2U && i <= 3U) ? "drop" : "keep"}}};
    ASSERT_STATUS(nbase_pdb_set_meta(pdb, ids[i], &fld, 1U), NBASE_OK);
  }

  /* Batch search across partitions */
  float bqueries[2 * 3] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  nbase_pdb_search_opts_t sopts = nbase_pdb_search_opts_default();
  sopts.search_all_partitions = true;
  nbase_result_t bres[4];
  uint32_t bcnts[2] = {0U, 0U};
  ASSERT_STATUS(nbase_pdb_search_batch(pdb, bqueries, 2U, 3U, 2U, &sopts, bres, bcnts), NBASE_OK);
  ASSERT_EQ_U64(bcnts[0], 2ULL);
  ASSERT_EQ_U64(bcnts[1], 2ULL);
  ASSERT_EQ_U64(bres[0].id, 1ULL);
  ASSERT_EQ_U64(bres[2].id, 3ULL);

  /* Extract relationships across partitions (2<->1 in p0, 3<->4 straddling p1/p1) */
  nbase_edge_t *edges = NULL;
  uint64_t edge_cnt = 0ULL;
  ASSERT_STATUS(nbase_pdb_extract_relationships(pdb, 2U, 0.25f, &sopts, &edges, &edge_cnt), NBASE_OK);
  ASSERT_TRUE(edge_cnt >= 2ULL && edges != NULL);
  nbase_free_edges(edges);

  /* Delete batch removes id 5 and ignores missing id 999 */
  uint64_t del_ids[2] = {5ULL, 999ULL};
  uint32_t deleted = 0U;
  ASSERT_STATUS(nbase_pdb_delete_batch(pdb, del_ids, 2U, &deleted), NBASE_OK);
  ASSERT_EQ_U64(deleted, 1ULL);
  ASSERT_TRUE(!nbase_pdb_contains(pdb, 5ULL));

  /* Delete by metadata filter removes ids 3 and 4 across partitions */
  nbase_meta_cond_t cond = {"tag", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "drop"}}};
  nbase_meta_filter_t flt = {&cond, 1U};
  ASSERT_STATUS(nbase_pdb_delete_by_filter(pdb, &flt, &deleted), NBASE_OK);
  ASSERT_EQ_U64(deleted, 2ULL);

  nbase_pdb_stats_t st;
  ASSERT_STATUS(nbase_pdb_stats(pdb, &st), NBASE_OK);
  ASSERT_EQ_U64(st.total_vectors, 2ULL);
  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);

  /* Reopen verifies persisted post-delete manifest and vectors */
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_stats(pdb, &st), NBASE_OK);
  ASSERT_EQ_U64(st.total_vectors, 2ULL);
  ASSERT_TRUE(nbase_pdb_contains(pdb, 1ULL));
  ASSERT_TRUE(nbase_pdb_contains(pdb, 2ULL));
  ASSERT_TRUE(!nbase_pdb_contains(pdb, 3ULL));
  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);
  test_rm_rf(dir);
}

static void test_pdb_upsert_batch_get_batch_and_rerank(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_pdb_config_t cfg = nbase_pdb_config_default(3U);
  cfg.partition_capacity = 2U;
  cfg.max_active_partitions = 1U;
  cfg.enable_hnsw = false;
  nbase_pdb_t *pdb = NULL;
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  uint64_t init_ids[3] = {1ULL, 2ULL, 3ULL};
  float init_vecs[9] = {
      1.0f, 0.0f, 0.0f,
      0.99f, 0.01f, 0.0f,
      0.8f, 0.6f, 0.0f
  };
  ASSERT_STATUS(nbase_pdb_add_batch(pdb, init_ids, init_vecs, 3U, 3U, 2U), NBASE_OK);

  /* Mixed upsert_batch updates IDs 2 & 3 across partitions and adds new ID 4 */
  uint64_t up_ids[3] = {2ULL, 3ULL, 4ULL};
  float up_vecs[9] = {
      0.98f, 0.02f, 0.0f,
      0.8f, 0.6f, 0.0f,
      0.0f, 1.0f, 0.0f
  };
  ASSERT_STATUS(nbase_pdb_upsert_batch(pdb, up_ids, up_vecs, 3U, 3U, 2U), NBASE_OK);

  nbase_meta_field_t f1 = {"rank_boost", {.type = NBASE_META_DOUBLE, .as = {.f64 = 0.0}}};
  nbase_meta_field_t f3 = {"rank_boost", {.type = NBASE_META_DOUBLE, .as = {.f64 = 2.0}}};
  ASSERT_STATUS(nbase_pdb_set_meta(pdb, 1ULL, &f1, 1U), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_set_meta(pdb, 3ULL, &f3, 1U), NBASE_OK);

  /* get_batch retrieves across partitions + flags missing ID 999 */
  uint64_t get_ids[4] = {1ULL, 999ULL, 2ULL, 4ULL};
  float out_vecs[12];
  bool found[4] = {false, false, false, false};
  ASSERT_STATUS(nbase_pdb_get_batch(pdb, get_ids, 4U, 3U, out_vecs, found), NBASE_OK);
  ASSERT_TRUE(found[0] && !found[1] && found[2] && found[3]);
  ASSERT_NEAR(out_vecs[0], 1.0, 1e-6, 0.0);
  ASSERT_NEAR(out_vecs[3], 0.0, 1e-6, 0.0);
  ASSERT_NEAR(out_vecs[6], 0.98, 1e-6, 0.0);
  ASSERT_NEAR(out_vecs[10], 1.0, 1e-6, 0.0);

  /* Search + MMR diversity rerank + Weighted rerank across partitions */
  float q[3] = {1.0f, 0.0f, 0.0f};
  nbase_pdb_search_opts_t sopts = nbase_pdb_search_opts_default();
  sopts.search_all_partitions = true;
  nbase_result_t cands[4];
  uint32_t cand_cnt = 0U;
  ASSERT_STATUS(nbase_pdb_search(pdb, q, 3U, 4U, &sopts, cands, &cand_cnt), NBASE_OK);
  ASSERT_EQ_U64(cand_cnt, 4ULL);

  nbase_result_t mmr_out[2];
  uint32_t mmr_cnt = 0U;
  ASSERT_STATUS(nbase_pdb_rerank_mmr(pdb, cands, cand_cnt, 2U, 0.2f, mmr_out, &mmr_cnt), NBASE_OK);
  ASSERT_EQ_U64(mmr_cnt, 2ULL);
  ASSERT_EQ_U64(mmr_out[0].id, 1ULL);
  ASSERT_TRUE(mmr_out[1].id == 4ULL || mmr_out[1].id == 3ULL);

  nbase_rerank_weight_t w = {"rank_boost", 1.0f};
  nbase_result_t w_out[2];
  uint32_t w_cnt = 0U;
  ASSERT_STATUS(nbase_pdb_rerank_weighted(pdb, cands, cand_cnt, &w, 1U, 2U, w_out, &w_cnt), NBASE_OK);
  ASSERT_EQ_U64(w_cnt, 2ULL);
  ASSERT_EQ_U64(w_out[0].id, 3ULL);

  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);
  test_rm_rf(dir);
}

static void test_pdb_list_ids_and_search_radius(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_pdb_config_t cfg = nbase_pdb_config_default(3U);
  cfg.partition_capacity = 2U;
  cfg.max_active_partitions = 1U;
  cfg.enable_hnsw = false;
  nbase_pdb_t *pdb = NULL;
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  uint64_t ids[5] = {50ULL, 10ULL, 30ULL, 20ULL, 40ULL};
  float vecs[15] = {
      1.0f, 0.0f, 0.0f,
      1.02f, 0.0f, 0.0f,
      1.15f, 0.0f, 0.0f,
      3.0f, 0.0f, 0.0f,
      4.0f, 0.0f, 0.0f
  };
  ASSERT_STATUS(nbase_pdb_add_batch(pdb, ids, vecs, 5U, 3U, 2U), NBASE_OK);

  uint64_t page[2] = {0ULL, 0ULL};
  uint32_t cnt = 0U;
  uint64_t total = 0ULL;
  ASSERT_STATUS(nbase_pdb_list_ids(pdb, 2ULL, 2U, page, &cnt, &total), NBASE_OK);
  ASSERT_EQ_U64(total, 5ULL);
  ASSERT_EQ_U64(cnt, 2ULL);
  ASSERT_EQ_U64(page[0], 30ULL);
  ASSERT_EQ_U64(page[1], 40ULL);

  float q[3] = {1.0f, 0.0f, 0.0f};
  nbase_pdb_search_opts_t sopts = nbase_pdb_search_opts_default();
  sopts.search_all_partitions = true;
  nbase_result_t hits[5];
  ASSERT_STATUS(nbase_pdb_search_radius(pdb, q, 3U, 0.2f, 5U, &sopts, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 3ULL);
  ASSERT_EQ_U64(hits[0].id, 50ULL);
  ASSERT_EQ_U64(hits[1].id, 10ULL);
  ASSERT_EQ_U64(hits[2].id, 30ULL);

  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);
  test_rm_rf(dir);
}

static void test_pdb_meta_text_batch_and_filtered_relationships(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_pdb_config_t cfg = nbase_pdb_config_default(3U);
  cfg.partition_capacity = 2U;
  cfg.max_active_partitions = 1U;
  cfg.enable_hnsw = false;
  nbase_pdb_t *pdb = NULL;
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  /* Spans 3 partitions across 6 vectors; max_active_partitions=1 exercises LRU eviction. */
  uint64_t ids[6] = {101ULL, 102ULL, 201ULL, 202ULL, 301ULL, 302ULL};
  float vecs[18] = {
      1.0f, 0.00f, 0.0f,
      1.0f, 0.05f, 0.0f,
      1.0f, 0.10f, 0.0f,
      1.0f, 0.12f, 0.0f,
      1.0f, 0.15f, 0.0f,
      1.0f, 0.18f, 0.0f};
  ASSERT_STATUS(nbase_pdb_add_batch(pdb, ids, vecs, 6U, 3U, 1U), NBASE_OK);

  nbase_meta_field_t f_keep[2] = {
      {"group", {.type = NBASE_META_STRING, .as = {.str = "keep"}}},
      {"rank", {.type = NBASE_META_INT64, .as = {.i64 = 1}}}};
  nbase_meta_field_t f_drop[2] = {
      {"group", {.type = NBASE_META_STRING, .as = {.str = "drop"}}},
      {"rank", {.type = NBASE_META_INT64, .as = {.i64 = 9}}}};

  /* Atomicity check on missing ID across partitions. */
  nbase_meta_batch_item_t bad_items[2] = {
      {101ULL, f_keep, 2U},
      {9999ULL, f_keep, 2U}};
  ASSERT_STATUS(nbase_pdb_set_meta_batch(pdb, bad_items, 2U), NBASE_ERR_NOT_FOUND);
  nbase_meta_value_t mv;
  char sbuf[32];
  ASSERT_STATUS(nbase_pdb_get_meta(pdb, 101ULL, "group", &mv, sbuf, sizeof(sbuf)), NBASE_ERR_NOT_FOUND);

  /* Interleaved IDs across partitions, including duplicate overwrite of 101 -> drop -> keep. */
  nbase_meta_batch_item_t set_items[7] = {
      {301ULL, f_keep, 2U},
      {101ULL, f_drop, 2U},
      {201ULL, f_keep, 2U},
      {102ULL, f_drop, 2U},
      {202ULL, f_drop, 2U},
      {302ULL, f_keep, 2U},
      {101ULL, f_keep, 2U}};
  ASSERT_STATUS(nbase_pdb_set_meta_batch(pdb, set_items, 7U), NBASE_OK);

  nbase_meta_field_t f_upd = {"rank", {.type = NBASE_META_INT64, .as = {.i64 = 77}}};
  nbase_meta_batch_item_t upd_items[2] = {
      {101ULL, &f_upd, 1U},
      {301ULL, &f_upd, 1U}};
  ASSERT_STATUS(nbase_pdb_update_meta_batch(pdb, upd_items, 2U), NBASE_OK);

  ASSERT_STATUS(nbase_pdb_get_meta(pdb, 101ULL, "group", &mv, sbuf, sizeof(sbuf)), NBASE_OK);
  ASSERT_TRUE(strcmp(mv.as.str, "keep") == 0);
  ASSERT_STATUS(nbase_pdb_get_meta(pdb, 101ULL, "rank", &mv, NULL, 0U), NBASE_OK);
  ASSERT_EQ_U64((uint64_t)mv.as.i64, 77ULL);

  const char *texts[6] = {
      "alpha shard partition zero",
      "bravo shard zero",
      "alpha shard partition one",
      "delta noise one",
      "alpha shard partition two",
      "foxtrot noise two"};
  ASSERT_STATUS(nbase_pdb_set_text_batch(pdb, ids, texts, 6U), NBASE_OK);

  /* Save/reopen persistence check. */
  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);
  pdb = NULL;
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  nbase_meta_cond_t cond = {"group", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "keep"}}};
  nbase_meta_filter_t flt = {&cond, 1U};
  nbase_pdb_search_opts_t sopts = nbase_pdb_search_opts_default();
  sopts.search_all_partitions = true;
  sopts.search.meta_filter = &flt;

  /* Verify filtered graph extraction connects ONLY keep nodes {101, 201, 301, 302}. */
  nbase_edge_t *edges = NULL;
  uint64_t edge_cnt = 0ULL;
  ASSERT_STATUS(nbase_pdb_extract_relationships(pdb, 3U, 0.5f, &sopts, &edges, &edge_cnt), NBASE_OK);
  ASSERT_TRUE(edge_cnt > 0ULL && edges != NULL);
  for (uint64_t e = 0ULL; e < edge_cnt; e++) {
    ASSERT_TRUE(edges[e].src_id != 102ULL && edges[e].dst_id != 102ULL);
    ASSERT_TRUE(edges[e].src_id != 202ULL && edges[e].dst_id != 202ULL);
  }
  nbase_free_edges(edges);

  nbase_scored_id_t thits[6];
  uint32_t tcnt = 0U;
  ASSERT_STATUS(nbase_pdb_search_text(pdb, "alpha shard", 6U, &sopts, thits, &tcnt), NBASE_OK);
  ASSERT_EQ_U64(tcnt, 3ULL);

  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);
  test_rm_rf(dir);
}

static void test_pdb_count_filtered_and_unfiltered(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_pdb_config_t cfg = nbase_pdb_config_default(3U);
  cfg.partition_capacity = 2U;
  cfg.max_active_partitions = 1U;
  cfg.enable_hnsw = false;
  nbase_pdb_t *pdb = NULL;
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  uint64_t cnt = 0ULL;
  ASSERT_STATUS(nbase_pdb_count(pdb, NULL, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 0ULL);

  uint64_t ids[6] = {1ULL, 2ULL, 3ULL, 4ULL, 5ULL, 6ULL};
  float vecs[18];
  test_fill_vectors(vecs, 6U, 3U, 888ULL);
  ASSERT_STATUS(nbase_pdb_add_batch(pdb, ids, vecs, 6U, 3U, 1U), NBASE_OK);

  nbase_meta_field_t f_keep = {"tier", {.type = NBASE_META_INT64, .as = {.i64 = 1}}};
  nbase_meta_field_t f_skip = {"tier", {.type = NBASE_META_INT64, .as = {.i64 = 2}}};
  nbase_meta_batch_item_t items[6] = {
      {1ULL, &f_keep, 1U}, {2ULL, &f_skip, 1U},
      {3ULL, &f_keep, 1U}, {4ULL, &f_skip, 1U},
      {5ULL, &f_keep, 1U}, {6ULL, &f_skip, 1U}};
  ASSERT_STATUS(nbase_pdb_set_meta_batch(pdb, items, 6U), NBASE_OK);

  /* O(1) unfiltered count does not trigger partition reload (cache_misses unchanged). */
  nbase_pdb_stats_t s0, s1;
  ASSERT_STATUS(nbase_pdb_stats(pdb, &s0), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_count(pdb, NULL, &cnt), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_stats(pdb, &s1), NBASE_OK);
  ASSERT_EQ_U64(cnt, 6ULL);
  ASSERT_EQ_U64(s1.cache_misses, s0.cache_misses);

  nbase_meta_cond_t cond = {"tier", NBASE_OP_EQ, {.type = NBASE_META_INT64, .as = {.i64 = 1}}};
  nbase_meta_filter_t flt = {&cond, 1U};
  ASSERT_STATUS(nbase_pdb_count(pdb, &flt, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 3ULL);

  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);
  test_rm_rf(dir);
}

void test_partitioned_run(void) {
  RUN_TEST(test_pdb_durable_eviction_and_partition_ids);
  RUN_TEST(test_pdb_batch_text_hybrid_and_cache_stats);
  RUN_TEST(test_pdb_delete_batch_filter_search_batch_and_graph);
  RUN_TEST(test_pdb_upsert_batch_get_batch_and_rerank);
  RUN_TEST(test_pdb_list_ids_and_search_radius);
  RUN_TEST(test_pdb_meta_text_batch_and_filtered_relationships);
  RUN_TEST(test_pdb_count_filtered_and_unfiltered);
}
