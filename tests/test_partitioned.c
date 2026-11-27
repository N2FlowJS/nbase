#include "test_common.h"

#include <pthread.h>
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

static void test_pdb_batch_delete_meta_text_and_search_text_hybrid_batch(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  const uint32_t dim = 3U;
  nbase_pdb_config_t cfg = nbase_pdb_config_default(dim);
  cfg.partition_capacity = 3U;
  cfg.max_active_partitions = 2U;
  cfg.enable_hnsw = false;
  nbase_pdb_t *pdb = NULL;
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  uint64_t ids[6] = {10ULL, 20ULL, 30ULL, 40ULL, 50ULL, 60ULL};
  float vecs[18] = {
      1.0f, 0.0f, 0.0f,
      0.9f, 0.1f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.9f, 0.1f,
      0.0f, 0.0f, 1.0f,
      0.1f, 0.0f, 0.9f};
  ASSERT_STATUS(nbase_pdb_add_batch(pdb, ids, vecs, 6U, dim, 2U), NBASE_OK);

  nbase_meta_field_t fa = {"grp", {.type = NBASE_META_STRING, .as = {.str = "a"}}};
  nbase_meta_field_t fb = {"grp", {.type = NBASE_META_STRING, .as = {.str = "b"}}};
  nbase_meta_batch_item_t mitems[6] = {
      {10ULL, &fa, 1U}, {20ULL, &fb, 1U}, {30ULL, &fa, 1U},
      {40ULL, &fb, 1U}, {50ULL, &fa, 1U}, {60ULL, &fb, 1U}};
  ASSERT_STATUS(nbase_pdb_set_meta_batch(pdb, mitems, 6U), NBASE_OK);

  const char *texts[6] = {
      "simd hnsw kernel",
      "simd quantization sq8",
      "distributed raft storage",
      "raft wal durability",
      "hybrid bm25 rrf fusion",
      "bm25 sparse ranking"};
  ASSERT_STATUS(nbase_pdb_set_text_batch(pdb, ids, texts, 6U), NBASE_OK);

  const char *tqs[3] = {"simd kernel", "raft durability", "bm25 fusion"};
  nbase_pdb_search_opts_t popts = nbase_pdb_search_opts_default();
  popts.search_all_partitions = true;
  popts.search.num_threads = 2U;

  nbase_scored_id_t tb_res[6];
  uint32_t tb_cnts[3] = {0U, 0U, 0U};
  ASSERT_STATUS(nbase_pdb_search_text_batch(pdb, NULL, 0U, 2U, &popts, NULL, NULL), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_search_text_batch(pdb, tqs, 3U, 2U, &popts, tb_res, tb_cnts), NBASE_OK);
  for (uint32_t q = 0U; q < 3U; q++) {
    nbase_scored_id_t single[2];
    uint32_t sc = 0U;
    ASSERT_STATUS(nbase_pdb_search_text(pdb, tqs[q], 2U, &popts, single, &sc), NBASE_OK);
    ASSERT_EQ_U64(tb_cnts[q], sc);
    for (uint32_t j = 0U; j < sc; j++) {
      ASSERT_EQ_U64(tb_res[(size_t)q * 2U + j].id, single[j].id);
      ASSERT_NEAR(tb_res[(size_t)q * 2U + j].score, single[j].score, 1e-6, 1e-6);
    }
  }

  float hq_vecs[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  nbase_scored_id_t hb_res[6];
  uint32_t hb_cnts[3] = {0U, 0U, 0U};
  ASSERT_STATUS(
      nbase_pdb_search_hybrid_batch(pdb, NULL, 0U, NULL, 0U, 2U, 0.5f, 60.0f, &popts, NULL, NULL),
      NBASE_OK);
  ASSERT_STATUS(
      nbase_pdb_search_hybrid_batch(pdb, hq_vecs, dim, tqs, 3U, 2U, 0.5f, 60.0f, &popts, hb_res, hb_cnts),
      NBASE_OK);
  for (uint32_t q = 0U; q < 3U; q++) {
    nbase_scored_id_t single[2];
    uint32_t sc = 0U;
    ASSERT_STATUS(
        nbase_pdb_search_hybrid(pdb, hq_vecs + (size_t)q * dim, dim, tqs[q], 2U, 0.5f, 60.0f, &popts, single, &sc),
        NBASE_OK);
    ASSERT_EQ_U64(hb_cnts[q], sc);
    for (uint32_t j = 0U; j < sc; j++) {
      ASSERT_EQ_U64(hb_res[(size_t)q * 2U + j].id, single[j].id);
      ASSERT_NEAR(hb_res[(size_t)q * 2U + j].score, single[j].score, 1e-6, 1e-6);
    }
  }
  ASSERT_STATUS(
      nbase_pdb_search_hybrid_batch(pdb, hq_vecs, dim + 1U, tqs, 3U, 2U, 0.5f, 60.0f, &popts, hb_res, hb_cnts),
      NBASE_ERR_DIM_MISMATCH);

  /* Batch delete meta & batch delete text across evicted/active partitions */
  uint64_t del_mids[3] = {10ULL, 999ULL, 50ULL};
  uint32_t del_m = 0U;
  ASSERT_STATUS(nbase_pdb_delete_meta_batch(pdb, del_mids, 3U, &del_m), NBASE_OK);
  ASSERT_EQ_U64(del_m, 2ULL);

  nbase_meta_cond_t ca = {"grp", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "a"}}};
  nbase_meta_filter_t fa_flt = {&ca, 1U};
  uint64_t grp_a_cnt = 0ULL;
  ASSERT_STATUS(nbase_pdb_count(pdb, &fa_flt, &grp_a_cnt), NBASE_OK);
  ASSERT_EQ_U64(grp_a_cnt, 1ULL); /* only id 30 remains in grp=a */

  uint64_t del_tids[3] = {20ULL, 60ULL, 999ULL};
  uint32_t del_t = 0U;
  ASSERT_STATUS(nbase_pdb_delete_text_batch(pdb, del_tids, 3U, &del_t), NBASE_OK);
  ASSERT_EQ_U64(del_t, 2ULL);

  ASSERT_STATUS(nbase_pdb_save(pdb), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);

  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);
  size_t len20 = 0U;
  ASSERT_STATUS(nbase_pdb_get_text(pdb, 20ULL, NULL, 0U, &len20), NBASE_ERR_NOT_FOUND);
  ASSERT_STATUS(nbase_pdb_get_text(pdb, 10ULL, NULL, 0U, &len20), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_count(pdb, &fa_flt, &grp_a_cnt), NBASE_OK);
  ASSERT_EQ_U64(grp_a_cnt, 1ULL);
  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);
  test_rm_rf(dir);
}

typedef struct pdb_text_batch_worker_ctx {
  nbase_pdb_t *pdb;
} pdb_text_batch_worker_ctx_t;

static void *pdb_text_batch_reader_worker(void *raw) {
  pdb_text_batch_worker_ctx_t *ctx = (pdb_text_batch_worker_ctx_t *)raw;
  static const uint64_t qids[8] = {50ULL, 20ULL, 10ULL, 999ULL, 50ULL, 40ULL, 30ULL, 60ULL};
  for (uint32_t it = 0U; it < 60U; it++) {
    char *texts[8] = {NULL};
    bool found[8] = {false};
    if (nbase_pdb_get_text_batch(ctx->pdb, qids, 8U, texts, found) != NBASE_OK) {
      return (void *)(uintptr_t)1U;
    }
    bool ok = !found[0] && texts[0] == NULL &&
              found[1] && texts[1] != NULL && strcmp(texts[1], "part text twenty") == 0 &&
              found[2] && texts[2] != NULL && strcmp(texts[2], "part text ten") == 0 &&
              !found[3] && texts[3] == NULL &&
              !found[4] && texts[4] == NULL &&
              found[5] && texts[5] != NULL && strcmp(texts[5], "part text forty") == 0 &&
              !found[6] && texts[6] == NULL &&
              found[7] && texts[7] != NULL && strcmp(texts[7], "part text sixty") == 0;
    nbase_free_texts(texts, 8U);
    if (!ok) {
      return (void *)(uintptr_t)1U;
    }
  }
  return NULL;
}

static void test_pdb_get_text_batch_multi_partition_and_eviction(void) {
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  nbase_pdb_config_t cfg = nbase_pdb_config_default(4U);
  cfg.partition_capacity = 2U;
  cfg.max_active_partitions = 1U;
  cfg.seed = 4317ULL;

  nbase_pdb_t *pdb = NULL;
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  float vecs[6 * 4];
  test_fill_vectors(vecs, 6U, 4U, 917ULL);
  uint64_t ids[6] = {10ULL, 20ULL, 30ULL, 40ULL, 50ULL, 60ULL};
  ASSERT_STATUS(nbase_pdb_add_batch(pdb, ids, vecs, 6U, 4U, 2U), NBASE_OK);

  uint64_t tids[4] = {10ULL, 20ULL, 40ULL, 60ULL};
  const char *tvals[4] = {
      "part text ten",
      "part text twenty",
      "part text forty",
      "part text sixty"};
  ASSERT_STATUS(nbase_pdb_set_text_batch(pdb, tids, tvals, 4U), NBASE_OK);

  char *bad_texts[2] = {NULL, NULL};
  bool bad_found[2] = {false, false};
  ASSERT_STATUS(nbase_pdb_get_text_batch(NULL, tids, 2U, bad_texts, bad_found), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_pdb_get_text_batch(pdb, NULL, 2U, bad_texts, bad_found), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_pdb_get_text_batch(pdb, tids, 2U, NULL, bad_found), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_pdb_get_text_batch(pdb, tids, 2U, bad_texts, NULL), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_pdb_get_text_batch(pdb, NULL, 0U, NULL, NULL), NBASE_OK);

  ASSERT_STATUS(nbase_pdb_save(pdb), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  uint64_t qids[8] = {50ULL, 20ULL, 10ULL, 999ULL, 50ULL, 40ULL, 30ULL, 60ULL};
  char *out_texts[8] = {NULL};
  bool out_found[8] = {false};
  ASSERT_STATUS(nbase_pdb_get_text_batch(pdb, qids, 8U, out_texts, out_found), NBASE_OK);
  ASSERT_TRUE(!out_found[0] && out_texts[0] == NULL);
  ASSERT_TRUE(out_found[1] && out_texts[1] != NULL && strcmp(out_texts[1], "part text twenty") == 0);
  ASSERT_TRUE(out_found[2] && out_texts[2] != NULL && strcmp(out_texts[2], "part text ten") == 0);
  ASSERT_TRUE(!out_found[3] && out_texts[3] == NULL);
  ASSERT_TRUE(!out_found[4] && out_texts[4] == NULL);
  ASSERT_TRUE(out_found[5] && out_texts[5] != NULL && strcmp(out_texts[5], "part text forty") == 0);
  ASSERT_TRUE(!out_found[6] && out_texts[6] == NULL);
  ASSERT_TRUE(out_found[7] && out_texts[7] != NULL && strcmp(out_texts[7], "part text sixty") == 0);
  nbase_free_texts(out_texts, 8U);

  pdb_text_batch_worker_ctx_t ctx = {pdb};
  pthread_t th[4];
  for (uint32_t t = 0U; t < 4U; t++) {
    ASSERT_TRUE(pthread_create(&th[t], NULL, pdb_text_batch_reader_worker, &ctx) == 0);
  }
  for (uint32_t t = 0U; t < 4U; t++) {
    void *ret = NULL;
    ASSERT_TRUE(pthread_join(th[t], &ret) == 0);
    ASSERT_TRUE(ret == NULL);
  }

  ASSERT_STATUS(nbase_pdb_close(pdb), NBASE_OK);
  test_rm_rf(dir);
}

typedef struct pdb_similar_batch_ctx {
  nbase_pdb_t *pdb;
  const uint64_t *ids;
  uint32_t qcnt;
  uint32_t k;
  uint32_t num_threads;
} pdb_similar_batch_ctx_t;

static void *pdb_find_similar_batch_worker(void *arg) {
  const pdb_similar_batch_ctx_t *ctx = (const pdb_similar_batch_ctx_t *)arg;
  nbase_pdb_search_opts_t opts = nbase_pdb_search_opts_default();
  opts.search_all_partitions = true;
  opts.search.num_threads = ctx->num_threads;
  nbase_result_t res[16];
  uint32_t cnts[4] = {0U, 0U, 0U, 0U};
  for (int it = 0; it < 4; it++) {
    if (nbase_pdb_find_similar_batch(ctx->pdb, ctx->ids, ctx->qcnt, ctx->k, &opts, res, cnts) !=
        NBASE_OK) {
      return (void *)1;
    }
    for (uint32_t q = 0U; q < ctx->qcnt; q++) {
      if (cnts[q] != ctx->k) {
        return (void *)1;
      }
      for (uint32_t j = 0U; j < cnts[q]; j++) {
        if (res[(size_t)q * (size_t)ctx->k + (size_t)j].id == ctx->ids[q]) {
          return (void *)1;
        }
      }
    }
  }
  return NULL;
}

static void test_pdb_find_similar_batch_multi_partition_and_tsan(void) {
  char dir[128];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));

  const uint32_t dim = 8U;
  const uint32_t nvecs = 18U;
  nbase_pdb_config_t cfg = nbase_pdb_config_default(dim);
  cfg.partition_capacity = 6U;
  cfg.max_active_partitions = 2U;
  cfg.seed = 4321ULL;

  nbase_pdb_t *pdb = NULL;
  ASSERT_STATUS(nbase_pdb_open(dir, &cfg, &pdb), NBASE_OK);

  float vecs[18U * 8U];
  test_fill_vectors(vecs, nvecs, dim, 5150ULL);
  for (uint32_t i = 0U; i < nvecs; i++) {
    uint64_t id = (uint64_t)i * 10ULL;
    ASSERT_STATUS(nbase_pdb_add(pdb, id, vecs + (size_t)i * (size_t)dim, dim), NBASE_OK);
    nbase_meta_field_t mf = {"even", {.type = NBASE_META_BOOL, .as = {.b = ((i % 2U) == 0U)}}};
    ASSERT_STATUS(nbase_pdb_set_meta(pdb, id, &mf, 1U), NBASE_OK);
  }

  nbase_pdb_stats_t pst;
  ASSERT_STATUS(nbase_pdb_stats(pdb, &pst), NBASE_OK);
  ASSERT_TRUE(pst.total_partitions >= 3ULL);
  ASSERT_TRUE(pst.eviction_count >= 1ULL);

  const uint64_t qids[4] = {0ULL, 70ULL, 130ULL, 160ULL};
  const uint32_t qcnt = 4U;
  const uint32_t k = 4U;
  nbase_result_t batch_res[4U * 4U];
  nbase_result_t single_res[4U];
  uint32_t bcnts[4] = {0};

  const uint64_t bad_qids[2] = {0ULL, 99999ULL};
  ASSERT_STATUS(nbase_pdb_find_similar_batch(pdb, bad_qids, 2U, k, NULL, batch_res, bcnts), NBASE_ERR_NOT_FOUND);
  ASSERT_STATUS(nbase_pdb_find_similar_batch(pdb, NULL, 0U, k, NULL, NULL, NULL), NBASE_OK);
  uint32_t zcnt[2] = {7U, 7U};
  ASSERT_STATUS(nbase_pdb_find_similar_batch(pdb, qids, 2U, 0U, NULL, NULL, zcnt), NBASE_OK);
  ASSERT_EQ_U64(zcnt[0], 0ULL);
  ASSERT_EQ_U64(zcnt[1], 0ULL);

  /* Filtered + metric override across resident and evicted partitions */
  nbase_meta_cond_t cond = {"even", NBASE_OP_EQ, {.type = NBASE_META_BOOL, .as = {.b = true}}};
  nbase_meta_filter_t flt = {&cond, 1U};
  nbase_pdb_search_opts_t popts = nbase_pdb_search_opts_default();
  popts.search_all_partitions = true;
  popts.search.meta_filter = &flt;
  popts.search.metric = NBASE_METRIC_COSINE;
  popts.search.use_metric_override = true;
  popts.search.num_threads = 4U;

  ASSERT_STATUS(nbase_pdb_find_similar_batch(pdb, qids, qcnt, k, &popts, batch_res, bcnts), NBASE_OK);
  for (uint32_t q = 0U; q < qcnt; q++) {
    uint32_t scnt = 0U;
    ASSERT_STATUS(nbase_pdb_find_similar(pdb, qids[q], k, &popts, single_res, &scnt), NBASE_OK);
    ASSERT_EQ_U64(bcnts[q], scnt);
    for (uint32_t j = 0U; j < scnt; j++) {
      size_t idx = (size_t)q * (size_t)k + (size_t)j;
      ASSERT_TRUE(batch_res[idx].id != qids[q]);
      ASSERT_EQ_U64(batch_res[idx].id, single_res[j].id);
      ASSERT_NEAR(batch_res[idx].distance, single_res[j].distance, 1e-6, 1e-6);
      ASSERT_EQ_U64((batch_res[idx].id / 10ULL) % 2ULL, 0ULL);
    }
  }

  /* TSan concurrent multi-partition find_similar_batch reads */
  pdb_similar_batch_ctx_t wctx = {pdb, qids, qcnt, 4U, 2U};
  pthread_t th[3];
  for (int t = 0; t < 3; t++) {
    ASSERT_TRUE(pthread_create(&th[t], NULL, pdb_find_similar_batch_worker, &wctx) == 0);
  }
  for (int t = 0; t < 3; t++) {
    void *ret = NULL;
    ASSERT_TRUE(pthread_join(th[t], &ret) == 0);
    ASSERT_TRUE(ret == NULL);
  }

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
  RUN_TEST(test_pdb_batch_delete_meta_text_and_search_text_hybrid_batch);
  RUN_TEST(test_pdb_get_text_batch_multi_partition_and_eviction);
  RUN_TEST(test_pdb_find_similar_batch_multi_partition_and_tsan);
}
