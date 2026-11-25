#include "test_common.h"

#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

static void test_mmr_diversity_vs_relevance_lambda(void) {
  const uint32_t dim = 3U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  /* Cluster A near (1, 0, 0), distinct candidate 4 near (0.7, 0.71, 0). */
  float v1[3] = {1.00f, 0.00f, 0.00f};
  float v2[3] = {0.99f, 0.01f, 0.00f};
  float v3[3] = {0.98f, 0.02f, 0.00f};
  float v4[3] = {0.70f, 0.71f, 0.00f};
  float v5[3] = {0.00f, 1.00f, 0.00f};

  ASSERT_STATUS(nbase_add(db, 1ULL, v1, dim), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 2ULL, v2, dim), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 3ULL, v3, dim), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 4ULL, v4, dim), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 5ULL, v5, dim), NBASE_OK);

  nbase_result_t cands[5];
  uint32_t cand_n = 0U;
  ASSERT_STATUS(nbase_search(db, v1, dim, 5U, NULL, cands, &cand_n), NBASE_OK);
  ASSERT_EQ_U64(cand_n, 5ULL);

  /* Pure relevance (lambda=1.0) keeps near-duplicates 1, 2, 3. */
  nbase_result_t rel_out[3];
  uint32_t out_n = 0U;
  ASSERT_STATUS(nbase_rerank_mmr(db, cands, cand_n, 3U, 1.0f, rel_out, &out_n), NBASE_OK);
  ASSERT_EQ_U64(out_n, 3ULL);
  ASSERT_EQ_U64(rel_out[0].id, 1ULL);
  ASSERT_EQ_U64(rel_out[1].id, 2ULL);
  ASSERT_EQ_U64(rel_out[2].id, 3ULL);

  /* Pure diversity (lambda=0.0) picks top seed 1, then orthogonal 5 next. */
  nbase_result_t div_out[3];
  ASSERT_STATUS(nbase_rerank_mmr(db, cands, cand_n, 3U, 0.0f, div_out, &out_n), NBASE_OK);
  ASSERT_EQ_U64(out_n, 3ULL);
  ASSERT_EQ_U64(div_out[0].id, 1ULL);
  ASSERT_EQ_U64(div_out[1].id, 5ULL);

  /* Balanced lambda=0.4 promotes distinct candidate 4/5 ahead of duplicate 2/3. */
  nbase_result_t bal_out[2];
  ASSERT_STATUS(nbase_rerank_mmr(db, cands, cand_n, 2U, 0.4f, bal_out, &out_n), NBASE_OK);
  ASSERT_EQ_U64(out_n, 2ULL);
  ASSERT_EQ_U64(bal_out[0].id, 1ULL);
  ASSERT_TRUE(bal_out[1].id == 4ULL || bal_out[1].id == 5ULL);

  ASSERT_STATUS(nbase_rerank_mmr(db, cands, cand_n, 2U, 1.5f, bal_out, &out_n), NBASE_ERR_INVALID_ARG);
  nbase_close(db);
}

static void test_extract_relationships_and_communities(void) {
  const uint32_t dim = 3U;
  nbase_config_t cfg = nbase_config_default(dim);
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  /* Three separated tight clusters along X, Y, Z axes. */
  uint64_t ids[12];
  for (uint32_t c = 0U; c < 3U; c++) {
    for (uint32_t i = 0U; i < 4U; i++) {
      uint32_t idx = c * 4U + i;
      ids[idx] = 100ULL + idx;
      float v[3] = {0.0f, 0.0f, 0.0f};
      v[c] = 10.0f;
      v[(c + 1U) % 3U] = 0.02f * (float)i;
      ASSERT_STATUS(nbase_add(db, ids[idx], v, dim), NBASE_OK);
    }
  }

  nbase_edge_t *edges = NULL;
  uint64_t edge_cnt = 0ULL;
  ASSERT_STATUS(nbase_extract_relationships(db, 3U, 0.5f, &edges, &edge_cnt), NBASE_OK);
  ASSERT_TRUE(edge_cnt > 0ULL && edges != NULL);
  for (uint64_t e = 0ULL; e < edge_cnt; e++) {
    ASSERT_TRUE(edges[e].src_id < edges[e].dst_id);
    ASSERT_TRUE(edges[e].weight <= 0.5f);
  }

  uint32_t labels[12];
  uint32_t comm_cnt = 0U;
  ASSERT_STATUS(nbase_extract_communities(ids, 12U, edges, edge_cnt, labels, &comm_cnt), NBASE_OK);
  ASSERT_EQ_U64(comm_cnt, 3ULL);
  for (uint32_t c = 0U; c < 3U; c++) {
    uint32_t base_lbl = labels[c * 4U];
    for (uint32_t i = 1U; i < 4U; i++) {
      ASSERT_EQ_U64(labels[c * 4U + i], base_lbl);
    }
  }
  ASSERT_TRUE(labels[0] != labels[4]);
  ASSERT_TRUE(labels[0] != labels[8]);
  ASSERT_TRUE(labels[4] != labels[8]);

  nbase_free_edges(edges);
  nbase_close(db);
}

static void test_communities_deep_chain_iterative(void) {
  const uint32_t n = 15000U;
  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  nbase_edge_t *edges = malloc((size_t)(n - 1U) * sizeof(nbase_edge_t));
  uint32_t *labels = malloc((size_t)n * sizeof(uint32_t));
  ASSERT_TRUE(ids != NULL && edges != NULL && labels != NULL);

  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = (uint64_t)i * 7ULL + 3ULL;
  }
  for (uint32_t i = 0U; i < n - 1U; i++) {
    edges[i].src_id = ids[i];
    edges[i].dst_id = ids[i + 1U];
    edges[i].weight = 1.0f;
  }

  uint32_t comm_cnt = 0U;
  ASSERT_STATUS(nbase_extract_communities(ids, n, edges, n - 1U, labels, &comm_cnt), NBASE_OK);
  ASSERT_EQ_U64(comm_cnt, 1ULL);
  ASSERT_EQ_U64(labels[0], 0ULL);
  ASSERT_EQ_U64(labels[n - 1U], 0ULL);

  free(ids);
  free(edges);
  free(labels);
}

static void test_rerank_weighted_metadata_and_edge_cases(void) {
  const uint32_t dim = 3U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.enable_hnsw = false;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  float v1[3] = {1.0f, 0.0f, 0.0f};
  float v2[3] = {0.8f, 0.6f, 0.0f};
  float v3[3] = {0.0f, 1.0f, 0.0f};
  ASSERT_STATUS(nbase_add(db, 1ULL, v1, dim), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 2ULL, v2, dim), NBASE_OK);
  ASSERT_STATUS(nbase_add(db, 3ULL, v3, dim), NBASE_OK);

  /* Id 1: pop=1, boost=0.1; Id 2: pop=5, boost=2.0; Id 3: no numeric meta */
  nbase_meta_field_t m1[2] = {
      {"pop", {.type = NBASE_META_INT64, .as = {.i64 = 1LL}}},
      {"boost", {.type = NBASE_META_DOUBLE, .as = {.f64 = 0.1}}},
  };
  nbase_meta_field_t m2[2] = {
      {"pop", {.type = NBASE_META_INT64, .as = {.i64 = 5LL}}},
      {"boost", {.type = NBASE_META_DOUBLE, .as = {.f64 = 2.0}}},
  };
  ASSERT_STATUS(nbase_set_meta(db, 1ULL, m1, 2U), NBASE_OK);
  ASSERT_STATUS(nbase_set_meta(db, 2ULL, m2, 2U), NBASE_OK);

  nbase_result_t cands[3];
  uint32_t cand_n = 0U;
  ASSERT_STATUS(nbase_search(db, v1, dim, 3U, NULL, cands, &cand_n), NBASE_OK);
  ASSERT_EQ_U64(cand_n, 3ULL);
  ASSERT_EQ_U64(cands[0].id, 1ULL);

  nbase_rerank_weight_t weights[2] = {
      {"pop", 0.2f},
      {"boost", 0.5f},
  };
  nbase_result_t out[2];
  uint32_t out_n = 0U;
  ASSERT_STATUS(nbase_rerank_weighted(db, cands, cand_n, weights, 2U, 2U, out, &out_n), NBASE_OK);
  ASSERT_EQ_U64(out_n, 2ULL);
  /* Id 2 gets subtraction 5*0.2 + 2.0*0.5 = 2.0 -> moves to #1 */
  ASSERT_EQ_U64(out[0].id, 2ULL);
  ASSERT_EQ_U64(out[1].id, 1ULL);

  nbase_rerank_weight_t bad_w = {"", 1.0f};
  ASSERT_STATUS(nbase_rerank_weighted(db, cands, cand_n, &bad_w, 1U, 2U, out, &out_n), NBASE_ERR_INVALID_ARG);
  nbase_result_t missing_c = {999ULL, 0.1f};
  ASSERT_STATUS(nbase_rerank_weighted(db, &missing_c, 1U, weights, 2U, 1U, out, &out_n), NBASE_ERR_NOT_FOUND);

  nbase_close(db);
}

static void test_extract_relationships_filtered(void) {
  /* Exercises both exact mode (<=64 matching nodes) and HNSW mode (>64 matching nodes). */
  const uint32_t dim = 4U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.enable_hnsw = true;
  cfg.hnsw_m = 16U;
  cfg.hnsw_ef_construction = 80U;
  cfg.hnsw_ef_search = 64U;
  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  /* Insert 80 vectors: 70 keep_large ("big"), 6 keep_small ("small"), 4 ignored ("none"). */
  for (uint32_t i = 0U; i < 80U; i++) {
    uint64_t id = (uint64_t)i + 1ULL;
    float v[4] = {(float)(i % 10U) * 0.05f, 0.0f, 0.0f, 0.0f};
    const char *cat = "big";
    if (i >= 70U && i < 76U) {
      v[1] = 0.01f * (float)(i - 70U);
      cat = "small";
    } else if (i >= 76U) {
      v[2] = 0.01f * (float)(i - 76U);
      cat = "none";
    }
    ASSERT_STATUS(nbase_add(db, id, v, dim), NBASE_OK);
    nbase_meta_field_t fld = {"cat", {.type = NBASE_META_STRING, .as = {.str = cat}}};
    ASSERT_STATUS(nbase_set_meta(db, id, &fld, 1U), NBASE_OK);
  }

  /* Exact filtered path (6 nodes -> ids 71..76). */
  nbase_meta_cond_t cond_s = {"cat", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "small"}}};
  nbase_meta_filter_t flt_s = {&cond_s, 1U};
  nbase_edge_t *edges = NULL;
  uint64_t edge_cnt = 0ULL;
  ASSERT_STATUS(nbase_extract_relationships_filtered(db, 3U, 2.0f, &flt_s, &edges, &edge_cnt), NBASE_OK);
  ASSERT_TRUE(edge_cnt > 0ULL && edges != NULL);
  for (uint64_t e = 0ULL; e < edge_cnt; e++) {
    ASSERT_TRUE(edges[e].src_id >= 71ULL && edges[e].src_id <= 76ULL);
    ASSERT_TRUE(edges[e].dst_id >= 71ULL && edges[e].dst_id <= 76ULL);
  }
  nbase_free_edges(edges);
  edges = NULL;

  /* HNSW filtered path (70 nodes -> ids 1..70). */
  nbase_meta_cond_t cond_b = {"cat", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "big"}}};
  nbase_meta_filter_t flt_b = {&cond_b, 1U};
  ASSERT_STATUS(nbase_extract_relationships_filtered(db, 4U, 2.0f, &flt_b, &edges, &edge_cnt), NBASE_OK);
  ASSERT_TRUE(edge_cnt > 0ULL && edges != NULL);
  for (uint64_t e = 0ULL; e < edge_cnt; e++) {
    ASSERT_TRUE(edges[e].src_id >= 1ULL && edges[e].src_id <= 70ULL);
    ASSERT_TRUE(edges[e].dst_id >= 1ULL && edges[e].dst_id <= 70ULL);
  }
  nbase_free_edges(edges);
  edges = NULL;

  /* Zero-match filter returns 0 edges cleanly. */
  nbase_meta_cond_t cond_z = {"cat", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "missing"}}};
  nbase_meta_filter_t flt_z = {&cond_z, 1U};
  ASSERT_STATUS(nbase_extract_relationships_filtered(db, 3U, 2.0f, &flt_z, &edges, &edge_cnt), NBASE_OK);
  ASSERT_EQ_U64(edge_cnt, 0ULL);
  ASSERT_TRUE(edges == NULL);

  /* Invalid filter struct must reject cleanly. */
  nbase_meta_filter_t bad_flt = {NULL, 1U};
  ASSERT_STATUS(nbase_extract_relationships_filtered(db, 3U, 2.0f, &bad_flt, &edges, &edge_cnt), NBASE_ERR_INVALID_ARG);

  nbase_close(db);
}

static void test_extract_relationships_no_hnsw_batch(void) {
  const uint32_t dim = 16U;
  const uint32_t n = 45U;
  nbase_config_t cfg = nbase_config_default(dim);
  cfg.metric = NBASE_METRIC_COSINE;
  cfg.enable_hnsw = false;

  nbase_db_t *db = NULL;
  ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

  uint64_t ids[45];
  for (uint32_t g = 0U; g < 3U; g++) {
    for (uint32_t i = 0U; i < 15U; i++) {
      uint32_t idx = g * 15U + i;
      ids[idx] = 500ULL + idx;
      float v[16] = {0.0f};
      v[g] = 1.0f;
      v[(g + 3U) % dim] = 0.01f * (float)i;
      ASSERT_STATUS(nbase_add(db, ids[idx], v, dim), NBASE_OK);
    }
  }

  nbase_edge_t *edges = NULL;
  uint64_t edge_cnt = 0ULL;
  ASSERT_STATUS(nbase_extract_relationships(db, 3U, 0.1f, &edges, &edge_cnt), NBASE_OK);
  ASSERT_TRUE(edge_cnt > 0ULL && edges != NULL);

  uint32_t labels[45];
  uint32_t comm_cnt = 0U;
  ASSERT_STATUS(nbase_extract_communities(ids, n, edges, edge_cnt, labels, &comm_cnt), NBASE_OK);
  ASSERT_EQ_U64(comm_cnt, 3ULL);
  nbase_free_edges(edges);
  nbase_close(db);
}

typedef struct rerank_tsan_ctx {
  nbase_db_t *db;
  const nbase_meta_filter_t *flt;
  atomic_int errors;
} rerank_tsan_ctx_t;

static void *rerank_parallel_reader_fn(void *raw) {
  rerank_tsan_ctx_t *ctx = (rerank_tsan_ctx_t *)raw;
  for (int iter = 0; iter < 3; iter++) {
    nbase_edge_t *e = NULL;
    uint64_t ec = 0ULL;
    if (nbase_extract_relationships_filtered(ctx->db, 4U, 2.0f, ctx->flt, &e, &ec) != NBASE_OK ||
        ec == 0ULL || !e) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      nbase_free_edges(e);
      break;
    }
    nbase_free_edges(e);
  }
  return NULL;
}

static void test_extract_relationships_parallel_hnsw_and_exact_tsan(void) {
  const uint32_t dim = 8U;
  const uint32_t n = 160U;

  for (int mode = 0; mode < 2; mode++) {
    nbase_config_t cfg = nbase_config_default(dim);
    cfg.metric = NBASE_METRIC_EUCLIDEAN;
    cfg.enable_hnsw = (mode == 0);

    nbase_db_t *db = NULL;
    ASSERT_STATUS(nbase_create(&cfg, &db), NBASE_OK);

    for (uint32_t i = 0U; i < n; i++) {
      uint64_t id = 1000ULL + i;
      float v[8] = {0.0f};
      v[i % 4U] = 1.0f;
      v[(i % 4U) + 4U] = 0.01f * (float)(i / 4U);
      ASSERT_STATUS(nbase_add(db, id, v, dim), NBASE_OK);

      const char *grp = (i < 132U) ? "big" : "small";
      nbase_meta_field_t mf = {"grp", {.type = NBASE_META_STRING, .as = {.str = grp}}};
      ASSERT_STATUS(nbase_set_meta(db, id, &mf, 1U), NBASE_OK);
    }

    /* Unfiltered parallel extraction (>= 128 active): deterministic across repeated calls. */
    nbase_edge_t *e1 = NULL;
    nbase_edge_t *e2 = NULL;
    uint64_t c1 = 0ULL;
    uint64_t c2 = 0ULL;
    ASSERT_STATUS(nbase_extract_relationships(db, 4U, 2.0f, &e1, &c1), NBASE_OK);
    ASSERT_STATUS(nbase_extract_relationships(db, 4U, 2.0f, &e2, &c2), NBASE_OK);
    ASSERT_TRUE(c1 > 0ULL && e1 != NULL);
    ASSERT_EQ_U64(c1, c2);
    for (uint64_t i = 0ULL; i < c1; i++) {
      ASSERT_TRUE(e1[i].src_id < e1[i].dst_id);
      ASSERT_EQ_U64(e1[i].src_id, e2[i].src_id);
      ASSERT_EQ_U64(e1[i].dst_id, e2[i].dst_id);
      ASSERT_NEAR(e1[i].weight, e2[i].weight, 1e-6, 0.0);
    }
    nbase_free_edges(e1);
    nbase_free_edges(e2);

    /* Filtered parallel extraction (active == 132 >= 128) + concurrent TSan callers. */
    nbase_meta_cond_t cond = {"grp", NBASE_OP_EQ, {.type = NBASE_META_STRING, .as = {.str = "big"}}};
    nbase_meta_filter_t flt = {&cond, 1U};
    nbase_edge_t *fe = NULL;
    uint64_t fc = 0ULL;
    ASSERT_STATUS(nbase_extract_relationships_filtered(db, 4U, 2.0f, &flt, &fe, &fc), NBASE_OK);
    ASSERT_TRUE(fc > 0ULL && fe != NULL);
    for (uint64_t i = 0ULL; i < fc; i++) {
      ASSERT_TRUE(fe[i].src_id >= 1000ULL && fe[i].src_id < 1132ULL);
      ASSERT_TRUE(fe[i].dst_id >= 1000ULL && fe[i].dst_id < 1132ULL);
    }
    nbase_free_edges(fe);

    rerank_tsan_ctx_t ctx = {db, &flt, 0};
    pthread_t t1, t2;
    ASSERT_TRUE(pthread_create(&t1, NULL, rerank_parallel_reader_fn, &ctx) == 0);
    ASSERT_TRUE(pthread_create(&t2, NULL, rerank_parallel_reader_fn, &ctx) == 0);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);

    nbase_close(db);
  }
}

void test_rerank_run(void) {
  RUN_TEST(test_mmr_diversity_vs_relevance_lambda);
  RUN_TEST(test_extract_relationships_and_communities);
  RUN_TEST(test_communities_deep_chain_iterative);
  RUN_TEST(test_rerank_weighted_metadata_and_edge_cases);
  RUN_TEST(test_extract_relationships_filtered);
  RUN_TEST(test_extract_relationships_no_hnsw_batch);
  RUN_TEST(test_extract_relationships_parallel_hnsw_and_exact_tsan);
}
