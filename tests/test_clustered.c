#include "test_common.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

static void test_clustered_online_assign_search_and_recluster(void) {
  nbase_clustered_config_t cfg = nbase_clustered_config_default(4U);
  cfg.target_cluster_size = 4U;
  cfg.max_clusters = 16U;
  cfg.distance_threshold = 0.35f;
  cfg.seed = 2025ULL;

  nbase_clustered_t *cl = NULL;
  ASSERT_STATUS(nbase_clustered_create(&cfg, &cl), NBASE_OK);

  /* Two separated clusters around (0,0,0,0) and (5,5,0,0) */
  float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float a1[4] = {0.1f, 0.0f, 0.0f, 0.0f};
  float a2[4] = {0.0f, 0.1f, 0.0f, 0.0f};
  float b0[4] = {5.0f, 5.0f, 0.0f, 0.0f};
  float b1[4] = {5.1f, 5.0f, 0.0f, 0.0f};
  float c0[4] = {10.0f, 0.0f, 0.0f, 0.0f};

  ASSERT_STATUS(nbase_clustered_add(cl, 1ULL, a0, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 2ULL, a1, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 3ULL, a2, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 4ULL, b0, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 5ULL, b1, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 6ULL, c0, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_add(cl, 1ULL, a0, 4U), NBASE_ERR_ALREADY_EXISTS);

  nbase_clustered_stats_t st;
  ASSERT_STATUS(nbase_clustered_stats(cl, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, 6ULL);
  ASSERT_EQ_U64(st.cluster_count, 3ULL);
  ASSERT_TRUE(st.memory_bytes > 0ULL);

  nbase_result_t hits[3];
  uint32_t cnt = 0U;
  ASSERT_STATUS(nbase_clustered_search(cl, b0, 4U, 2U, 1U, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 2ULL);
  ASSERT_EQ_U64(hits[0].id, 4ULL);
  ASSERT_EQ_U64(hits[1].id, 5ULL);

  /* Upsert moves id 6 from c0 near a0 and delete removes id 5 */
  float a3[4] = {0.05f, 0.05f, 0.0f, 0.0f};
  ASSERT_STATUS(nbase_clustered_upsert(cl, 6ULL, a3, 4U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_delete(cl, 5ULL), NBASE_OK);
  ASSERT_TRUE(!nbase_clustered_contains(cl, 5ULL));

  float out_v[4] = {0};
  ASSERT_STATUS(nbase_clustered_get(cl, 6ULL, out_v, 4U), NBASE_OK);
  ASSERT_NEAR(out_v[0], 0.05, 1e-6, 0.0);

  /* K-Means recluster into 2 clusters */
  ASSERT_STATUS(nbase_clustered_recluster(cl, 2U), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_stats(cl, &st), NBASE_OK);
  ASSERT_EQ_U64(st.cluster_count, 2ULL);
  ASSERT_EQ_U64(st.active_vectors, 5ULL);

  ASSERT_STATUS(nbase_clustered_search(cl, a0, 4U, 3U, 0U, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 3ULL);
  ASSERT_EQ_U64(hits[0].id, 1ULL);

  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  char path[300];
  snprintf(path, sizeof(path), "%s/index.nbcl", dir);
  ASSERT_STATUS(nbase_clustered_save(cl, path), NBASE_OK);
  nbase_clustered_destroy(cl);

  nbase_clustered_t *loaded = NULL;
  ASSERT_STATUS(nbase_clustered_load(path, &loaded), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_stats(loaded, &st), NBASE_OK);
  ASSERT_EQ_U64(st.active_vectors, 5ULL);
  ASSERT_EQ_U64(st.cluster_count, 2ULL);
  ASSERT_STATUS(nbase_clustered_search(loaded, b0, 4U, 1U, 1U, hits, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 1ULL);
  ASSERT_EQ_U64(hits[0].id, 4ULL);
  nbase_clustered_destroy(loaded);

  ASSERT_TRUE(test_flip_file_byte(path, 20L));
  ASSERT_STATUS(nbase_clustered_load(path, &loaded), NBASE_ERR_CORRUPT);

  test_rm_rf(dir);
}

static bool cl_even_filter(uint64_t id, void *ctx) {
  (void)ctx;
  return (id % 2ULL) == 0ULL;
}

typedef struct cl_mt_arg {
  nbase_clustered_t *cl;
  int err;
} cl_mt_arg_t;

static void *cl_reader_thread(void *raw) {
  cl_mt_arg_t *arg = (cl_mt_arg_t *)raw;
  float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  nbase_result_t hits[4];
  for (int iter = 0; iter < 50; iter++) {
    uint32_t cnt = 0U;
    nbase_status_t st = nbase_clustered_search(arg->cl, q, 4U, 4U, 0U, hits, &cnt);
    if (st != NBASE_OK) {
      arg->err = 1;
      break;
    }
    nbase_clustered_stats_t st_info;
    if (nbase_clustered_stats(arg->cl, &st_info) != NBASE_OK) {
      arg->err = 1;
      break;
    }
  }
  return NULL;
}

static void test_clustered_batch_filtered_and_concurrent(void) {
  nbase_clustered_config_t cfg = nbase_clustered_config_default(4U);
  cfg.target_cluster_size = 4U;
  cfg.max_clusters = 16U;
  cfg.distance_threshold = 0.5f;

  nbase_clustered_t *cl = NULL;
  ASSERT_STATUS(nbase_clustered_create(&cfg, &cl), NBASE_OK);

  uint64_t ids[6] = {10ULL, 11ULL, 12ULL, 13ULL, 14ULL, 15ULL};
  float vecs[24] = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.99f, 0.01f, 0.0f, 0.0f,
      0.98f, 0.02f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.99f, 0.01f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f
  };
  ASSERT_STATUS(nbase_clustered_add_batch(cl, ids, vecs, 6U, 4U), NBASE_OK);

  /* Duplicate ID inside add_batch fails cleanly without inserting partial batch */
  uint64_t bad_ids[2] = {100ULL, 100ULL};
  ASSERT_STATUS(nbase_clustered_add_batch(cl, bad_ids, vecs, 2U, 4U), NBASE_ERR_ALREADY_EXISTS);
  ASSERT_TRUE(!nbase_clustered_contains(cl, 100ULL));

  /* Filtered search selects only even IDs */
  float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  nbase_result_t hits[4];
  uint32_t cnt = 0U;
  ASSERT_STATUS(
      nbase_clustered_search_filtered(cl, q, 4U, 4U, 0U, cl_even_filter, NULL, hits, &cnt),
      NBASE_OK);
  ASSERT_EQ_U64(cnt, 3ULL);
  for (uint32_t i = 0U; i < cnt; i++) {
    ASSERT_TRUE((hits[i].id % 2ULL) == 0ULL);
  }
  ASSERT_EQ_U64(hits[0].id, 10ULL);

  /* Batch delete removes found IDs and ignores missing IDs */
  uint64_t del_ids[3] = {11ULL, 13ULL, 999ULL};
  uint32_t deleted = 0U;
  ASSERT_STATUS(nbase_clustered_delete_batch(cl, del_ids, 3U, &deleted), NBASE_OK);
  ASSERT_EQ_U64(deleted, 2ULL);
  ASSERT_TRUE(!nbase_clustered_contains(cl, 11ULL));
  ASSERT_TRUE(!nbase_clustered_contains(cl, 13ULL));
  ASSERT_TRUE(nbase_clustered_contains(cl, 12ULL));

  /* Concurrent readers + writer verify rwlock under TSan */
  pthread_t readers[2];
  cl_mt_arg_t args[2] = {{cl, 0}, {cl, 0}};
  for (int t = 0; t < 2; t++) {
    ASSERT_TRUE(pthread_create(&readers[t], NULL, cl_reader_thread, &args[t]) == 0);
  }
  for (uint64_t wid = 1000ULL; wid < 1030ULL; wid++) {
    float wv[4] = {(float)(wid % 5ULL), 0.1f, 0.0f, 0.0f};
    ASSERT_STATUS(nbase_clustered_upsert(cl, wid, wv, 4U), NBASE_OK);
  }
  for (int t = 0; t < 2; t++) {
    pthread_join(readers[t], NULL);
    ASSERT_EQ_I64(args[t].err, 0);
  }

  nbase_clustered_destroy(cl);
}

static void *cl_list_reader_thread(void *raw) {
  cl_mt_arg_t *arg = (cl_mt_arg_t *)raw;
  uint64_t ids_buf[8];
  for (int iter = 0; iter < 50; iter++) {
    uint32_t got = 0U;
    uint64_t total = 0ULL;
    nbase_status_t st = nbase_clustered_list_ids(arg->cl, 0ULL, 8U, ids_buf, &got, &total);
    if (st != NBASE_OK || got > 8U) {
      arg->err = 1;
      break;
    }
    for (uint32_t i = 1U; i < got; i++) {
      if (ids_buf[i] <= ids_buf[i - 1U]) {
        arg->err = 1;
        break;
      }
    }
  }
  return NULL;
}

static void test_clustered_upsert_batch_and_list_ids(void) {
  nbase_clustered_config_t cfg = nbase_clustered_config_default(3U);
  cfg.target_cluster_size = 4U;
  cfg.max_clusters = 16U;
  cfg.distance_threshold = 0.5f;

  nbase_clustered_t *cl = NULL;
  ASSERT_STATUS(nbase_clustered_create(&cfg, &cl), NBASE_OK);

  uint64_t init_ids[3] = {30ULL, 10ULL, 20ULL};
  float init_vecs[9] = {
      0.0f, 0.0f, 1.0f,
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f
  };
  ASSERT_STATUS(nbase_clustered_upsert_batch(cl, init_ids, init_vecs, 3U, 3U), NBASE_OK);

  /* Overwrite 20 + add 40 and 5 in a single batch */
  uint64_t up_ids[3] = {40ULL, 20ULL, 5ULL};
  float up_vecs[9] = {
      1.0f, 1.0f, 0.0f,
      0.0f, 0.5f, 0.5f,
      0.2f, 0.0f, 0.0f
  };
  ASSERT_STATUS(nbase_clustered_upsert_batch(cl, up_ids, up_vecs, 3U, 3U), NBASE_OK);

  float got_v[3] = {0.0f, 0.0f, 0.0f};
  ASSERT_STATUS(nbase_clustered_get(cl, 20ULL, got_v, 3U), NBASE_OK);
  ASSERT_NEAR(got_v[1], 0.5, 1e-6, 0.0);

  /* Reject duplicate ID within same batch without adding new ID 99 */
  uint64_t dup_ids[3] = {99ULL, 40ULL, 99ULL};
  ASSERT_STATUS(nbase_clustered_upsert_batch(cl, dup_ids, up_vecs, 3U, 3U), NBASE_ERR_ALREADY_EXISTS);
  ASSERT_TRUE(!nbase_clustered_contains(cl, 99ULL));

  /* Reject NaN vector atomically without adding ID 88 */
  uint64_t nan_ids[2] = {88ULL, 89ULL};
  float nan_vecs[6] = {1.0f, 0.0f, 0.0f, 0.0f, NAN, 0.0f};
  ASSERT_STATUS(nbase_clustered_upsert_batch(cl, nan_ids, nan_vecs, 2U, 3U), NBASE_ERR_INVALID_VECTOR);
  ASSERT_TRUE(!nbase_clustered_contains(cl, 88ULL));

  /* Sorted paginated listing: [5, 10, 20, 30, 40] */
  uint64_t page[3];
  uint32_t cnt = 0U;
  uint64_t total = 0ULL;
  ASSERT_STATUS(nbase_clustered_list_ids(cl, 0ULL, 3U, page, &cnt, &total), NBASE_OK);
  ASSERT_EQ_U64(total, 5ULL);
  ASSERT_EQ_U64(cnt, 3ULL);
  ASSERT_EQ_U64(page[0], 5ULL);
  ASSERT_EQ_U64(page[1], 10ULL);
  ASSERT_EQ_U64(page[2], 20ULL);

  ASSERT_STATUS(nbase_clustered_list_ids(cl, 3ULL, 3U, page, &cnt, &total), NBASE_OK);
  ASSERT_EQ_U64(total, 5ULL);
  ASSERT_EQ_U64(cnt, 2ULL);
  ASSERT_EQ_U64(page[0], 30ULL);
  ASSERT_EQ_U64(page[1], 40ULL);

  ASSERT_STATUS(nbase_clustered_list_ids(cl, 10ULL, 3U, page, &cnt, &total), NBASE_OK);
  ASSERT_EQ_U64(total, 5ULL);
  ASSERT_EQ_U64(cnt, 0ULL);

  /* Concurrent list_ids reader + batch upsert writer under TSan */
  pthread_t rthr;
  cl_mt_arg_t rarg = {cl, 0};
  ASSERT_TRUE(pthread_create(&rthr, NULL, cl_list_reader_thread, &rarg) == 0);
  for (uint64_t step = 0ULL; step < 20ULL; step++) {
    uint64_t bids[2] = {500ULL + step, 600ULL + (step % 4ULL)};
    float bvecs[6] = {
        (float)step * 0.1f, 0.2f, 0.3f,
        0.4f, (float)step * 0.05f, 0.1f
    };
    ASSERT_STATUS(nbase_clustered_upsert_batch(cl, bids, bvecs, 2U, 3U), NBASE_OK);
  }
  pthread_join(rthr, NULL);
  ASSERT_EQ_I64(rarg.err, 0);

  nbase_clustered_destroy(cl);
}

static void *cl_batch_radius_reader_thread(void *raw) {
  cl_mt_arg_t *arg = (cl_mt_arg_t *)raw;
  uint64_t gids[3] = {10ULL, 20ULL, 999999ULL};
  float gvecs[9];
  bool gfound[3];
  float q[3] = {1.0f, 0.0f, 0.0f};
  nbase_result_t rres[8];
  for (int iter = 0; iter < 50; iter++) {
    if (nbase_clustered_get_batch(arg->cl, gids, 3U, 3U, gvecs, gfound) != NBASE_OK ||
        !gfound[0] || gfound[2]) {
      arg->err = 1;
      break;
    }
    uint32_t got = 0U;
    if (nbase_clustered_search_radius(arg->cl, q, 3U, 1.5f, 8U, 0U, NULL, NULL, rres, &got) != NBASE_OK) {
      arg->err = 2;
      break;
    }
    for (uint32_t i = 0U; i < got; i++) {
      if (rres[i].distance > 1.5f) {
        arg->err = 3;
        break;
      }
    }
  }
  return NULL;
}

static void test_clustered_get_batch_and_search_radius(void) {
  nbase_clustered_config_t cfg = nbase_clustered_config_default(3U);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.target_cluster_size = 3U;
  cfg.max_clusters = 16U;
  cfg.distance_threshold = 0.6f;

  nbase_clustered_t *cl = NULL;
  ASSERT_STATUS(nbase_clustered_create(&cfg, &cl), NBASE_OK);

  uint64_t ids[5] = {10ULL, 11ULL, 20ULL, 21ULL, 30ULL};
  float vecs[15] = {
      1.0f, 0.0f, 0.0f,
      1.2f, 0.0f, 0.0f,
      1.0f, 0.4f, 0.0f,
      0.0f, 3.0f, 0.0f,
      1.0f, 0.8f, 0.0f
  };
  ASSERT_STATUS(nbase_clustered_upsert_batch(cl, ids, vecs, 5U, 3U), NBASE_OK);

  /* Batch retrieval including missing ID + NULL out_found + count == 0 */
  uint64_t qids[4] = {30ULL, 999ULL, 10ULL, 11ULL};
  float out_v[12];
  bool found[4] = {false, true, false, false};
  for (int i = 0; i < 12; i++) {
    out_v[i] = 99.0f;
  }
  ASSERT_STATUS(nbase_clustered_get_batch(cl, qids, 4U, 3U, out_v, found), NBASE_OK);
  ASSERT_TRUE(found[0] && !found[1] && found[2] && found[3]);
  ASSERT_NEAR(out_v[0], 1.0, 1e-6, 0.0);
  ASSERT_NEAR(out_v[1], 0.8, 1e-6, 0.0);
  ASSERT_NEAR(out_v[3], 0.0, 1e-6, 0.0);
  ASSERT_NEAR(out_v[4], 0.0, 1e-6, 0.0);
  ASSERT_NEAR(out_v[5], 0.0, 1e-6, 0.0);
  ASSERT_NEAR(out_v[6], 1.0, 1e-6, 0.0);
  ASSERT_NEAR(out_v[9], 1.2, 1e-6, 0.0);
  ASSERT_STATUS(nbase_clustered_get_batch(cl, qids, 4U, 3U, out_v, NULL), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_get_batch(cl, NULL, 0U, 3U, NULL, NULL), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_get_batch(cl, qids, 4U, 2U, out_v, found), NBASE_ERR_DIM_MISMATCH);
  ASSERT_STATUS(nbase_clustered_get_batch(NULL, qids, 1U, 3U, out_v, found), NBASE_ERR_INVALID_ARG);

  /* Radius search: query [1, 0, 0], max_distance 0.45 should match IDs 10 (0.0), 11 (0.2), 20 (0.4) */
  float q[3] = {1.0f, 0.0f, 0.0f};
  nbase_result_t res[5];
  uint32_t cnt = 0U;
  ASSERT_STATUS(nbase_clustered_search_radius(cl, q, 3U, 0.45f, 5U, 0U, NULL, NULL, res, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 3ULL);
  ASSERT_EQ_U64(res[0].id, 10ULL);
  ASSERT_EQ_U64(res[1].id, 11ULL);
  ASSERT_EQ_U64(res[2].id, 20ULL);

  /* Filtered radius search skips odd ID 11 */
  ASSERT_STATUS(
      nbase_clustered_search_radius(cl, q, 3U, 0.45f, 5U, 0U, cl_even_filter, NULL, res, &cnt),
      NBASE_OK);
  ASSERT_EQ_U64(cnt, 2ULL);
  ASSERT_EQ_U64(res[0].id, 10ULL);
  ASSERT_EQ_U64(res[1].id, 20ULL);

  /* Radius error and zero-capacity checks */
  ASSERT_STATUS(nbase_clustered_search_radius(cl, q, 3U, 0.45f, 0U, 0U, NULL, NULL, NULL, &cnt), NBASE_OK);
  ASSERT_EQ_U64(cnt, 0ULL);
  ASSERT_STATUS(nbase_clustered_search_radius(cl, q, 3U, NAN, 5U, 0U, NULL, NULL, res, &cnt), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_clustered_search_radius(cl, q, 2U, 0.5f, 5U, 0U, NULL, NULL, res, &cnt), NBASE_ERR_DIM_MISMATCH);
  float nanq[3] = {1.0f, NAN, 0.0f};
  ASSERT_STATUS(nbase_clustered_search_radius(cl, nanq, 3U, 0.5f, 5U, 0U, NULL, NULL, res, &cnt), NBASE_ERR_INVALID_VECTOR);

  /* Concurrent get_batch + search_radius readers against upsert writer under TSan */
  pthread_t readers[2];
  cl_mt_arg_t args[2] = {{cl, 0}, {cl, 0}};
  for (int t = 0; t < 2; t++) {
    ASSERT_TRUE(pthread_create(&readers[t], NULL, cl_batch_radius_reader_thread, &args[t]) == 0);
  }
  for (uint64_t step = 0ULL; step < 25ULL; step++) {
    float wv[3] = {0.0f, 5.0f + (float)step * 0.1f, 0.0f};
    ASSERT_STATUS(nbase_clustered_upsert(cl, 700ULL + step, wv, 3U), NBASE_OK);
  }
  for (int t = 0; t < 2; t++) {
    pthread_join(readers[t], NULL);
    ASSERT_EQ_I64(args[t].err, 0);
  }

  nbase_clustered_destroy(cl);
}

static void *cl_search_batch_reader_thread(void *raw) {
  cl_mt_arg_t *arg = (cl_mt_arg_t *)raw;
  float qs[4U * 4U];
  test_fill_vectors(qs, 4U, 4U, 9911ULL);
  nbase_result_t bres[4U * 3U];
  uint32_t bcnts[4U];
  for (int iter = 0; iter < 40; iter++) {
    if (nbase_clustered_search_batch(arg->cl, qs, 4U, 4U, 3U, 2U, 2U, bres, bcnts) != NBASE_OK) {
      arg->err = 1;
      break;
    }
  }
  return NULL;
}

static void test_clustered_compact_and_search_batch_mt(void) {
  const uint32_t dim = 4U;
  const uint32_t n = 120U;
  const uint32_t nq = 10U;
  const uint32_t k = 5U;

  nbase_clustered_config_t cfg = nbase_clustered_config_default(dim);
  cfg.metric = NBASE_METRIC_COSINE;
  cfg.target_cluster_size = 16U;
  cfg.max_clusters = 32U;
  cfg.distance_threshold = 0.5f;

  nbase_clustered_t *cl = NULL;
  ASSERT_STATUS(nbase_clustered_create(&cfg, &cl), NBASE_OK);

  uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  float *qs = malloc((size_t)nq * dim * sizeof(float));
  nbase_result_t *bres = malloc((size_t)nq * k * sizeof(nbase_result_t));
  uint32_t *bcnts = malloc((size_t)nq * sizeof(uint32_t));
  ASSERT_TRUE(ids && vecs && qs && bres && bcnts);

  test_fill_vectors(vecs, n, dim, 3100ULL);
  test_fill_vectors(qs, nq, dim, 3200ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = 100ULL + i;
  }
  ASSERT_STATUS(nbase_clustered_add_batch(cl, ids, vecs, n, dim), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_recluster(cl, 8U), NBASE_OK);

  /* Batch search with 1 and 4 threads must match serial nbase_clustered_search */
  const uint32_t threads[2] = {1U, 4U};
  for (size_t ti = 0; ti < 2; ti++) {
    ASSERT_STATUS(
        nbase_clustered_search_batch(cl, qs, nq, dim, k, 4U, threads[ti], bres, bcnts),
        NBASE_OK);
    for (uint32_t q = 0U; q < nq; q++) {
      nbase_result_t single[5];
      uint32_t sc = 0U;
      ASSERT_STATUS(
          nbase_clustered_search(cl, qs + (size_t)q * dim, dim, k, 4U, single, &sc),
          NBASE_OK);
      ASSERT_EQ_U64(bcnts[q], sc);
      for (uint32_t r = 0U; r < sc; r++) {
        ASSERT_EQ_U64(bres[(size_t)q * k + r].id, single[r].id);
        ASSERT_NEAR(bres[(size_t)q * k + r].distance, single[r].distance, 1e-6, 1e-6);
      }
    }
  }

  /* Validation & edge cases */
  ASSERT_STATUS(nbase_clustered_search_batch(cl, NULL, 0U, dim, k, 2U, 2U, NULL, NULL), NBASE_OK);
  bcnts[0] = 77U;
  ASSERT_STATUS(nbase_clustered_search_batch(cl, qs, 2U, dim, 0U, 2U, 2U, NULL, bcnts), NBASE_OK);
  ASSERT_EQ_U64(bcnts[0], 0ULL);
  ASSERT_STATUS(nbase_clustered_search_batch(cl, qs, 2U, dim - 1U, k, 2U, 2U, bres, bcnts), NBASE_ERR_DIM_MISMATCH);
  qs[dim] = NAN;
  ASSERT_STATUS(nbase_clustered_search_batch(cl, qs, 2U, dim, k, 2U, 2U, bres, bcnts), NBASE_ERR_INVALID_VECTOR);
  qs[dim] = 0.25f;

  /* Delete 90 vectors to build tombstones, verify compact reclaims memory and preserves results */
  ASSERT_STATUS(nbase_clustered_delete_batch(cl, ids, 90U, NULL), NBASE_OK);
  nbase_clustered_stats_t pre_st, post_st;
  ASSERT_STATUS(nbase_clustered_stats(cl, &pre_st), NBASE_OK);
  ASSERT_EQ_U64(pre_st.active_vectors, 30ULL);

  ASSERT_STATUS(nbase_clustered_compact(cl), NBASE_OK);
  /* Idempotent no-op when tombstones == 0 */
  ASSERT_STATUS(nbase_clustered_compact(cl), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_stats(cl, &post_st), NBASE_OK);
  ASSERT_EQ_U64(post_st.active_vectors, 30ULL);
  ASSERT_TRUE(post_st.memory_bytes < pre_st.memory_bytes);
  ASSERT_STATUS(nbase_clustered_compact(NULL), NBASE_ERR_INVALID_ARG);

  float got_v[4] = {0};
  ASSERT_STATUS(nbase_clustered_get(cl, ids[95], got_v, dim), NBASE_OK);
  ASSERT_NEAR(got_v[0], vecs[95U * dim], 1e-6, 0.0);

  /* Concurrent batch searches + writer under TSan */
  pthread_t readers[2];
  cl_mt_arg_t rargs[2] = {{cl, 0}, {cl, 0}};
  for (int t = 0; t < 2; t++) {
    ASSERT_TRUE(pthread_create(&readers[t], NULL, cl_search_batch_reader_thread, &rargs[t]) == 0);
  }
  for (uint64_t wid = 1000ULL; wid < 1020ULL; wid++) {
    float wv[4] = {1.0f, (float)wid * 0.01f, 0.2f, 0.3f};
    ASSERT_STATUS(nbase_clustered_upsert(cl, wid, wv, dim), NBASE_OK);
  }
  for (int t = 0; t < 2; t++) {
    pthread_join(readers[t], NULL);
    ASSERT_EQ_I64(rargs[t].err, 0);
  }

  /* Verify save + load round-trip post-compact */
  char dir[256];
  ASSERT_TRUE(test_make_tmpdir(dir, sizeof(dir)));
  char path[300];
  snprintf(path, sizeof(path), "%s/compacted.nbcl", dir);
  ASSERT_STATUS(nbase_clustered_save(cl, path), NBASE_OK);
  nbase_clustered_t *reloaded = NULL;
  ASSERT_STATUS(nbase_clustered_load(path, &reloaded), NBASE_OK);
  ASSERT_TRUE(nbase_clustered_contains(reloaded, ids[95]));
  ASSERT_TRUE(!nbase_clustered_contains(reloaded, ids[0]));
  nbase_clustered_destroy(reloaded);
  test_rm_rf(dir);

  free(bcnts);
  free(bres);
  free(qs);
  free(vecs);
  free(ids);
  nbase_clustered_destroy(cl);
}

typedef struct cl_sim_tsan_ctx {
  const nbase_clustered_t *cl;
  atomic_bool stop;
  atomic_int errors;
} cl_sim_tsan_ctx_t;

static void *cl_sim_reader_fn(void *raw) {
  cl_sim_tsan_ctx_t *ctx = (cl_sim_tsan_ctx_t *)raw;
  while (!atomic_load_explicit(&ctx->stop, memory_order_relaxed)) {
    nbase_result_t out[5];
    uint32_t got = 0U;
    if (nbase_clustered_find_similar(ctx->cl, 100ULL, 5U, 8U, NULL, NULL, out, &got) != NBASE_OK ||
        got == 0U) {
      atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      break;
    }
    for (uint32_t i = 0U; i < got; i++) {
      if (out[i].id == 100ULL) {
        atomic_fetch_add_explicit(&ctx->errors, 1, memory_order_relaxed);
      }
    }
  }
  return NULL;
}

static void test_clustered_find_similar_and_tsan(void) {
  const uint32_t dim = 8U;
  nbase_clustered_config_t cfg = nbase_clustered_config_default(dim);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.target_cluster_size = 8U;
  cfg.max_clusters = 8U;
  cfg.seed = 4418ULL;

  nbase_clustered_t *cl = NULL;
  ASSERT_STATUS(nbase_clustered_create(&cfg, &cl), NBASE_OK);

  const uint32_t n = 32U;
  uint64_t *ids = malloc(n * sizeof(uint64_t));
  float *vecs = malloc((size_t)n * dim * sizeof(float));
  ASSERT_TRUE(ids != NULL && vecs != NULL);
  test_fill_vectors(vecs, n, dim, 5118ULL);
  for (uint32_t i = 0U; i < n; i++) {
    ids[i] = 100ULL + i;
  }
  /* Make id 101 closest to id 100, and id 102 second closest (even). */
  for (uint32_t d = 0U; d < dim; d++) {
    vecs[1U * dim + d] = vecs[d] + 0.01f;
    vecs[2U * dim + d] = vecs[d] + 0.05f;
  }

  ASSERT_STATUS(nbase_clustered_add_batch(cl, ids, vecs, n, dim), NBASE_OK);
  ASSERT_STATUS(nbase_clustered_recluster(cl, 4U), NBASE_OK);

  nbase_result_t res[5];
  uint32_t got = 0U;
  ASSERT_STATUS(nbase_clustered_find_similar(NULL, 100ULL, 5U, 4U, NULL, NULL, res, &got), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_clustered_find_similar(cl, 100ULL, 5U, 4U, NULL, NULL, NULL, &got), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_clustered_find_similar(cl, 100ULL, 5U, 4U, NULL, NULL, res, NULL), NBASE_ERR_INVALID_ARG);
  ASSERT_STATUS(nbase_clustered_find_similar(cl, 9999ULL, 5U, 4U, NULL, NULL, res, &got), NBASE_ERR_NOT_FOUND);
  ASSERT_STATUS(nbase_clustered_find_similar(cl, 100ULL, 0U, 4U, NULL, NULL, NULL, &got), NBASE_OK);
  ASSERT_EQ_U64(got, 0ULL);

  ASSERT_STATUS(nbase_clustered_find_similar(cl, 100ULL, 5U, 0U, NULL, NULL, res, &got), NBASE_OK);
  ASSERT_EQ_U64(got, 5ULL);
  ASSERT_EQ_U64(res[0].id, 101ULL);
  for (uint32_t i = 0U; i < got; i++) {
    ASSERT_TRUE(res[i].id != 100ULL);
  }

  ASSERT_STATUS(nbase_clustered_find_similar(cl, 100ULL, 4U, 0U, cl_even_filter, NULL, res, &got), NBASE_OK);
  ASSERT_EQ_U64(got, 4ULL);
  ASSERT_EQ_U64(res[0].id, 102ULL);
  for (uint32_t i = 0U; i < got; i++) {
    ASSERT_TRUE((res[i].id % 2ULL) == 0ULL && res[i].id != 100ULL);
  }

  cl_sim_tsan_ctx_t ctx;
  ctx.cl = cl;
  atomic_init(&ctx.stop, false);
  atomic_init(&ctx.errors, 0);
  pthread_t r1, r2;
  ASSERT_TRUE(pthread_create(&r1, NULL, cl_sim_reader_fn, &ctx) == 0);
  ASSERT_TRUE(pthread_create(&r2, NULL, cl_sim_reader_fn, &ctx) == 0);
  for (uint32_t step = 0U; step < 8U; step++) {
    ASSERT_STATUS(
        nbase_clustered_upsert(cl, 110ULL + (step % 4U), vecs + (size_t)(10U + step) * dim, dim),
        NBASE_OK);
  }
  atomic_store_explicit(&ctx.stop, true, memory_order_relaxed);
  pthread_join(r1, NULL);
  pthread_join(r2, NULL);
  ASSERT_EQ_U64((uint64_t)atomic_load_explicit(&ctx.errors, memory_order_relaxed), 0ULL);

  free(vecs);
  free(ids);
  nbase_clustered_destroy(cl);
}

void test_clustered_run(void) {
  RUN_TEST(test_clustered_online_assign_search_and_recluster);
  RUN_TEST(test_clustered_batch_filtered_and_concurrent);
  RUN_TEST(test_clustered_upsert_batch_and_list_ids);
  RUN_TEST(test_clustered_get_batch_and_search_radius);
  RUN_TEST(test_clustered_compact_and_search_batch_mt);
  RUN_TEST(test_clustered_find_similar_and_tsan);
}
