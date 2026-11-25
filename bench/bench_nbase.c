#include "nbase.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t rng_next(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static void fill_vectors(float *out, uint32_t count, uint32_t dim, uint64_t seed) {
  uint64_t state = seed;
  size_t total = (size_t)count * (size_t)dim;
  for (size_t i = 0; i < total; i++) {
    uint64_t r = rng_next(&state) >> 40;
    out[i] = (float)((double)r / 16777216.0);
  }
}

static double ref_euclidean(const float *a, const float *b, uint32_t dim) {
  double sum = 0.0;
  for (uint32_t i = 0U; i < dim; i++) {
    double d = (double)a[i] - (double)b[i];
    sum += d * d;
  }
  return sqrt(sum);
}

static int cmp_res(const void *pa, const void *pb) {
  const nbase_result_t *a = (const nbase_result_t *)pa;
  const nbase_result_t *b = (const nbase_result_t *)pb;
  if (a->distance < b->distance) {
    return -1;
  }
  if (a->distance > b->distance) {
    return 1;
  }
  if (a->id < b->id) {
    return -1;
  }
  if (a->id > b->id) {
    return 1;
  }
  return 0;
}

static void bench_distance_metric(const char *label, nbase_metric_t metric, uint32_t dim, uint32_t iters) {
  float *a = malloc((size_t)dim * sizeof(float));
  float *b = malloc((size_t)dim * sizeof(float));
  if (!a || !b) {
    free(a);
    free(b);
    return;
  }
  fill_vectors(a, 1U, dim, 11ULL);
  fill_vectors(b, 1U, dim, 22ULL);

  volatile float sink = 0.0f;

  uint64_t t0 = now_ns();
  for (uint32_t i = 0U; i < iters; i++) {
    sink += nbase_distance_scalar(metric, a, b, dim);
  }
  uint64_t t1 = now_ns();

  uint64_t t2 = now_ns();
  for (uint32_t i = 0U; i < iters; i++) {
    sink += nbase_distance(metric, a, b, dim);
  }
  uint64_t t3 = now_ns();
  (void)sink;

  double scalar_ns = (double)(t1 - t0) / (double)iters;
  double simd_ns = (double)(t3 - t2) / (double)iters;
  double speedup = (simd_ns > 0.0) ? (scalar_ns / simd_ns) : 1.0;

  printf("  %-18s dim=%-4u  scalar=%6.2f ns/op  simd=%6.2f ns/op  speedup=%.2fx\n", label, dim, scalar_ns, simd_ns, speedup);

  free(a);
  free(b);
}

static void bench_distance_batch_metric(const char *label, nbase_metric_t metric, uint32_t dim, uint32_t count, uint32_t rounds) {
  float *q = malloc((size_t)dim * sizeof(float));
  float *vecs = malloc((size_t)count * (size_t)dim * sizeof(float));
  float *dists = malloc((size_t)count * sizeof(float));
  if (!q || !vecs || !dists) {
    free(q);
    free(vecs);
    free(dists);
    return;
  }
  fill_vectors(q, 1U, dim, 33ULL);
  fill_vectors(vecs, count, dim, 44ULL);

  volatile float sink = 0.0f;
  uint64_t t0 = now_ns();
  for (uint32_t r = 0U; r < rounds; r++) {
    for (uint32_t i = 0U; i < count; i++) {
      dists[i] = nbase_distance(metric, q, vecs + (size_t)i * (size_t)dim, dim);
    }
    sink += dists[r % count];
  }
  uint64_t t1 = now_ns();

  uint64_t t2 = now_ns();
  for (uint32_t r = 0U; r < rounds; r++) {
    (void)nbase_distance_batch(metric, q, vecs, count, dim, dists);
    sink += dists[r % count];
  }
  uint64_t t3 = now_ns();
  (void)sink;

  double single_us = (double)(t1 - t0) / (double)rounds / 1e3;
  double batch_us = (double)(t3 - t2) / (double)rounds / 1e3;
  double speedup = (batch_us > 0.0) ? (single_us / batch_us) : 1.0;

  printf("  %-18s dim=%-4u  1-by-1(N=%u)=%6.2f us  batch=%6.2f us  speedup=%.2fx\n",
         label, dim, count, single_us, batch_us, speedup);

  free(dists);
  free(vecs);
  free(q);
}

static void bench_hnsw_dataset(uint32_t count, uint32_t dim, uint32_t nqueries) {
  const uint32_t k = 10U;
  float *vecs = malloc((size_t)count * (size_t)dim * sizeof(float));
  float *queries = malloc((size_t)nqueries * (size_t)dim * sizeof(float));
  nbase_result_t *oracle = malloc((size_t)count * sizeof(nbase_result_t));
  if (!vecs || !queries || !oracle) {
    free(vecs);
    free(queries);
    free(oracle);
    return;
  }

  fill_vectors(vecs, count, dim, 1001ULL);
  fill_vectors(queries, nqueries, dim, 2002ULL);

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.hnsw_ef_search = 100U;
  cfg.seed = 3003ULL;
  nbase_db_t *db = NULL;
  if (nbase_create(&cfg, &db) != NBASE_OK) {
    free(vecs);
    free(queries);
    free(oracle);
    return;
  }

  uint64_t b0 = now_ns();
  for (uint32_t i = 0U; i < count; i++) {
    if (nbase_add(db, (uint64_t)i, vecs + (size_t)i * dim, dim) != NBASE_OK) {
      nbase_close(db);
      free(vecs);
      free(queries);
      free(oracle);
      return;
    }
  }
  uint64_t b1 = now_ns();

  double total_recall = 0.0;
  uint64_t query_ns_sum = 0ULL;

  for (uint32_t q = 0U; q < nqueries; q++) {
    const float *qv = queries + (size_t)q * dim;
    for (uint32_t i = 0U; i < count; i++) {
      oracle[i].id = (uint64_t)i;
      oracle[i].distance = (float)ref_euclidean(qv, vecs + (size_t)i * dim, dim);
    }
    qsort(oracle, count, sizeof(nbase_result_t), cmp_res);

    nbase_result_t hits[10];
    uint32_t hit_n = 0U;
    uint64_t q0 = now_ns();
    (void)nbase_search(db, qv, dim, k, NULL, hits, &hit_n);
    uint64_t q1 = now_ns();
    query_ns_sum += (q1 - q0);

    uint32_t matched = 0U;
    for (uint32_t i = 0U; i < k; i++) {
      for (uint32_t j = 0U; j < hit_n; j++) {
        if (oracle[i].id == hits[j].id) {
          matched++;
          break;
        }
      }
    }
    total_recall += (double)matched / (double)k;
  }

  double build_ms = (double)(b1 - b0) / 1e6;
  double avg_query_us = ((double)query_ns_sum / (double)nqueries) / 1e3;
  double qps = (avg_query_us > 0.0) ? (1e6 / avg_query_us) : 0.0;
  double avg_recall = total_recall / (double)nqueries;

  printf("  N=%-5u dim=%-4u  build=%7.2f ms  avg_query=%6.2f us  QPS=%7.0f  recall@10=%.3f\n", count, dim, build_ms, avg_query_us, qps, avg_recall);

  nbase_close(db);
  free(vecs);
  free(queries);
  free(oracle);
}

static void bench_batch_threads_and_pq(void) {
  const uint32_t count = 8000U;
  const uint32_t dim = 64U;
  const uint32_t nqueries = 200U;
  const uint32_t k = 10U;

  float *vecs = malloc((size_t)count * dim * sizeof(float));
  float *queries = malloc((size_t)nqueries * dim * sizeof(float));
  nbase_result_t *res = malloc((size_t)nqueries * k * sizeof(nbase_result_t));
  uint32_t *counts = malloc(nqueries * sizeof(uint32_t));
  if (!vecs || !queries || !res || !counts) {
    free(vecs);
    free(queries);
    free(res);
    free(counts);
    return;
  }

  fill_vectors(vecs, count, dim, 5005ULL);
  fill_vectors(queries, nqueries, dim, 6006ULL);

  nbase_config_t cfg = nbase_config_default(dim);
  cfg.hnsw_ef_search = 100U;
  cfg.seed = 7007ULL;
  nbase_db_t *db = NULL;
  if (nbase_create(&cfg, &db) != NBASE_OK) {
    free(vecs);
    free(queries);
    free(res);
    free(counts);
    return;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (nbase_add(db, (uint64_t)i, vecs + (size_t)i * dim, dim) != NBASE_OK) {
      nbase_close(db);
      free(vecs);
      free(queries);
      free(res);
      free(counts);
      return;
    }
  }

  printf("\n[Multi-Threaded Batch Search (N=8000, dim=64, 200 queries, efSearch=100)]\n");
  static const uint32_t thread_counts[] = {1U, 4U, 8U};
  for (size_t ti = 0; ti < sizeof(thread_counts) / sizeof(thread_counts[0]); ti++) {
    nbase_search_opts_t opts = nbase_search_opts_default();
    opts.num_threads = thread_counts[ti];
    uint64_t t0 = now_ns();
    (void)nbase_search_batch(db, queries, nqueries, dim, k, &opts, res, counts);
    uint64_t t1 = now_ns();
    double ms = (double)(t1 - t0) / 1e6;
    double qps = (ms > 0.0) ? ((double)nqueries * 1000.0 / ms) : 0.0;
    printf("  threads=%u  total=%6.2f ms  QPS=%7.0f\n", thread_counts[ti], ms, qps);
  }
  nbase_close(db);

  printf("\n[Product Quantization (N=8000, dim=64, M=8, ks=256)]\n");
  const uint32_t pq_m = 8U;
  const uint32_t pq_ks = 256U;
  nbase_pq_t *pq = NULL;
  uint64_t pt0 = now_ns();
  if (nbase_pq_train(vecs, count, dim, pq_m, pq_ks, 15U, 8008ULL, &pq) == NBASE_OK) {
    uint64_t pt1 = now_ns();
    uint8_t *codes = malloc((size_t)count * pq_m * sizeof(uint8_t));
    float *lut = malloc((size_t)pq_m * pq_ks * sizeof(float));
    if (codes && lut) {
      uint64_t et0 = now_ns();
      (void)nbase_pq_encode(pq, vecs, count, codes);
      uint64_t et1 = now_ns();

      volatile float sink = 0.0f;
      uint64_t st0 = now_ns();
      for (uint32_t q = 0U; q < 20U; q++) {
        (void)nbase_pq_build_lut(pq, queries + (size_t)q * dim, lut);
        for (uint32_t i = 0U; i < count; i++) {
          sink += nbase_pq_adc_distance(pq, lut, codes + (size_t)i * pq_m);
        }
      }
      uint64_t st1 = now_ns();

      double avg_batch_us = 0.0;
      float *adc_out = malloc((size_t)count * sizeof(float));
      if (adc_out) {
        uint64_t bt0 = now_ns();
        for (uint32_t q = 0U; q < 20U; q++) {
          (void)nbase_pq_build_lut(pq, queries + (size_t)q * dim, lut);
          (void)nbase_pq_adc_distance_batch(pq, lut, codes, count, adc_out);
          sink += adc_out[0];
        }
        uint64_t bt1 = now_ns();
        avg_batch_us = (double)(bt1 - bt0) / 20.0 / 1e3;
        free(adc_out);
      }
      (void)sink;

      double ratio = (double)(dim * sizeof(float)) / (double)pq_m;
      double avg_scan_us = (double)(st1 - st0) / 20.0 / 1e3;
      printf("  train=%6.2f ms  encode(8000)=%5.2f ms  compression=%.0fx  adc_scan_8000=%6.2f us/query  adc_batch=%6.2f us/query\n",
             (double)(pt1 - pt0) / 1e6,
             (double)(et1 - et0) / 1e6,
             ratio,
             avg_scan_us,
             avg_batch_us);
    }
    free(codes);
    free(lut);
    nbase_pq_destroy(pq);
  }

  printf("\n[Parallel HNSW Batch Insert & IVF-PQ (N=8000, dim=64)]\n");
  uint64_t *batch_ids = malloc((size_t)count * sizeof(uint64_t));
  if (batch_ids) {
    for (uint32_t i = 0U; i < count; i++) {
      batch_ids[i] = (uint64_t)i + 1ULL;
    }

    nbase_config_t mt_cfg = nbase_config_default(dim);
    mt_cfg.seed = 4040ULL;
    nbase_db_t *mt_db = NULL;
    if (nbase_create(&mt_cfg, &mt_db) == NBASE_OK) {
      uint64_t mt0 = now_ns();
      (void)nbase_add_batch(mt_db, batch_ids, vecs, count, dim, 4U);
      uint64_t mt1 = now_ns();
      uint64_t rb0 = now_ns();
      (void)nbase_rebuild_hnsw_mt(mt_db, 4U);
      uint64_t rb1 = now_ns();
      nbase_search_opts_t ex_opts = nbase_search_opts_default();
      ex_opts.mode = NBASE_SEARCH_EXACT;
      ex_opts.skip_cache = true;
      nbase_result_t ex_hits[10];
      uint32_t ex_got = 0U;
      uint64_t ex0 = now_ns();
      for (uint32_t q = 0U; q < 20U; q++) {
        (void)nbase_search(mt_db, queries + (size_t)q * dim, dim, k, &ex_opts, ex_hits, &ex_got);
      }
      uint64_t ex1 = now_ns();
      printf("  add_batch_mt(4t)=%6.2f ms  rebuild_mt(4t)=%6.2f ms  exact_scan_8000=%6.2f us/query\n",
             (double)(mt1 - mt0) / 1e6,
             (double)(rb1 - rb0) / 1e6,
             (double)(ex1 - ex0) / 20.0 / 1e3);
      nbase_close(mt_db);
    }

    nbase_ivfpq_config_t ivf_cfg = nbase_ivfpq_config_default(dim, 32U, 8U);
    ivf_cfg.num_centroids = 128U;
    ivf_cfg.max_iters = 10U;
    nbase_ivfpq_t *ivf = NULL;
    uint64_t it0 = now_ns();
    if (nbase_ivfpq_train(&ivf_cfg, vecs, count, &ivf) == NBASE_OK) {
      uint64_t it1 = now_ns();
      uint64_t ia0 = now_ns();
      (void)nbase_ivfpq_add(ivf, batch_ids, vecs, count);
      uint64_t ia1 = now_ns();
      uint64_t iq0 = now_ns();
      nbase_result_t ivf_hits[10];
      uint32_t ivf_got = 0U;
      for (uint32_t q = 0U; q < nqueries; q++) {
        (void)nbase_ivfpq_search(ivf, queries + (size_t)q * dim, k, 4U, ivf_hits, &ivf_got);
      }
      uint64_t iq1 = now_ns();
      double ivf_q_us = (double)(iq1 - iq0) / (double)nqueries / 1e3;
      uint64_t ib0 = now_ns();
      (void)nbase_ivfpq_search_batch(ivf, queries, nqueries, k, 4U, 4U, res, counts);
      uint64_t ib1 = now_ns();
      double ivf_batch_ms = (double)(ib1 - ib0) / 1e6;
      double ivf_batch_qps = (ivf_batch_ms > 0.0) ? ((double)nqueries * 1000.0 / ivf_batch_ms) : 0.0;
      printf("  ivfpq(nlist=32,M=8)  train=%6.2f ms  add=%6.2f ms  search(nprobe=4)=%6.2f us/query  search_batch(4t)=%.2f ms (%.0f QPS)\n",
             (double)(it1 - it0) / 1e6,
             (double)(ia1 - ia0) / 1e6,
             ivf_q_us,
             ivf_batch_ms,
             ivf_batch_qps);
      nbase_ivfpq_destroy(ivf);
    }
    free(batch_ids);
  }

  printf("\n[SQ8 Scalar Quantization & LSH (N=8000, dim=64)]\n");
  nbase_sq8_t *sq8 = NULL;
  uint64_t sq_t0 = now_ns();
  if (nbase_sq8_train(vecs, count, dim, &sq8) == NBASE_OK) {
    uint64_t sq_t1 = now_ns();
    uint8_t *sq_codes = malloc((size_t)count * dim);
    if (sq_codes) {
      uint64_t sq_e0 = now_ns();
      (void)nbase_sq8_encode(sq8, vecs, count, sq_codes);
      uint64_t sq_e1 = now_ns();
      volatile float sq_sink = 0.0f;
      uint64_t sq_s0 = now_ns();
      for (uint32_t q = 0U; q < 20U; q++) {
        const float *qv = queries + (size_t)q * dim;
        for (uint32_t i = 0U; i < count; i++) {
          sq_sink += nbase_sq8_l2_sq_asymmetric(sq8, qv, sq_codes + (size_t)i * dim);
        }
      }
      uint64_t sq_s1 = now_ns();
      double asym_batch_us = 0.0;
      double sym_scan_us = 0.0;
      double sym_batch_us = 0.0;
      float *sq_out = malloc((size_t)count * sizeof(float));
      if (sq_out) {
        uint64_t sb0 = now_ns();
        for (uint32_t q = 0U; q < 20U; q++) {
          (void)nbase_sq8_l2_sq_asymmetric_batch(
              sq8, queries + (size_t)q * dim, sq_codes, count, sq_out);
          sq_sink += sq_out[0];
        }
        uint64_t sb1 = now_ns();
        asym_batch_us = (double)(sb1 - sb0) / 20.0 / 1e3;

        uint64_t sy0 = now_ns();
        for (uint32_t q = 0U; q < 20U; q++) {
          const uint8_t *qc = sq_codes + (size_t)q * dim;
          for (uint32_t i = 0U; i < count; i++) {
            sq_sink += nbase_sq8_l2_sq_symmetric(sq8, qc, sq_codes + (size_t)i * dim);
          }
        }
        uint64_t sy1 = now_ns();
        sym_scan_us = (double)(sy1 - sy0) / 20.0 / 1e3;

        uint64_t syb0 = now_ns();
        for (uint32_t q = 0U; q < 20U; q++) {
          (void)nbase_sq8_l2_sq_symmetric_batch(
              sq8, sq_codes + (size_t)q * dim, sq_codes, count, sq_out);
          sq_sink += sq_out[0];
        }
        uint64_t syb1 = now_ns();
        sym_batch_us = (double)(syb1 - syb0) / 20.0 / 1e3;
        free(sq_out);
      }
      (void)sq_sink;
      printf("  sq8  train=%5.2f ms  encode(8000)=%5.2f ms  compression=4x  asym_scan_8000=%6.2f us/query  asym_batch=%6.2f us/query  sym_scan=%6.2f us  sym_batch=%6.2f us\n",
             (double)(sq_t1 - sq_t0) / 1e6,
             (double)(sq_e1 - sq_e0) / 1e6,
             (double)(sq_s1 - sq_s0) / 20.0 / 1e3,
             asym_batch_us,
             sym_scan_us,
             sym_batch_us);
      free(sq_codes);
    }
    nbase_sq8_destroy(sq8);
  }

  nbase_lsh_config_t lsh_cfg = nbase_lsh_config_default(dim);
  lsh_cfg.metric = NBASE_METRIC_COSINE;
  lsh_cfg.num_tables = 8U;
  lsh_cfg.num_bits = 10U;
  nbase_lsh_t *lsh = NULL;
  if (nbase_lsh_create(&lsh_cfg, &lsh) == NBASE_OK) {
    uint64_t *lids = malloc((size_t)count * sizeof(uint64_t));
    double add_batch_ms = 0.0;
    if (lids) {
      for (uint32_t i = 0U; i < count; i++) {
        lids[i] = (uint64_t)i + 1ULL;
      }
    }
    uint64_t l0 = now_ns();
    if (lids && nbase_lsh_add_batch(lsh, lids, vecs, count, dim) == NBASE_OK) {
      add_batch_ms = (double)(now_ns() - l0) / 1e6;
    } else {
      for (uint32_t i = 0U; i < count; i++) {
        (void)nbase_lsh_add(lsh, (uint64_t)i + 1ULL, vecs + (size_t)i * dim, dim);
      }
      add_batch_ms = (double)(now_ns() - l0) / 1e6;
    }
    free(lids);
    nbase_result_t lhits[10];
    uint32_t lgot = 0U;
    uint64_t lq0 = now_ns();
    for (uint32_t q = 0U; q < nqueries; q++) {
      (void)nbase_lsh_search(lsh, queries + (size_t)q * dim, dim, k, 2U, lhits, &lgot);
    }
    uint64_t lq1 = now_ns();
    nbase_result_t *lb_res = malloc((size_t)nqueries * k * sizeof(nbase_result_t));
    uint32_t *lb_cnts = malloc((size_t)nqueries * sizeof(uint32_t));
    double lb_qps = 0.0;
    if (lb_res && lb_cnts) {
      uint64_t lb0 = now_ns();
      (void)nbase_lsh_search_batch(lsh, queries, nqueries, dim, k, 2U, 4U, lb_res, lb_cnts);
      uint64_t lb1 = now_ns();
      double lb_sec = (double)(lb1 - lb0) / 1e9;
      lb_qps = (lb_sec > 0.0) ? ((double)nqueries / lb_sec) : 0.0;
    }
    free(lb_res);
    free(lb_cnts);
    printf("  lsh(8x10b)  build_batch=%5.2f ms  search(multiprobe=2)=%6.2f us/query  search_batch(4t)=%7.0f QPS\n",
           add_batch_ms,
           (double)(lq1 - lq0) / (double)nqueries / 1e3,
           lb_qps);
    nbase_lsh_destroy(lsh);
  }

  printf("\n[Binary Quantization (BQ) & BM25 Full-Text (N=8000)]\n");
  nbase_bq_t *bq = NULL;
  uint64_t bq_t0 = now_ns();
  if (nbase_bq_train(vecs, count, dim, &bq) == NBASE_OK) {
    uint64_t bq_t1 = now_ns();
    uint32_t wpv = nbase_bq_words_per_vec(bq);
    uint64_t *bq_words = malloc((size_t)count * wpv * sizeof(uint64_t));
    uint64_t q_words[4];
    if (bq_words) {
      uint64_t bq_e0 = now_ns();
      (void)nbase_bq_encode(bq, vecs, count, bq_words);
      uint64_t bq_e1 = now_ns();
      (void)nbase_bq_encode(bq, queries, 1U, q_words);
      volatile uint32_t ham_sink = 0U;
      uint64_t bq_s0 = now_ns();
      for (uint32_t q = 0U; q < 20U; q++) {
        for (uint32_t i = 0U; i < count; i++) {
          ham_sink += nbase_bq_hamming(q_words, bq_words + (size_t)i * wpv, wpv);
        }
      }
      uint64_t bq_s1 = now_ns();
      double bq_batch_us = 0.0;
      uint32_t *bq_out = malloc((size_t)count * sizeof(uint32_t));
      if (bq_out) {
        uint64_t bb0 = now_ns();
        for (uint32_t q = 0U; q < 20U; q++) {
          (void)nbase_bq_hamming_batch(q_words, bq_words, count, wpv, bq_out);
          ham_sink += bq_out[0];
        }
        uint64_t bb1 = now_ns();
        bq_batch_us = (double)(bb1 - bb0) / 20.0 / 1e3;
        free(bq_out);
      }
      (void)ham_sink;
      printf("  bq(dim=64)  train=%5.2f ms  encode(8000)=%5.2f ms  compression=32x  hamming_scan_8000=%5.2f us/query  hamming_batch=%5.2f us/query\n",
             (double)(bq_t1 - bq_t0) / 1e6,
             (double)(bq_e1 - bq_e0) / 1e6,
             (double)(bq_s1 - bq_s0) / 20.0 / 1e3,
             bq_batch_us);
      free(bq_words);
    }
    nbase_bq_destroy(bq);
  }

  nbase_bm25_t *bm25 = NULL;
  if (nbase_bm25_create(1.5f, 0.75f, &bm25) == NBASE_OK) {
    static const char *vocab[8] = {"vector", "hnsw", "avx2", "quantization", "graph", "storage", "wal", "hybrid"};
    uint64_t bm_i0 = now_ns();
    for (uint32_t i = 0U; i < 2000U; i++) {
      char doc[128];
      snprintf(doc, sizeof(doc), "embedded %s engine %s index %s %s doc%u",
               vocab[i & 7U], vocab[(i * 3U) & 7U], vocab[(i * 5U) & 7U], (i % 10U == 0U) ? "avx2" : "c11", i);
      (void)nbase_bm25_index(bm25, (uint64_t)i + 1ULL, doc);
    }
    uint64_t bm_i1 = now_ns();
    nbase_scored_id_t bhits[10];
    uint32_t bgot = 0U;
    uint64_t bm_s0 = now_ns();
    for (uint32_t q = 0U; q < 50U; q++) {
      (void)nbase_bm25_search(bm25, "vector avx2 hnsw", 10U, bhits, &bgot);
    }
    uint64_t bm_s1 = now_ns();
    const char *bm_qs[50];
    for (uint32_t q = 0U; q < 50U; q++) {
      bm_qs[q] = "vector avx2 hnsw";
    }
    nbase_scored_id_t *bmb_res = malloc(50U * 10U * sizeof(nbase_scored_id_t));
    uint32_t bmb_cnts[50];
    double bm_batch_qps = 0.0;
    if (bmb_res) {
      uint64_t bmb0 = now_ns();
      (void)nbase_bm25_search_batch(bm25, bm_qs, 50U, 10U, 4U, bmb_res, bmb_cnts);
      uint64_t bmb1 = now_ns();
      bm_batch_qps = 50.0 / ((double)(bmb1 - bmb0) / 1e9);
      free(bmb_res);
    }
    printf("  bm25(2000 docs)  index=%5.2f ms  search=%6.2f us/query  search_batch(4t)=%7.0f QPS\n",
           (double)(bm_i1 - bm_i0) / 1e6,
           (double)(bm_s1 - bm_s0) / 50.0 / 1e3,
           bm_batch_qps);
    nbase_bm25_destroy(bm25);
  }

  printf("\n[DB-Integrated Hybrid Search (N=2000, dim=64)]\n");
  nbase_config_t hcfg = nbase_config_default(dim);
  hcfg.hnsw_ef_search = 64U;
  hcfg.seed = 9009ULL;
  nbase_db_t *hdb = NULL;
  if (nbase_create(&hcfg, &hdb) == NBASE_OK) {
    static const char *hvocab[6] = {"vector", "hnsw", "avx2", "bm25", "hybrid", "c11"};
    uint64_t hb0 = now_ns();
    for (uint32_t i = 0U; i < 2000U; i++) {
      uint64_t id = (uint64_t)i + 1ULL;
      char doc[96];
      snprintf(doc, sizeof(doc), "%s %s engine doc%u", hvocab[i % 6U], hvocab[(i * 5U) % 6U], i);
      (void)nbase_add(hdb, id, vecs + (size_t)i * dim, dim);
      (void)nbase_set_text(hdb, id, doc);
    }
    uint64_t hb1 = now_ns();
    nbase_scored_id_t hhits[10];
    uint32_t hgot = 0U;
    uint64_t hs0 = now_ns();
    for (uint32_t q = 0U; q < 50U; q++) {
      (void)nbase_search_hybrid(hdb, queries + (size_t)q * dim, dim, "vector avx2 hybrid", 10U, 0.5f, 60.0f, NULL, hhits, &hgot);
    }
    uint64_t hs1 = now_ns();
    printf("  hybrid_db(N=2000,dim=64)  build=%6.2f ms  search=%6.2f us/query\n",
           (double)(hb1 - hb0) / 1e6,
           (double)(hs1 - hs0) / 50.0 / 1e3);
    nbase_close(hdb);
  }

  printf("\n[Dynamic Clustered Vector DB (N=4000, dim=64)]\n");
  nbase_clustered_config_t cl_cfg = nbase_clustered_config_default(dim);
  cl_cfg.target_cluster_size = 128U;
  cl_cfg.max_clusters = 64U;
  cl_cfg.distance_threshold = 2.8f;
  nbase_clustered_t *cl = NULL;
  if (nbase_clustered_create(&cl_cfg, &cl) == NBASE_OK) {
    const uint32_t cl_n = 4000U;
    uint64_t ca0 = now_ns();
    for (uint32_t i = 0U; i < cl_n; i++) {
      (void)nbase_clustered_add(cl, (uint64_t)i + 1ULL, vecs + (size_t)i * dim, dim);
    }
    uint64_t ca1 = now_ns();
    uint64_t cr0 = now_ns();
    (void)nbase_clustered_recluster(cl, 32U);
    uint64_t cr1 = now_ns();
    nbase_result_t chits[10];
    uint32_t cgot = 0U;
    uint64_t cs0 = now_ns();
    for (uint32_t q = 0U; q < nqueries; q++) {
      (void)nbase_clustered_search(cl, queries + (size_t)q * dim, dim, k, 4U, chits, &cgot);
    }
    uint64_t cs1 = now_ns();
    uint64_t crd0 = now_ns();
    for (uint32_t q = 0U; q < nqueries; q++) {
      (void)nbase_clustered_search_radius(cl, queries + (size_t)q * dim, dim, 2.5f, k, 4U, NULL, NULL, chits, &cgot);
    }
    uint64_t crd1 = now_ns();
    uint64_t gids[128];
    float *gbuf = malloc(128U * (size_t)dim * sizeof(float));
    bool gfound[128];
    double gbatch_us = 0.0;
    if (gbuf) {
      for (uint32_t i = 0U; i < 128U; i++) {
        gids[i] = (uint64_t)i * 10ULL + 1ULL;
      }
      uint64_t gb0 = now_ns();
      (void)nbase_clustered_get_batch(cl, gids, 128U, dim, gbuf, gfound);
      uint64_t gb1 = now_ns();
      gbatch_us = (double)(gb1 - gb0) / 1e3;
      free(gbuf);
    }
    double cb_qps = 0.0;
    nbase_result_t *cb_res = malloc((size_t)nqueries * k * sizeof(nbase_result_t));
    uint32_t *cb_cnts = malloc((size_t)nqueries * sizeof(uint32_t));
    if (cb_res && cb_cnts) {
      uint64_t cb0 = now_ns();
      (void)nbase_clustered_search_batch(cl, queries, nqueries, dim, k, 4U, 4U, cb_res, cb_cnts);
      uint64_t cb1 = now_ns();
      double cb_sec = (double)(cb1 - cb0) / 1e9;
      cb_qps = (cb_sec > 0.0) ? ((double)nqueries / cb_sec) : 0.0;
    }
    free(cb_res);
    free(cb_cnts);
    printf("  clustered(N=4000,k=32)  add=%6.2f ms  recluster=%6.2f ms  search(nprobe=4)=%6.2f us/query  search_batch(4t)=%7.0f QPS  radius=%6.2f us/query  get_batch(128)=%5.2f us\n",
           (double)(ca1 - ca0) / 1e6,
           (double)(cr1 - cr0) / 1e6,
           (double)(cs1 - cs0) / (double)nqueries / 1e3,
           cb_qps,
           (double)(crd1 - crd0) / (double)nqueries / 1e3,
           gbatch_us);
    nbase_clustered_destroy(cl);
  }

  printf("\n[Query LRU Cache & Weighted Rerank (N=2000, dim=64)]\n");
  nbase_config_t ccfg = nbase_config_default(dim);
  ccfg.cache_capacity = 256U;
  ccfg.seed = 777ULL;
  nbase_db_t *cdb = NULL;
  if (nbase_create(&ccfg, &cdb) == NBASE_OK) {
    for (uint32_t i = 0U; i < 2000U; i++) {
      uint64_t id = (uint64_t)i + 1ULL;
      (void)nbase_add(cdb, id, vecs + (size_t)i * dim, dim);
      nbase_meta_field_t fld = {"score", {.type = NBASE_META_DOUBLE, .as = {.f64 = (double)(i % 100U) * 0.01}}};
      (void)nbase_set_meta(cdb, id, &fld, 1U);
    }
    nbase_result_t cands[20];
    nbase_result_t reranked[10];
    uint32_t cgot = 0U;
    uint32_t rgot = 0U;
    uint64_t m0 = now_ns();
    for (uint32_t q = 0U; q < nqueries; q++) {
      (void)nbase_search(cdb, queries + (size_t)q * dim, dim, k, NULL, cands, &cgot);
    }
    uint64_t m1 = now_ns();
    uint64_t h0 = now_ns();
    for (uint32_t r = 0U; r < 10U; r++) {
      for (uint32_t q = 0U; q < nqueries; q++) {
        (void)nbase_search(cdb, queries + (size_t)q * dim, dim, k, NULL, cands, &cgot);
      }
    }
    uint64_t h1 = now_ns();
    nbase_search_opts_t r_opts = nbase_search_opts_default();
    r_opts.skip_cache = true;
    (void)nbase_search(cdb, queries, dim, 20U, &r_opts, cands, &cgot);
    nbase_rerank_weight_t w = {"score", 0.2f};
    uint64_t w0 = now_ns();
    for (uint32_t r = 0U; r < 200U; r++) {
      (void)nbase_rerank_weighted(cdb, cands, cgot, &w, 1U, k, reranked, &rgot);
    }
    uint64_t w1 = now_ns();
    uint64_t rad0 = now_ns();
    for (uint32_t q = 0U; q < nqueries; q++) {
      (void)nbase_search_radius(
          cdb, queries + (size_t)q * dim, dim, 2.5f, k, NULL, reranked, &rgot);
    }
    uint64_t rad1 = now_ns();

    nbase_meta_batch_item_t *mitems = malloc(2000U * sizeof(*mitems));
    nbase_meta_field_t *mflds = malloc(2000U * sizeof(*mflds));
    uint64_t *tids = malloc(2000U * sizeof(uint64_t));
    const char **tptrs = malloc(2000U * sizeof(const char *));
    double meta_batch_ms = 0.0;
    double text_batch_ms = 0.0;
    if (mitems && mflds && tids && tptrs) {
      for (uint32_t i = 0U; i < 2000U; i++) {
        mflds[i].key = "score";
        mflds[i].value.type = NBASE_META_DOUBLE;
        mflds[i].value.as.f64 = (double)(i % 100U) * 0.01;
        mitems[i].id = (uint64_t)i + 1ULL;
        mitems[i].fields = &mflds[i];
        mitems[i].field_count = 1U;
        tids[i] = (uint64_t)i + 1ULL;
        tptrs[i] = "vector benchmark batch text indexing doc";
      }
      uint64_t mb0 = now_ns();
      (void)nbase_set_meta_batch(cdb, mitems, 2000U);
      uint64_t mb1 = now_ns();
      uint64_t tb0 = now_ns();
      (void)nbase_set_text_batch(cdb, tids, tptrs, 2000U);
      uint64_t tb1 = now_ns();
      meta_batch_ms = (double)(mb1 - mb0) / 1e6;
      text_batch_ms = (double)(tb1 - tb0) / 1e6;
    }
    free(mitems);
    free(mflds);
    free(tids);
    free((void *)tptrs);

    printf("  lru_cache  miss=%6.2f us/query  hit=%5.2f us/query  weighted_rerank(20->10)=%5.2f us/call  radius=%6.2f us/query  meta_batch(2000)=%.2f ms  text_batch(2000)=%.2f ms\n",
           (double)(m1 - m0) / (double)nqueries / 1e3,
           (double)(h1 - h0) / (double)(10U * nqueries) / 1e3,
           (double)(w1 - w0) / 200.0 / 1e3,
           (double)(rad1 - rad0) / (double)nqueries / 1e3,
           meta_batch_ms,
           text_batch_ms);
    nbase_close(cdb);
  }

  free(vecs);
  free(queries);
  free(res);
  free(counts);
}

int main(void) {
  printf("libnbase %s benchmark (SIMD backend: %s)\n\n", nbase_version(), nbase_simd_backend());

  printf("[Distance Kernels]\n");
  bench_distance_metric("Euclidean", NBASE_METRIC_EUCLIDEAN, 128U, 2000000U);
  bench_distance_metric("Cosine", NBASE_METRIC_COSINE, 128U, 2000000U);
  bench_distance_metric("Euclidean", NBASE_METRIC_EUCLIDEAN, 1536U, 250000U);
  bench_distance_metric("Cosine", NBASE_METRIC_COSINE, 1536U, 250000U);
  bench_distance_batch_metric("Batch SqL2", NBASE_METRIC_SQUARED_EUCLIDEAN, 8U, 10000U, 200U);
  bench_distance_batch_metric("Batch SqL2", NBASE_METRIC_SQUARED_EUCLIDEAN, 128U, 10000U, 200U);
  bench_distance_batch_metric("Batch Cosine", NBASE_METRIC_COSINE, 128U, 10000U, 200U);
  bench_distance_batch_metric("Batch InnerProd", NBASE_METRIC_INNER_PRODUCT, 128U, 10000U, 200U);
  bench_distance_batch_metric("Batch Manhattan", NBASE_METRIC_MANHATTAN, 128U, 10000U, 200U);
  bench_distance_batch_metric("Batch Chebyshev", NBASE_METRIC_CHEBYSHEV, 128U, 10000U, 200U);
  bench_distance_batch_metric("Batch Hamming", NBASE_METRIC_HAMMING, 128U, 10000U, 200U);

  printf("\n[HNSW Index Build & Search (M=16, efConstruction=200, efSearch=100)]\n");
  bench_hnsw_dataset(2000U, 128U, 20U);
  bench_hnsw_dataset(8000U, 64U, 20U);

  bench_batch_threads_and_pq();

  return 0;
}
