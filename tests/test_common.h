#ifndef NBASE_TEST_COMMON_H
#define NBASE_TEST_COMMON_H

#include "nbase.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int g_tests_run;
extern int g_tests_failed;

#define TEST_FAIL_MSG(fmt, ...)                                                            \
  do {                                                                                     \
    fprintf(stderr, "FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, __VA_ARGS__);             \
    g_tests_failed++;                                                                      \
    return;                                                                                \
  } while (0)

#define ASSERT_TRUE(cond)                                                                  \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      TEST_FAIL_MSG("expected true: %s", #cond);                                           \
    }                                                                                      \
  } while (0)

#define ASSERT_EQ_U64(actual, expected)                                                    \
  do {                                                                                     \
    uint64_t _a = (uint64_t)(actual);                                                      \
    uint64_t _e = (uint64_t)(expected);                                                    \
    if (_a != _e) {                                                                        \
      TEST_FAIL_MSG("%s == %llu, expected %llu", #actual, (unsigned long long)_a, (unsigned long long)_e); \
    }                                                                                      \
  } while (0)

#define ASSERT_EQ_I64(actual, expected)                                                    \
  do {                                                                                     \
    int64_t _a = (int64_t)(actual);                                                        \
    int64_t _e = (int64_t)(expected);                                                      \
    if (_a != _e) {                                                                        \
      TEST_FAIL_MSG("%s == %lld, expected %lld", #actual, (long long)_a, (long long)_e);   \
    }                                                                                      \
  } while (0)

#define ASSERT_STATUS(actual, expected)                                                    \
  do {                                                                                     \
    nbase_status_t _a = (actual);                                                          \
    nbase_status_t _e = (expected);                                                        \
    if (_a != _e) {                                                                        \
      TEST_FAIL_MSG("%s returned %d (%s), expected %d (%s)", #actual, (int)_a, nbase_strerror(_a), (int)_e, nbase_strerror(_e)); \
    }                                                                                      \
  } while (0)

#define ASSERT_NEAR(actual, expected, abs_tol, rel_tol)                                    \
  do {                                                                                     \
    double _a = (double)(actual);                                                          \
    double _e = (double)(expected);                                                        \
    double _diff = fabs(_a - _e);                                                          \
    double _limit = (double)(abs_tol) + (double)(rel_tol) * fabs(_e);                      \
    if (!isfinite(_a) || _diff > _limit) {                                                 \
      TEST_FAIL_MSG("%s == %.8g, expected %.8g (diff %.8g > %.8g)", #actual, _a, _e, _diff, _limit); \
    }                                                                                      \
  } while (0)

#define RUN_TEST(fn)                                                                       \
  do {                                                                                     \
    int _before = g_tests_failed;                                                          \
    g_tests_run++;                                                                         \
    fn();                                                                                  \
    if (g_tests_failed == _before) {                                                       \
      printf("  [PASS] %s\n", #fn);                                                        \
    } else {                                                                               \
      printf("  [FAIL] %s\n", #fn);                                                        \
    }                                                                                      \
  } while (0)

void test_fill_vectors(float *out, uint32_t count, uint32_t dim, uint64_t seed);
double test_ref_distance(nbase_metric_t metric, const float *a, const float *b, uint32_t dim);
void test_brute_force(const uint64_t *ids, const float *vectors, const bool *deleted, uint32_t count, uint32_t dim, const float *query, uint32_t k, nbase_metric_t metric, nbase_filter_fn filter, void *filter_ctx, nbase_result_t *out_results, uint32_t *out_count);
double test_recall_at(const nbase_result_t *actual, uint32_t actual_count, const nbase_result_t *expected, uint32_t expected_count, uint32_t k);
bool test_make_tmpdir(char *buf, size_t buf_len);
void test_rm_rf(const char *path);
bool test_flip_file_byte(const char *path, long offset);

void test_distance_run(void);
void test_store_run(void);
void test_hnsw_run(void);
void test_db_run(void);
void test_meta_run(void);
void test_pq_run(void);
void test_partitioned_run(void);
void test_clustered_run(void);
void test_wal_run(void);
void test_ivfpq_run(void);
void test_rerank_run(void);
void test_sq8_run(void);
void test_bq_run(void);
void test_bm25_run(void);
void test_hybrid_run(void);
void test_lsh_run(void);
void test_json_run(void);
void test_server_run(void);
void test_concurrency_run(void);

#endif /* NBASE_TEST_COMMON_H */
