#include "test_common.h"

#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int g_tests_run = 0;
int g_tests_failed = 0;

static uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

void test_fill_vectors(float *out, uint32_t count, uint32_t dim, uint64_t seed) {
  uint64_t state = seed;
  size_t total = (size_t)count * (size_t)dim;
  for (size_t i = 0; i < total; i++) {
    uint64_t r = splitmix64_next(&state) >> 40;
    out[i] = (float)((double)r / 16777216.0);
  }
}

double test_ref_distance(nbase_metric_t metric, const float *a, const float *b, uint32_t dim) {
  switch (metric) {
    case NBASE_METRIC_EUCLIDEAN:
    case NBASE_METRIC_SQUARED_EUCLIDEAN: {
      double sum = 0.0;
      for (uint32_t i = 0U; i < dim; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
      }
      return (metric == NBASE_METRIC_EUCLIDEAN) ? sqrt(sum) : sum;
    }
    case NBASE_METRIC_COSINE: {
      double dot = 0.0;
      double na = 0.0;
      double nb = 0.0;
      for (uint32_t i = 0U; i < dim; i++) {
        double ai = (double)a[i];
        double bi = (double)b[i];
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
      }
      double denom = sqrt(na) * sqrt(nb);
      if (denom == 0.0) {
        return 1.0;
      }
      double sim = dot / denom;
      if (sim > 1.0) {
        sim = 1.0;
      } else if (sim < -1.0) {
        sim = -1.0;
      }
      return 1.0 - sim;
    }
    case NBASE_METRIC_INNER_PRODUCT: {
      double sum = 0.0;
      for (uint32_t i = 0U; i < dim; i++) {
        sum += (double)a[i] * (double)b[i];
      }
      return -sum;
    }
    case NBASE_METRIC_MANHATTAN: {
      double sum = 0.0;
      for (uint32_t i = 0U; i < dim; i++) {
        sum += fabs((double)a[i] - (double)b[i]);
      }
      return sum;
    }
    case NBASE_METRIC_CHEBYSHEV: {
      double max_d = 0.0;
      for (uint32_t i = 0U; i < dim; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > max_d) {
          max_d = d;
        }
      }
      return max_d;
    }
    case NBASE_METRIC_HAMMING: {
      uint32_t diff = 0U;
      for (uint32_t i = 0U; i < dim; i++) {
        if (a[i] != b[i]) {
          diff++;
        }
      }
      return (double)diff;
    }
    default:
      return 0.0;
  }
}

static int cmp_res_asc(const void *pa, const void *pb) {
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

void test_brute_force(
    const uint64_t *ids,
    const float *vectors,
    const bool *deleted,
    uint32_t count,
    uint32_t dim,
    const float *query,
    uint32_t k,
    nbase_metric_t metric,
    nbase_filter_fn filter,
    void *filter_ctx,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  nbase_result_t *all = malloc((size_t)count * sizeof(nbase_result_t));
  if (!all) {
    *out_count = 0U;
    return;
  }
  uint32_t n = 0U;
  for (uint32_t i = 0U; i < count; i++) {
    if (deleted != NULL && deleted[i]) {
      continue;
    }
    uint64_t id = ids ? ids[i] : (uint64_t)i;
    if (filter != NULL && !filter(id, filter_ctx)) {
      continue;
    }
    all[n].id = id;
    all[n].distance = (float)test_ref_distance(metric, query, vectors + (size_t)i * (size_t)dim, dim);
    n++;
  }
  qsort(all, n, sizeof(nbase_result_t), cmp_res_asc);
  uint32_t ret = (n < k) ? n : k;
  if (ret > 0U) {
    memcpy(out_results, all, (size_t)ret * sizeof(nbase_result_t));
  }
  *out_count = ret;
  free(all);
}

double test_recall_at(const nbase_result_t *actual, uint32_t actual_count, const nbase_result_t *expected, uint32_t expected_count, uint32_t k) {
  uint32_t exp_n = (expected_count < k) ? expected_count : k;
  uint32_t act_n = (actual_count < k) ? actual_count : k;
  if (exp_n == 0U) {
    return 1.0;
  }
  uint32_t hits = 0U;
  for (uint32_t i = 0U; i < exp_n; i++) {
    for (uint32_t j = 0U; j < act_n; j++) {
      if (expected[i].id == actual[j].id) {
        hits++;
        break;
      }
    }
  }
  return (double)hits / (double)exp_n;
}

bool test_make_tmpdir(char *buf, size_t buf_len) {
  const char *tmpl = "/tmp/libnbase-test-XXXXXX";
  if (buf_len <= strlen(tmpl)) {
    return false;
  }
  snprintf(buf, buf_len, "%s", tmpl);
  return mkdtemp(buf) != NULL;
}

void test_rm_rf(const char *path) {
  struct stat st;
  if (lstat(path, &st) != 0) {
    return;
  }
  if (!S_ISDIR(st.st_mode)) {
    unlink(path);
    return;
  }
  DIR *dir = opendir(path);
  if (!dir) {
    return;
  }
  struct dirent *ent;
  while ((ent = readdir(dir)) != NULL) {
    if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
      continue;
    }
    size_t len = strlen(path) + 1U + strlen(ent->d_name) + 1U;
    char *child = malloc(len);
    if (!child) {
      continue;
    }
    snprintf(child, len, "%s/%s", path, ent->d_name);
    test_rm_rf(child);
    free(child);
  }
  closedir(dir);
  rmdir(path);
}

bool test_flip_file_byte(const char *path, long offset) {
  FILE *fp = fopen(path, "r+b");
  if (!fp) {
    return false;
  }
  if (fseek(fp, offset, SEEK_SET) != 0) {
    fclose(fp);
    return false;
  }
  int c = fgetc(fp);
  if (c == EOF) {
    fclose(fp);
    return false;
  }
  if (fseek(fp, offset, SEEK_SET) != 0) {
    fclose(fp);
    return false;
  }
  if (fputc((unsigned char)(c ^ 0xFF), fp) == EOF) {
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}

int main(void) {
  printf("libnbase %s (SIMD backend: %s)\n", nbase_version(), nbase_simd_backend());

  printf("\n[Distance tests]\n");
  test_distance_run();

  printf("\n[Store tests]\n");
  test_store_run();

  printf("\n[HNSW tests]\n");
  test_hnsw_run();

  printf("\n[DB facade tests]\n");
  test_db_run();

  printf("\n[Metadata tests]\n");
  test_meta_run();

  printf("\n[K-Means & PQ tests]\n");
  test_pq_run();

  printf("\n[Partitioned DB tests]\n");
  test_partitioned_run();

  printf("\n[Clustered Vector DB tests]\n");
  test_clustered_run();

  printf("\n[WAL & crash recovery tests]\n");
  test_wal_run();

  printf("\n[IVF-PQ compressed index tests]\n");
  test_ivfpq_run();

  printf("\n[MMR rerank & graph analytics tests]\n");
  test_rerank_run();

  printf("\n[SQ8 scalar quantization tests]\n");
  test_sq8_run();

  printf("\n[BQ binary quantization tests]\n");
  test_bq_run();

  printf("\n[BM25 & hybrid RRF tests]\n");
  test_bm25_run();

  printf("\n[DB Text & Hybrid Search tests]\n");
  test_hybrid_run();

  printf("\n[LSH index tests]\n");
  test_lsh_run();

  printf("\n[Strict JSON tests]\n");
  test_json_run();

  printf("\n[HTTP/1.1 REST server tests]\n");
  test_server_run();

  printf("\n[Concurrency & TSan tests]\n");
  test_concurrency_run();

  printf("\nSummary: %d run, %d failed\n", g_tests_run, g_tests_failed);
  return g_tests_failed == 0 ? 0 : 1;
}
