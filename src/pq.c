#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct nbase_pq {
  uint32_t dim;
  uint32_t num_subvectors;
  uint32_t sub_dim;
  uint32_t num_centroids;
  float *codebooks;
};

typedef struct pq_train_ctx {
  const float *vectors;
  uint32_t count;
  uint32_t dim;
  uint32_t num_subvectors;
  uint32_t sub_dim;
  uint32_t num_centroids;
  uint32_t max_iters;
  uint64_t seed;
  float *codebooks;
  atomic_uint next_m;
  atomic_int first_err;
} pq_train_ctx_t;

static void *pq_train_worker(void *raw) {
  pq_train_ctx_t *ctx = (pq_train_ctx_t *)raw;
  float *slice = malloc((size_t)ctx->count * (size_t)ctx->sub_dim * sizeof(float));
  if (!slice) {
    int expected = (int)NBASE_OK;
    (void)atomic_compare_exchange_strong(&ctx->first_err, &expected, (int)NBASE_ERR_NOMEM);
    return NULL;
  }

  while (true) {
    if (atomic_load_explicit(&ctx->first_err, memory_order_acquire) != (int)NBASE_OK) {
      break;
    }
    uint32_t m = atomic_fetch_add_explicit(&ctx->next_m, 1U, memory_order_relaxed);
    if (m >= ctx->num_subvectors) {
      break;
    }
    uint32_t offset = m * ctx->sub_dim;
    for (uint32_t i = 0U; i < ctx->count; i++) {
      memcpy(
          slice + (size_t)i * (size_t)ctx->sub_dim,
          ctx->vectors + (size_t)i * (size_t)ctx->dim + offset,
          (size_t)ctx->sub_dim * sizeof(float));
    }
    float *dst_cb = ctx->codebooks + (size_t)m * (size_t)ctx->num_centroids * (size_t)ctx->sub_dim;
    nbase_status_t st = nbase_kmeans_train(
        slice,
        ctx->count,
        ctx->sub_dim,
        ctx->num_centroids,
        ctx->max_iters,
        ctx->seed + (uint64_t)m,
        dst_cb,
        NULL);
    if (st != NBASE_OK) {
      int expected = (int)NBASE_OK;
      (void)atomic_compare_exchange_strong(&ctx->first_err, &expected, (int)st);
      break;
    }
  }

  free(slice);
  return NULL;
}

nbase_status_t nbase_pq_train(
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    uint32_t num_subvectors,
    uint32_t num_centroids,
    uint32_t max_iters,
    uint64_t seed,
    nbase_pq_t **out_pq) {
  if (!vectors || !out_pq || dim == 0U || dim > NBASE_MAX_DIM || num_subvectors == 0U || (dim % num_subvectors) != 0U || num_centroids == 0U || num_centroids > 256U || count < num_centroids) {
    return NBASE_ERR_INVALID_ARG;
  }
  uint32_t sub_dim = dim / num_subvectors;
  size_t codebook_floats = (size_t)num_subvectors * (size_t)num_centroids * (size_t)sub_dim;
  if (codebook_floats > SIZE_MAX / sizeof(float) || (size_t)count > SIZE_MAX / ((size_t)sub_dim * sizeof(float))) {
    return NBASE_ERR_NOMEM;
  }

  nbase_pq_t *pq = calloc(1, sizeof(*pq));
  if (!pq) {
    return NBASE_ERR_NOMEM;
  }
  pq->dim = dim;
  pq->num_subvectors = num_subvectors;
  pq->sub_dim = sub_dim;
  pq->num_centroids = num_centroids;
  pq->codebooks = malloc(codebook_floats * sizeof(float));
  if (!pq->codebooks) {
    nbase_pq_destroy(pq);
    return NBASE_ERR_NOMEM;
  }

  pq_train_ctx_t ctx;
  ctx.vectors = vectors;
  ctx.count = count;
  ctx.dim = dim;
  ctx.num_subvectors = num_subvectors;
  ctx.sub_dim = sub_dim;
  ctx.num_centroids = num_centroids;
  ctx.max_iters = max_iters;
  ctx.seed = seed;
  ctx.codebooks = pq->codebooks;
  atomic_init(&ctx.next_m, 0U);
  atomic_init(&ctx.first_err, (int)NBASE_OK);

  uint32_t nthreads = 1U;
  if (num_subvectors > 1U && (uint64_t)count * (uint64_t)num_centroids >= 16384ULL) {
    nthreads = (num_subvectors < 4U) ? num_subvectors : 4U;
  }

  pthread_t workers[3];
  uint32_t spawned = 0U;
  for (uint32_t t = 1U; t < nthreads; t++) {
    if (pthread_create(&workers[spawned], NULL, pq_train_worker, &ctx) == 0) {
      spawned++;
    }
  }
  (void)pq_train_worker(&ctx);
  for (uint32_t t = 0U; t < spawned; t++) {
    pthread_join(workers[t], NULL);
  }

  nbase_status_t st = (nbase_status_t)atomic_load_explicit(&ctx.first_err, memory_order_acquire);
  if (st != NBASE_OK) {
    nbase_pq_destroy(pq);
    return st;
  }

  *out_pq = pq;
  return NBASE_OK;
}

void nbase_pq_destroy(nbase_pq_t *pq) {
  if (!pq) {
    return;
  }
  free(pq->codebooks);
  free(pq);
}

uint32_t nbase_pq_dim(const nbase_pq_t *pq) {
  return pq ? pq->dim : 0U;
}

uint32_t nbase_pq_num_subvectors(const nbase_pq_t *pq) {
  return pq ? pq->num_subvectors : 0U;
}

uint32_t nbase_pq_num_centroids(const nbase_pq_t *pq) {
  return pq ? pq->num_centroids : 0U;
}

nbase_status_t nbase_pq_encode(const nbase_pq_t *pq, const float *vectors, uint32_t count, uint8_t *out_codes) {
  if (!pq || (count > 0U && (!vectors || !out_codes))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)pq->dim, pq->dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  float dists[256] = {0.0f};
  for (uint32_t i = 0U; i < count; i++) {
    const float *vec = vectors + (size_t)i * (size_t)pq->dim;
    uint8_t *code = out_codes + (size_t)i * (size_t)pq->num_subvectors;
    for (uint32_t m = 0U; m < pq->num_subvectors; m++) {
      const float *sub = vec + (size_t)m * (size_t)pq->sub_dim;
      const float *cb = pq->codebooks + (size_t)m * (size_t)pq->num_centroids * (size_t)pq->sub_dim;
      nbase_status_t st = nbase_distance_batch(
          NBASE_METRIC_SQUARED_EUCLIDEAN, sub, cb, pq->num_centroids, pq->sub_dim, dists);
      if (st != NBASE_OK) {
        return st;
      }
      uint32_t best_c = 0U;
      float best_d = FLT_MAX;
      for (uint32_t c = 0U; c < pq->num_centroids; c++) {
        if (dists[c] < best_d) {
          best_d = dists[c];
          best_c = c;
        }
      }
      code[m] = (uint8_t)best_c;
    }
  }
  return NBASE_OK;
}

nbase_status_t nbase_pq_decode(const nbase_pq_t *pq, const uint8_t *codes, uint32_t count, float *out_vectors) {
  if (!pq || (count > 0U && (!codes || !out_vectors))) {
    return NBASE_ERR_INVALID_ARG;
  }
  for (uint32_t i = 0U; i < count; i++) {
    const uint8_t *code = codes + (size_t)i * (size_t)pq->num_subvectors;
    float *dst = out_vectors + (size_t)i * (size_t)pq->dim;
    for (uint32_t m = 0U; m < pq->num_subvectors; m++) {
      uint32_t c = (uint32_t)code[m];
      if (c >= pq->num_centroids) {
        return NBASE_ERR_INVALID_ARG;
      }
      const float *cb = pq->codebooks + (size_t)m * (size_t)pq->num_centroids * (size_t)pq->sub_dim;
      memcpy(dst + (size_t)m * (size_t)pq->sub_dim, cb + (size_t)c * (size_t)pq->sub_dim, (size_t)pq->sub_dim * sizeof(float));
    }
  }
  return NBASE_OK;
}

nbase_status_t nbase_pq_build_lut(const nbase_pq_t *pq, const float *query, float *out_lut) {
  if (!pq || !query || !out_lut) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (!nbase_vec_is_finite(query, pq->dim)) {
    return NBASE_ERR_INVALID_VECTOR;
  }
  for (uint32_t m = 0U; m < pq->num_subvectors; m++) {
    const float *q_sub = query + (size_t)m * (size_t)pq->sub_dim;
    const float *cb = pq->codebooks + (size_t)m * (size_t)pq->num_centroids * (size_t)pq->sub_dim;
    float *lut_row = out_lut + (size_t)m * (size_t)pq->num_centroids;
    nbase_status_t st = nbase_distance_batch(
        NBASE_METRIC_SQUARED_EUCLIDEAN, q_sub, cb, pq->num_centroids, pq->sub_dim, lut_row);
    if (st != NBASE_OK) {
      return st;
    }
  }
  return NBASE_OK;
}

static inline float pq_adc_one_256(const float *lut, const uint8_t *code, uint32_t m_cnt) {
  float s0 = 0.0f;
  float s1 = 0.0f;
  float s2 = 0.0f;
  float s3 = 0.0f;
  uint32_t m = 0U;
  for (; m + 3U < m_cnt; m += 4U) {
    const float *row = lut + (size_t)m * 256U;
    s0 += row[code[m]];
    s1 += row[256U + (size_t)code[m + 1U]];
    s2 += row[512U + (size_t)code[m + 2U]];
    s3 += row[768U + (size_t)code[m + 3U]];
  }
  float sum = (s0 + s1) + (s2 + s3);
  for (; m < m_cnt; m++) {
    sum += lut[(size_t)m * 256U + (size_t)code[m]];
  }
  return sum;
}

float nbase_pq_adc_distance(const nbase_pq_t *pq, const float *lut, const uint8_t *code) {
  if (!pq || !lut || !code) {
    return 0.0f;
  }
  if (pq->num_centroids == 256U) {
    return pq_adc_one_256(lut, code, pq->num_subvectors);
  }
  float sum = 0.0f;
  for (uint32_t m = 0U; m < pq->num_subvectors; m++) {
    uint32_t c = (uint32_t)code[m];
    if (c < pq->num_centroids) {
      sum += lut[(size_t)m * (size_t)pq->num_centroids + c];
    }
  }
  return sum;
}

nbase_status_t nbase_pq_adc_distance_batch(
    const nbase_pq_t *pq,
    const float *lut,
    const uint8_t *codes,
    uint32_t count,
    float *out_distances) {
  if (!pq || !lut || (count > 0U && (!codes || !out_distances))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }
  const uint32_t m_cnt = pq->num_subvectors;
  const uint32_t nc = pq->num_centroids;
  if (nc == 256U) {
    for (uint32_t i = 0U; i < count; i++) {
      out_distances[i] = pq_adc_one_256(lut, codes + (size_t)i * (size_t)m_cnt, m_cnt);
    }
    return NBASE_OK;
  }
  for (uint32_t i = 0U; i < count; i++) {
    const uint8_t *code = codes + (size_t)i * (size_t)m_cnt;
    float sum = 0.0f;
    for (uint32_t m = 0U; m < m_cnt; m++) {
      uint32_t c = (uint32_t)code[m];
      if (c >= nc) {
        return NBASE_ERR_INVALID_ARG;
      }
      sum += lut[(size_t)m * (size_t)nc + c];
    }
    out_distances[i] = sum;
  }
  return NBASE_OK;
}

nbase_status_t nbase_pq_save(const nbase_pq_t *pq, const char *file_path) {
  if (!pq || !file_path) {
    return NBASE_ERR_INVALID_ARG;
  }
  size_t payload_bytes = (size_t)pq->num_subvectors * (size_t)pq->num_centroids * (size_t)pq->sub_dim * sizeof(float);

  nbq_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBQ_MAGIC, 8);
  hdr.version = NBASE_NBQ_VERSION;
  hdr.dim = pq->dim;
  hdr.num_subvectors = pq->num_subvectors;
  hdr.num_centroids = pq->num_centroids;
  hdr.payload_bytes = (uint64_t)payload_bytes;
  hdr.payload_crc32 = nbase_crc32(pq->codebooks, payload_bytes);
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nbq_header_t, header_crc32));

  nbase_io_chunk_t chunk = {pq->codebooks, payload_bytes};
  return nbase_write_file_atomic(file_path, &hdr, sizeof(hdr), &chunk, 1);
}

nbase_status_t nbase_pq_load(const char *file_path, nbase_pq_t **out_pq) {
  if (!file_path || !out_pq) {
    return NBASE_ERR_INVALID_ARG;
  }
  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    return (errno == ENOENT) ? NBASE_ERR_NOT_FOUND : NBASE_ERR_IO;
  }

  struct stat st_buf;
  if (fstat(fd, &st_buf) != 0) {
    close(fd);
    return NBASE_ERR_IO;
  }
  if (st_buf.st_size < (off_t)sizeof(nbq_header_t)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;

  nbq_header_t hdr;
  nbase_status_t st = nbase_read_exact(fd, &hdr, sizeof(hdr));
  if (st != NBASE_OK) {
    close(fd);
    return st;
  }
  if (memcmp(hdr.magic, NBASE_NBQ_MAGIC, 8) != 0 || hdr.version != NBASE_NBQ_VERSION) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.header_crc32 != nbase_crc32(&hdr, offsetof(nbq_header_t, header_crc32))) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.dim == 0U || hdr.dim > NBASE_MAX_DIM || hdr.num_subvectors == 0U || (hdr.dim % hdr.num_subvectors) != 0U || hdr.num_centroids == 0U || hdr.num_centroids > 256U) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint32_t sub_dim = hdr.dim / hdr.num_subvectors;
  uint64_t expected_bytes = (uint64_t)hdr.num_subvectors * (uint64_t)hdr.num_centroids * (uint64_t)sub_dim * sizeof(float);
  if (hdr.payload_bytes != expected_bytes || file_size != sizeof(nbq_header_t) + expected_bytes || expected_bytes > (uint64_t)SIZE_MAX) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  nbase_pq_t *pq = calloc(1, sizeof(*pq));
  if (!pq) {
    close(fd);
    return NBASE_ERR_NOMEM;
  }
  pq->dim = hdr.dim;
  pq->num_subvectors = hdr.num_subvectors;
  pq->sub_dim = sub_dim;
  pq->num_centroids = hdr.num_centroids;
  pq->codebooks = malloc((size_t)expected_bytes);
  if (!pq->codebooks) {
    close(fd);
    nbase_pq_destroy(pq);
    return NBASE_ERR_NOMEM;
  }

  st = nbase_read_exact(fd, pq->codebooks, (size_t)expected_bytes);
  close(fd);
  if (st != NBASE_OK) {
    nbase_pq_destroy(pq);
    return st;
  }
  if (nbase_crc32(pq->codebooks, (size_t)expected_bytes) != hdr.payload_crc32 || !nbase_vec_is_finite(pq->codebooks, hdr.num_subvectors * hdr.num_centroids * sub_dim)) {
    nbase_pq_destroy(pq);
    return NBASE_ERR_CORRUPT;
  }

  *out_pq = pq;
  return NBASE_OK;
}

const float *nbase_pq_codebooks(const nbase_pq_t *pq) {
  return pq ? pq->codebooks : NULL;
}

nbase_status_t nbase_pq_create_from_codebooks(
    uint32_t dim,
    uint32_t num_subvectors,
    uint32_t num_centroids,
    const float *codebooks,
    nbase_pq_t **out_pq) {
  if (!codebooks || !out_pq || dim == 0U || dim > NBASE_MAX_DIM || num_subvectors == 0U || (dim % num_subvectors) != 0U || num_centroids == 0U || num_centroids > 256U) {
    return NBASE_ERR_INVALID_ARG;
  }
  uint32_t sub_dim = dim / num_subvectors;
  size_t floats = (size_t)num_subvectors * (size_t)num_centroids * (size_t)sub_dim;
  if (floats > SIZE_MAX / sizeof(float)) {
    return NBASE_ERR_NOMEM;
  }
  if (!nbase_vec_is_finite(codebooks, num_subvectors * num_centroids * sub_dim)) {
    return NBASE_ERR_CORRUPT;
  }
  nbase_pq_t *pq = calloc(1, sizeof(*pq));
  if (!pq) {
    return NBASE_ERR_NOMEM;
  }
  pq->dim = dim;
  pq->num_subvectors = num_subvectors;
  pq->sub_dim = sub_dim;
  pq->num_centroids = num_centroids;
  pq->codebooks = malloc(floats * sizeof(float));
  if (!pq->codebooks) {
    free(pq);
    return NBASE_ERR_NOMEM;
  }
  memcpy(pq->codebooks, codebooks, floats * sizeof(float));
  *out_pq = pq;
  return NBASE_OK;
}
