#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct nbase_bq {
  uint32_t dim;
  uint32_t words_per_vec;
  float *thresholds;
};

nbase_status_t nbase_bq_train(
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    nbase_bq_t **out_bq) {
  if (!vectors || !out_bq || count == 0U || dim == 0U || dim > NBASE_MAX_DIM) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_bq = NULL;

  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)dim, dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  nbase_bq_t *bq = calloc(1, sizeof(*bq));
  float *thresholds = malloc((size_t)dim * sizeof(float));
  double *sums = calloc((size_t)dim, sizeof(double));
  if (!bq || !thresholds || !sums) {
    free(bq);
    free(thresholds);
    free(sums);
    return NBASE_ERR_NOMEM;
  }

  for (uint32_t i = 0U; i < count; i++) {
    const float *v = vectors + (size_t)i * (size_t)dim;
    for (uint32_t d = 0U; d < dim; d++) {
      sums[d] += (double)v[d];
    }
  }
  double inv_n = 1.0 / (double)count;
  for (uint32_t d = 0U; d < dim; d++) {
    thresholds[d] = (float)(sums[d] * inv_n);
  }
  free(sums);

  bq->dim = dim;
  bq->words_per_vec = (dim + 63U) / 64U;
  bq->thresholds = thresholds;
  *out_bq = bq;
  return NBASE_OK;
}

void nbase_bq_destroy(nbase_bq_t *bq) {
  if (!bq) {
    return;
  }
  free(bq->thresholds);
  free(bq);
}

uint32_t nbase_bq_dim(const nbase_bq_t *bq) {
  return bq ? bq->dim : 0U;
}

uint32_t nbase_bq_words_per_vec(const nbase_bq_t *bq) {
  return bq ? bq->words_per_vec : 0U;
}

nbase_status_t nbase_bq_encode(
    const nbase_bq_t *bq,
    const float *vectors,
    uint32_t count,
    uint64_t *out_words) {
  if (!bq || (count > 0U && (!vectors || !out_words))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)bq->dim, bq->dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  for (uint32_t i = 0U; i < count; i++) {
    const float *v = vectors + (size_t)i * (size_t)bq->dim;
    uint64_t *dst = out_words + (size_t)i * (size_t)bq->words_per_vec;
    memset(dst, 0, (size_t)bq->words_per_vec * sizeof(uint64_t));
    for (uint32_t d = 0U; d < bq->dim; d++) {
      if (v[d] >= bq->thresholds[d]) {
        dst[d >> 6U] |= (1ULL << (d & 63U));
      }
    }
  }
  return NBASE_OK;
}

static inline uint32_t popcnt64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return (uint32_t)__builtin_popcountll(x);
#else
  x = x - ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  return (uint32_t)((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
#endif
}

uint32_t nbase_bq_hamming(const uint64_t *a, const uint64_t *b, uint32_t num_words) {
  if (!a || !b) {
    return 0U;
  }
  uint32_t dist = 0U;
  for (uint32_t w = 0U; w < num_words; w++) {
    dist += popcnt64(a[w] ^ b[w]);
  }
  return dist;
}

nbase_status_t nbase_bq_hamming_batch(
    const uint64_t *query_words,
    const uint64_t *codes,
    uint32_t count,
    uint32_t num_words,
    uint32_t *out_distances) {
  if (count > 0U && (!out_distances || (num_words > 0U && (!query_words || !codes)))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }
  if (num_words == 0U) {
    memset(out_distances, 0, (size_t)count * sizeof(uint32_t));
    return NBASE_OK;
  }
  if (num_words == 1U) {
    uint64_t q0 = query_words[0];
    for (uint32_t i = 0U; i < count; i++) {
      out_distances[i] = popcnt64(q0 ^ codes[i]);
    }
    return NBASE_OK;
  }
  if (num_words == 2U) {
    uint64_t q0 = query_words[0];
    uint64_t q1 = query_words[1];
    for (uint32_t i = 0U; i < count; i++) {
      const uint64_t *row = codes + (size_t)i * 2U;
      out_distances[i] = popcnt64(q0 ^ row[0]) + popcnt64(q1 ^ row[1]);
    }
    return NBASE_OK;
  }
  if (num_words == 4U) {
    uint64_t q0 = query_words[0];
    uint64_t q1 = query_words[1];
    uint64_t q2 = query_words[2];
    uint64_t q3 = query_words[3];
    for (uint32_t i = 0U; i < count; i++) {
      const uint64_t *row = codes + (size_t)i * 4U;
      out_distances[i] =
          popcnt64(q0 ^ row[0]) + popcnt64(q1 ^ row[1]) +
          popcnt64(q2 ^ row[2]) + popcnt64(q3 ^ row[3]);
    }
    return NBASE_OK;
  }
  for (uint32_t i = 0U; i < count; i++) {
    const uint64_t *row = codes + (size_t)i * (size_t)num_words;
    uint32_t dist = 0U;
    uint32_t w = 0U;
    for (; w + 3U < num_words; w += 4U) {
      dist += popcnt64(query_words[w] ^ row[w]) +
              popcnt64(query_words[w + 1U] ^ row[w + 1U]) +
              popcnt64(query_words[w + 2U] ^ row[w + 2U]) +
              popcnt64(query_words[w + 3U] ^ row[w + 3U]);
    }
    for (; w < num_words; w++) {
      dist += popcnt64(query_words[w] ^ row[w]);
    }
    out_distances[i] = dist;
  }
  return NBASE_OK;
}

nbase_status_t nbase_bq_save(const nbase_bq_t *bq, const char *file_path) {
  if (!bq || !file_path || file_path[0] == '\0') {
    return NBASE_ERR_INVALID_ARG;
  }

  size_t payload_bytes = (size_t)bq->dim * sizeof(float);
  nbase_io_chunk_t chunk = {bq->thresholds, payload_bytes};

  nbbq_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBBQ_MAGIC, 8);
  hdr.version = NBASE_NBBQ_VERSION;
  hdr.dim = bq->dim;
  hdr.payload_bytes = (uint64_t)payload_bytes;
  hdr.payload_crc32 = nbase_crc32(bq->thresholds, payload_bytes);
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nbbq_header_t, header_crc32));

  return nbase_write_file_atomic(file_path, &hdr, sizeof(hdr), &chunk, 1U);
}

nbase_status_t nbase_bq_load(const char *file_path, nbase_bq_t **out_bq) {
  if (!file_path || file_path[0] == '\0' || !out_bq) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_bq = NULL;

  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    return errno == ENOENT ? NBASE_ERR_NOT_FOUND : NBASE_ERR_IO;
  }

  struct stat st_buf;
  if (fstat(fd, &st_buf) != 0) {
    close(fd);
    return NBASE_ERR_IO;
  }
  if (st_buf.st_size < (off_t)sizeof(nbbq_header_t)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;

  nbbq_header_t hdr;
  nbase_status_t st = nbase_read_exact(fd, &hdr, sizeof(hdr));
  if (st != NBASE_OK) {
    close(fd);
    return st;
  }
  if (memcmp(hdr.magic, NBASE_NBBQ_MAGIC, 8) != 0 || hdr.version != NBASE_NBBQ_VERSION) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.header_crc32 != nbase_crc32(&hdr, offsetof(nbbq_header_t, header_crc32))) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.dim == 0U || hdr.dim > NBASE_MAX_DIM) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint64_t expected_bytes = (uint64_t)hdr.dim * (uint64_t)sizeof(float);
  if (hdr.payload_bytes != expected_bytes ||
      file_size != sizeof(nbbq_header_t) + expected_bytes ||
      expected_bytes > (uint64_t)SIZE_MAX) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  nbase_bq_t *bq = calloc(1, sizeof(*bq));
  float *thresholds = malloc((size_t)expected_bytes);
  if (!bq || !thresholds) {
    close(fd);
    free(bq);
    free(thresholds);
    return NBASE_ERR_NOMEM;
  }

  st = nbase_read_exact(fd, thresholds, (size_t)expected_bytes);
  close(fd);
  if (st != NBASE_OK) {
    free(bq);
    free(thresholds);
    return st;
  }
  if (nbase_crc32(thresholds, (size_t)expected_bytes) != hdr.payload_crc32 ||
      !nbase_vec_is_finite(thresholds, hdr.dim)) {
    free(bq);
    free(thresholds);
    return NBASE_ERR_CORRUPT;
  }

  bq->dim = hdr.dim;
  bq->words_per_vec = (hdr.dim + 63U) / 64U;
  bq->thresholds = thresholds;
  *out_bq = bq;
  return NBASE_OK;
}
