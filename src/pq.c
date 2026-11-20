#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
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
  if (codebook_floats > SIZE_MAX / sizeof(float) || (size_t)count > SIZE_MAX / (size_t)sub_dim) {
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
  float *slice = malloc((size_t)count * (size_t)sub_dim * sizeof(float));
  if (!pq->codebooks || !slice) {
    free(slice);
    nbase_pq_destroy(pq);
    return NBASE_ERR_NOMEM;
  }

  for (uint32_t m = 0U; m < num_subvectors; m++) {
    uint32_t offset = m * sub_dim;
    for (uint32_t i = 0U; i < count; i++) {
      memcpy(slice + (size_t)i * (size_t)sub_dim, vectors + (size_t)i * (size_t)dim + offset, (size_t)sub_dim * sizeof(float));
    }
    float *dst_cb = pq->codebooks + (size_t)m * (size_t)num_centroids * (size_t)sub_dim;
    nbase_status_t st = nbase_kmeans_train(slice, count, sub_dim, num_centroids, max_iters, seed + (uint64_t)m, dst_cb, NULL);
    if (st != NBASE_OK) {
      free(slice);
      nbase_pq_destroy(pq);
      return st;
    }
  }

  free(slice);
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

  nbase_dist_fn sq_l2 = nbase_get_distance_fn(NBASE_METRIC_SQUARED_EUCLIDEAN);
  for (uint32_t i = 0U; i < count; i++) {
    const float *vec = vectors + (size_t)i * (size_t)pq->dim;
    uint8_t *code = out_codes + (size_t)i * (size_t)pq->num_subvectors;
    for (uint32_t m = 0U; m < pq->num_subvectors; m++) {
      const float *sub = vec + (size_t)m * (size_t)pq->sub_dim;
      const float *cb = pq->codebooks + (size_t)m * (size_t)pq->num_centroids * (size_t)pq->sub_dim;
      uint32_t best_c = 0U;
      float best_d = FLT_MAX;
      for (uint32_t c = 0U; c < pq->num_centroids; c++) {
        float d = sq_l2(sub, cb + (size_t)c * (size_t)pq->sub_dim, pq->sub_dim);
        if (d < best_d) {
          best_d = d;
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
  nbase_dist_fn sq_l2 = nbase_get_distance_fn(NBASE_METRIC_SQUARED_EUCLIDEAN);
  for (uint32_t m = 0U; m < pq->num_subvectors; m++) {
    const float *q_sub = query + (size_t)m * (size_t)pq->sub_dim;
    const float *cb = pq->codebooks + (size_t)m * (size_t)pq->num_centroids * (size_t)pq->sub_dim;
    float *lut_row = out_lut + (size_t)m * (size_t)pq->num_centroids;
    for (uint32_t c = 0U; c < pq->num_centroids; c++) {
      lut_row[c] = sq_l2(q_sub, cb + (size_t)c * (size_t)pq->sub_dim, pq->sub_dim);
    }
  }
  return NBASE_OK;
}

float nbase_pq_adc_distance(const nbase_pq_t *pq, const float *lut, const uint8_t *code) {
  if (!pq || !lut || !code) {
    return 0.0f;
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
