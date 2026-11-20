#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct ivf_list {
  uint64_t *ids;
  uint8_t *codes;
  uint32_t len;
  uint32_t cap;
} ivf_list_t;

struct nbase_ivfpq {
  uint32_t dim;
  nbase_metric_t metric;
  uint32_t nlist;
  uint32_t num_subvectors;
  uint32_t num_centroids;
  uint64_t count;
  float *coarse_centroids;
  nbase_pq_t *pq;
  ivf_list_t *lists;
};

typedef struct probe_cand {
  uint32_t list_id;
  float dist;
} probe_cand_t;

static int cmp_probe_asc(const void *pa, const void *pb) {
  const probe_cand_t *a = (const probe_cand_t *)pa;
  const probe_cand_t *b = (const probe_cand_t *)pb;
  if (a->dist < b->dist) {
    return -1;
  }
  if (a->dist > b->dist) {
    return 1;
  }
  if (a->list_id < b->list_id) {
    return -1;
  }
  if (a->list_id > b->list_id) {
    return 1;
  }
  return 0;
}

static bool ivf_metric_supported(nbase_metric_t metric) {
  return metric == NBASE_METRIC_EUCLIDEAN || metric == NBASE_METRIC_COSINE || metric == NBASE_METRIC_INNER_PRODUCT;
}

static float vec_l2_sq(const float *a, const float *b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0U; i < dim; i++) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

static float vec_dot(const float *a, const float *b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0U; i < dim; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

static void normalize_vec(const float *src, float *dst, uint32_t dim) {
  float norm2 = 0.0f;
  for (uint32_t i = 0U; i < dim; i++) {
    norm2 += src[i] * src[i];
  }
  if (norm2 > 0.0f) {
    float inv = 1.0f / sqrtf(norm2);
    for (uint32_t i = 0U; i < dim; i++) {
      dst[i] = src[i] * inv;
    }
  } else {
    memset(dst, 0, (size_t)dim * sizeof(float));
  }
}

static uint32_t find_nearest_coarse(const float *coarse, uint32_t nlist, uint32_t dim, const float *vec) {
  uint32_t best = 0U;
  float best_d = FLT_MAX;
  for (uint32_t c = 0U; c < nlist; c++) {
    float d = vec_l2_sq(vec, coarse + (size_t)c * (size_t)dim, dim);
    if (d < best_d) {
      best_d = d;
      best = c;
    }
  }
  return best;
}

static void build_dot_lut(const nbase_pq_t *pq, const float *query, float *out_lut) {
  uint32_t m_sub = nbase_pq_num_subvectors(pq);
  uint32_t k_cent = nbase_pq_num_centroids(pq);
  uint32_t sub_dim = nbase_pq_dim(pq) / m_sub;
  const float *codebooks = nbase_pq_codebooks(pq);

  for (uint32_t m = 0U; m < m_sub; m++) {
    const float *q_sub = query + (size_t)m * (size_t)sub_dim;
    const float *cb_m = codebooks + (size_t)m * (size_t)k_cent * (size_t)sub_dim;
    for (uint32_t c = 0U; c < k_cent; c++) {
      const float *cent = cb_m + (size_t)c * (size_t)sub_dim;
      float s = 0.0f;
      for (uint32_t j = 0U; j < sub_dim; j++) {
        s += q_sub[j] * cent[j];
      }
      out_lut[(size_t)m * (size_t)k_cent + c] = s;
    }
  }
}

typedef struct res_maxheap {
  nbase_result_t *data;
  uint32_t len;
  uint32_t cap;
} res_maxheap_t;

static bool res_higher(nbase_result_t a, nbase_result_t b) {
  return a.distance > b.distance || (a.distance == b.distance && a.id > b.id);
}

static void res_heap_push_bounded(res_maxheap_t *h, uint64_t id, float dist) {
  nbase_result_t item;
  item.id = id;
  item.distance = dist;
  if (h->len < h->cap) {
    uint32_t idx = h->len++;
    while (idx > 0U) {
      uint32_t parent = (idx - 1U) / 2U;
      if (!res_higher(item, h->data[parent])) {
        break;
      }
      h->data[idx] = h->data[parent];
      idx = parent;
    }
    h->data[idx] = item;
  } else if (h->len > 0U && res_higher(h->data[0], item)) {
    uint32_t idx = 0U;
    while (true) {
      uint32_t left = 2U * idx + 1U;
      if (left >= h->len) {
        break;
      }
      uint32_t right = left + 1U;
      uint32_t best = left;
      if (right < h->len && res_higher(h->data[right], h->data[left])) {
        best = right;
      }
      if (!res_higher(h->data[best], item)) {
        break;
      }
      h->data[idx] = h->data[best];
      idx = best;
    }
    h->data[idx] = item;
  }
}

nbase_ivfpq_config_t nbase_ivfpq_config_default(uint32_t dim, uint32_t nlist, uint32_t num_subvectors) {
  nbase_ivfpq_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.dim = dim;
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.nlist = nlist;
  cfg.num_subvectors = num_subvectors;
  cfg.num_centroids = 256U;
  cfg.max_iters = 20U;
  cfg.seed = 0x4956465051534545ULL;
  return cfg;
}

nbase_status_t nbase_ivfpq_train(
    const nbase_ivfpq_config_t *cfg,
    const float *train_vectors,
    uint32_t train_count,
    nbase_ivfpq_t **out_index) {
  if (!cfg || !train_vectors || !out_index) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_index = NULL;

  if (!nbase_metric_is_valid(cfg->metric)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (!ivf_metric_supported(cfg->metric)) {
    return NBASE_ERR_UNSUPPORTED;
  }
  if (cfg->dim == 0U || cfg->dim > NBASE_MAX_DIM || cfg->nlist == 0U || train_count < cfg->nlist ||
      cfg->num_subvectors == 0U || (cfg->dim % cfg->num_subvectors) != 0U ||
      cfg->num_centroids == 0U || cfg->num_centroids > 256U || train_count < cfg->num_centroids) {
    return NBASE_ERR_INVALID_ARG;
  }

  size_t vec_floats = (size_t)train_count * (size_t)cfg->dim;
  size_t coarse_floats = (size_t)cfg->nlist * (size_t)cfg->dim;
  if (vec_floats > SIZE_MAX / sizeof(float) || coarse_floats > SIZE_MAX / sizeof(float)) {
    return NBASE_ERR_NOMEM;
  }
  for (uint32_t i = 0U; i < train_count; i++) {
    if (!nbase_vec_is_finite(train_vectors + (size_t)i * (size_t)cfg->dim, cfg->dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  float *work = malloc(vec_floats * sizeof(float));
  float *coarse = malloc(coarse_floats * sizeof(float));
  uint32_t *assignments = malloc((size_t)train_count * sizeof(uint32_t));
  if (!work || !coarse || !assignments) {
    free(work);
    free(coarse);
    free(assignments);
    return NBASE_ERR_NOMEM;
  }

  if (cfg->metric == NBASE_METRIC_COSINE) {
    for (uint32_t i = 0U; i < train_count; i++) {
      normalize_vec(train_vectors + (size_t)i * (size_t)cfg->dim, work + (size_t)i * (size_t)cfg->dim, cfg->dim);
    }
  } else {
    memcpy(work, train_vectors, vec_floats * sizeof(float));
  }

  uint32_t iters = cfg->max_iters == 0U ? 20U : cfg->max_iters;
  uint64_t seed = cfg->seed == 0ULL ? 0x4956465051534545ULL : cfg->seed;
  nbase_status_t st = nbase_kmeans_train(work, train_count, cfg->dim, cfg->nlist, iters, seed, coarse, assignments);
  if (st != NBASE_OK) {
    free(work);
    free(coarse);
    free(assignments);
    return st;
  }

  /* Replace work in-place with residual vectors r = x - coarse[assignments[i]]. */
  for (uint32_t i = 0U; i < train_count; i++) {
    float *r = work + (size_t)i * (size_t)cfg->dim;
    const float *c = coarse + (size_t)assignments[i] * (size_t)cfg->dim;
    for (uint32_t d = 0U; d < cfg->dim; d++) {
      r[d] -= c[d];
    }
  }
  free(assignments);

  nbase_pq_t *pq = NULL;
  st = nbase_pq_train(work, train_count, cfg->dim, cfg->num_subvectors, cfg->num_centroids, iters, seed ^ 0x9E3779B97F4A7C15ULL, &pq);
  free(work);
  if (st != NBASE_OK) {
    free(coarse);
    return st;
  }

  nbase_ivfpq_t *index = calloc(1, sizeof(*index));
  ivf_list_t *lists = calloc((size_t)cfg->nlist, sizeof(ivf_list_t));
  if (!index || !lists) {
    free(index);
    free(lists);
    free(coarse);
    nbase_pq_destroy(pq);
    return NBASE_ERR_NOMEM;
  }

  index->dim = cfg->dim;
  index->metric = cfg->metric;
  index->nlist = cfg->nlist;
  index->num_subvectors = cfg->num_subvectors;
  index->num_centroids = cfg->num_centroids;
  index->count = 0ULL;
  index->coarse_centroids = coarse;
  index->pq = pq;
  index->lists = lists;

  *out_index = index;
  return NBASE_OK;
}

nbase_status_t nbase_ivfpq_add(
    nbase_ivfpq_t *index,
    const uint64_t *ids,
    const float *vectors,
    uint32_t count) {
  if (!index || (count > 0U && (!ids || !vectors))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)index->dim, index->dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  float *vec_buf = malloc((size_t)index->dim * sizeof(float));
  float *res_buf = malloc((size_t)index->dim * sizeof(float));
  uint8_t *code_buf = malloc((size_t)index->num_subvectors);
  if (!vec_buf || !res_buf || !code_buf) {
    free(vec_buf);
    free(res_buf);
    free(code_buf);
    return NBASE_ERR_NOMEM;
  }

  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < count; i++) {
    const float *src = vectors + (size_t)i * (size_t)index->dim;
    if (index->metric == NBASE_METRIC_COSINE) {
      normalize_vec(src, vec_buf, index->dim);
    } else {
      memcpy(vec_buf, src, (size_t)index->dim * sizeof(float));
    }

    uint32_t list_id = find_nearest_coarse(index->coarse_centroids, index->nlist, index->dim, vec_buf);
    const float *c = index->coarse_centroids + (size_t)list_id * (size_t)index->dim;
    for (uint32_t d = 0U; d < index->dim; d++) {
      res_buf[d] = vec_buf[d] - c[d];
    }

    st = nbase_pq_encode(index->pq, res_buf, 1U, code_buf);
    if (st != NBASE_OK) {
      break;
    }

    ivf_list_t *list = &index->lists[list_id];
    if (list->len == list->cap) {
      uint32_t new_cap = list->cap == 0U ? 16U : list->cap * 2U;
      uint64_t *next_ids = realloc(list->ids, (size_t)new_cap * sizeof(uint64_t));
      uint8_t *next_codes = realloc(list->codes, (size_t)new_cap * (size_t)index->num_subvectors);
      if (!next_ids || !next_codes) {
        if (next_ids) {
          list->ids = next_ids;
        }
        if (next_codes) {
          list->codes = next_codes;
        }
        st = NBASE_ERR_NOMEM;
        break;
      }
      list->ids = next_ids;
      list->codes = next_codes;
      list->cap = new_cap;
    }

    list->ids[list->len] = ids[i];
    memcpy(list->codes + (size_t)list->len * (size_t)index->num_subvectors, code_buf, index->num_subvectors);
    list->len++;
    index->count++;
  }

  free(vec_buf);
  free(res_buf);
  free(code_buf);
  return st;
}

nbase_status_t nbase_ivfpq_search(
    const nbase_ivfpq_t *index,
    const float *query,
    uint32_t k,
    uint32_t nprobe,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!index || !query || !out_count || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  if (!nbase_vec_is_finite(query, index->dim)) {
    return NBASE_ERR_INVALID_VECTOR;
  }
  if (k == 0U || index->count == 0ULL) {
    return NBASE_OK;
  }

  uint32_t eff_probe = (nprobe == 0U) ? 1U : ((nprobe > index->nlist) ? index->nlist : nprobe);

  float *q_buf = malloc((size_t)index->dim * sizeof(float));
  float *q_res = malloc((size_t)index->dim * sizeof(float));
  probe_cand_t *probes = malloc((size_t)index->nlist * sizeof(probe_cand_t));
  float *lut = malloc((size_t)index->num_subvectors * (size_t)index->num_centroids * sizeof(float));
  nbase_result_t *heap_buf = malloc((size_t)k * sizeof(nbase_result_t));
  if (!q_buf || !q_res || !probes || !lut || !heap_buf) {
    free(q_buf);
    free(q_res);
    free(probes);
    free(lut);
    free(heap_buf);
    return NBASE_ERR_NOMEM;
  }

  if (index->metric == NBASE_METRIC_COSINE) {
    normalize_vec(query, q_buf, index->dim);
  } else {
    memcpy(q_buf, query, (size_t)index->dim * sizeof(float));
  }

  for (uint32_t c = 0U; c < index->nlist; c++) {
    const float *cent = index->coarse_centroids + (size_t)c * (size_t)index->dim;
    probes[c].list_id = c;
    if (index->metric == NBASE_METRIC_INNER_PRODUCT) {
      probes[c].dist = -vec_dot(q_buf, cent, index->dim);
    } else {
      probes[c].dist = vec_l2_sq(q_buf, cent, index->dim);
    }
  }
  qsort(probes, index->nlist, sizeof(probe_cand_t), cmp_probe_asc);

  res_maxheap_t heap = {heap_buf, 0U, k};
  nbase_status_t st = NBASE_OK;

  if (index->metric == NBASE_METRIC_INNER_PRODUCT) {
    build_dot_lut(index->pq, q_buf, lut);
  }

  for (uint32_t p = 0U; p < eff_probe && st == NBASE_OK; p++) {
    uint32_t list_id = probes[p].list_id;
    const ivf_list_t *list = &index->lists[list_id];
    if (list->len == 0U) {
      continue;
    }

    const float *cent = index->coarse_centroids + (size_t)list_id * (size_t)index->dim;
    if (index->metric == NBASE_METRIC_INNER_PRODUCT) {
      float base_dot = vec_dot(q_buf, cent, index->dim);
      for (uint32_t j = 0U; j < list->len; j++) {
        float res_dot = nbase_pq_adc_distance(index->pq, lut, list->codes + (size_t)j * (size_t)index->num_subvectors);
        res_heap_push_bounded(&heap, list->ids[j], -(base_dot + res_dot));
      }
    } else {
      for (uint32_t d = 0U; d < index->dim; d++) {
        q_res[d] = q_buf[d] - cent[d];
      }
      st = nbase_pq_build_lut(index->pq, q_res, lut);
      if (st != NBASE_OK) {
        break;
      }
      for (uint32_t j = 0U; j < list->len; j++) {
        float l2_sq = nbase_pq_adc_distance(index->pq, lut, list->codes + (size_t)j * (size_t)index->num_subvectors);
        float d = (index->metric == NBASE_METRIC_COSINE) ? (0.5f * fmaxf(0.0f, l2_sq)) : sqrtf(fmaxf(0.0f, l2_sq));
        res_heap_push_bounded(&heap, list->ids[j], d);
      }
    }
  }

  if (st == NBASE_OK && heap.len > 0U) {
    qsort(heap.data, heap.len, sizeof(nbase_result_t), nbase_cmp_result_asc);
    memcpy(out_results, heap.data, (size_t)heap.len * sizeof(nbase_result_t));
    *out_count = heap.len;
  }

  free(q_buf);
  free(q_res);
  free(probes);
  free(lut);
  free(heap_buf);
  return st;
}

nbase_status_t nbase_ivfpq_save(const nbase_ivfpq_t *index, const char *file_path) {
  if (!index || !file_path || file_path[0] == '\0') {
    return NBASE_ERR_INVALID_ARG;
  }

  uint32_t sub_dim = index->dim / index->num_subvectors;
  size_t coarse_bytes = (size_t)index->nlist * (size_t)index->dim * sizeof(float);
  size_t codebook_bytes = (size_t)index->num_subvectors * (size_t)index->num_centroids * (size_t)sub_dim * sizeof(float);
  size_t lens_bytes = (size_t)index->nlist * sizeof(uint32_t);
  size_t max_chunks = 3U + (size_t)index->nlist * 2U;

  uint32_t *list_lens = malloc(lens_bytes);
  nbase_io_chunk_t *chunks = calloc(max_chunks, sizeof(*chunks));
  if (!list_lens || !chunks) {
    free(list_lens);
    free(chunks);
    return NBASE_ERR_NOMEM;
  }

  const float *codebooks = nbase_pq_codebooks(index->pq);
  size_t chunk_cnt = 0U;
  chunks[chunk_cnt++] = (nbase_io_chunk_t){index->coarse_centroids, coarse_bytes};
  chunks[chunk_cnt++] = (nbase_io_chunk_t){codebooks, codebook_bytes};
  chunks[chunk_cnt++] = (nbase_io_chunk_t){list_lens, lens_bytes};

  uint64_t payload_bytes = (uint64_t)coarse_bytes + (uint64_t)codebook_bytes + (uint64_t)lens_bytes;
  for (uint32_t c = 0U; c < index->nlist; c++) {
    uint32_t len = index->lists[c].len;
    list_lens[c] = len;
    if (len > 0U) {
      size_t id_bytes = (size_t)len * sizeof(uint64_t);
      size_t code_bytes = (size_t)len * (size_t)index->num_subvectors;
      chunks[chunk_cnt++] = (nbase_io_chunk_t){index->lists[c].ids, id_bytes};
      chunks[chunk_cnt++] = (nbase_io_chunk_t){index->lists[c].codes, code_bytes};
      payload_bytes += (uint64_t)id_bytes + (uint64_t)code_bytes;
    }
  }

  uint32_t payload_crc = 0U;
  for (size_t i = 0U; i < chunk_cnt; i++) {
    if (chunks[i].len > 0U && chunks[i].ptr != NULL) {
      payload_crc = nbase_crc32_update(payload_crc, chunks[i].ptr, chunks[i].len);
    }
  }

  nbiv_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBIV_MAGIC, 8);
  hdr.version = NBASE_NBIV_VERSION;
  hdr.dim = index->dim;
  hdr.metric = (uint32_t)index->metric;
  hdr.nlist = index->nlist;
  hdr.num_subvectors = index->num_subvectors;
  hdr.num_centroids = index->num_centroids;
  hdr.count = index->count;
  hdr.payload_bytes = payload_bytes;
  hdr.payload_crc32 = payload_crc;
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nbiv_header_t, header_crc32));

  nbase_status_t st = nbase_write_file_atomic(file_path, &hdr, sizeof(hdr), chunks, chunk_cnt);
  free(list_lens);
  free(chunks);
  return st;
}

nbase_status_t nbase_ivfpq_load(const char *file_path, nbase_ivfpq_t **out_index) {
  if (!file_path || file_path[0] == '\0' || !out_index) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_index = NULL;

  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    return errno == ENOENT ? NBASE_ERR_NOT_FOUND : NBASE_ERR_IO;
  }

  struct stat st_buf;
  if (fstat(fd, &st_buf) != 0) {
    close(fd);
    return NBASE_ERR_IO;
  }
  if (st_buf.st_size < (off_t)sizeof(nbiv_header_t)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;

  nbiv_header_t hdr;
  nbase_status_t st = nbase_read_exact(fd, &hdr, sizeof(hdr));
  if (st != NBASE_OK) {
    close(fd);
    return st;
  }
  if (memcmp(hdr.magic, NBASE_NBIV_MAGIC, 8) != 0 || hdr.version != NBASE_NBIV_VERSION) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.header_crc32 != nbase_crc32(&hdr, offsetof(nbiv_header_t, header_crc32))) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  nbase_metric_t metric = (nbase_metric_t)hdr.metric;
  if (!ivf_metric_supported(metric) || hdr.dim == 0U || hdr.dim > NBASE_MAX_DIM || hdr.nlist == 0U ||
      hdr.num_subvectors == 0U || (hdr.dim % hdr.num_subvectors) != 0U || hdr.num_centroids == 0U || hdr.num_centroids > 256U) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (file_size != sizeof(nbiv_header_t) + hdr.payload_bytes || hdr.payload_bytes > (uint64_t)SIZE_MAX) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint8_t *payload = malloc((size_t)hdr.payload_bytes);
  if (!payload) {
    close(fd);
    return NBASE_ERR_NOMEM;
  }
  st = nbase_read_exact(fd, payload, (size_t)hdr.payload_bytes);
  close(fd);
  if (st != NBASE_OK) {
    free(payload);
    return st;
  }
  if (nbase_crc32(payload, (size_t)hdr.payload_bytes) != hdr.payload_crc32) {
    free(payload);
    return NBASE_ERR_CORRUPT;
  }

  uint32_t sub_dim = hdr.dim / hdr.num_subvectors;
  uint64_t lens_bytes_u64 = (uint64_t)hdr.nlist * (uint64_t)sizeof(uint32_t);
  uint64_t coarse_bytes_u64 = (uint64_t)hdr.nlist * (uint64_t)hdr.dim * (uint64_t)sizeof(float);
  uint64_t codebook_bytes_u64 = (uint64_t)hdr.num_subvectors * (uint64_t)hdr.num_centroids * (uint64_t)sub_dim * (uint64_t)sizeof(float);
  if ((uint64_t)hdr.nlist > hdr.payload_bytes / (uint64_t)sizeof(uint32_t) ||
      (uint64_t)hdr.nlist > hdr.payload_bytes / ((uint64_t)hdr.dim * (uint64_t)sizeof(float)) ||
      coarse_bytes_u64 > hdr.payload_bytes ||
      codebook_bytes_u64 > hdr.payload_bytes - coarse_bytes_u64 ||
      lens_bytes_u64 > hdr.payload_bytes - coarse_bytes_u64 - codebook_bytes_u64) {
    free(payload);
    return NBASE_ERR_CORRUPT;
  }

  size_t coarse_bytes = (size_t)coarse_bytes_u64;
  size_t codebook_bytes = (size_t)codebook_bytes_u64;
  size_t lens_bytes = (size_t)lens_bytes_u64;

  size_t off = 0U;
  float *coarse = malloc(coarse_bytes);
  float *codebooks_tmp = malloc(codebook_bytes);
  uint32_t *list_lens = malloc(lens_bytes);
  if (!coarse || !codebooks_tmp || !list_lens) {
    free(coarse);
    free(codebooks_tmp);
    free(list_lens);
    free(payload);
    return NBASE_ERR_NOMEM;
  }
  memcpy(coarse, payload + off, coarse_bytes);
  off += coarse_bytes;
  memcpy(codebooks_tmp, payload + off, codebook_bytes);
  off += codebook_bytes;

  nbase_pq_t *pq = NULL;
  st = nbase_pq_create_from_codebooks(hdr.dim, hdr.num_subvectors, hdr.num_centroids, codebooks_tmp, &pq);
  free(codebooks_tmp);
  bool coarse_ok = true;
  for (uint32_t c = 0U; c < hdr.nlist && coarse_ok; c++) {
    if (!nbase_vec_is_finite(coarse + (size_t)c * (size_t)hdr.dim, hdr.dim)) {
      coarse_ok = false;
    }
  }
  if (st != NBASE_OK || !coarse_ok) {
    nbase_pq_destroy(pq);
    free(list_lens);
    free(coarse);
    free(payload);
    return st == NBASE_ERR_NOMEM ? NBASE_ERR_NOMEM : NBASE_ERR_CORRUPT;
  }

  memcpy(list_lens, payload + off, lens_bytes);
  off += lens_bytes;

  uint64_t total_len = 0ULL;
  uint64_t expected_list_bytes = 0ULL;
  for (uint32_t c = 0U; c < hdr.nlist; c++) {
    total_len += list_lens[c];
    expected_list_bytes += (uint64_t)list_lens[c] * (sizeof(uint64_t) + (size_t)hdr.num_subvectors);
  }
  if (total_len != hdr.count || (uint64_t)off + expected_list_bytes != hdr.payload_bytes) {
    nbase_pq_destroy(pq);
    free(list_lens);
    free(coarse);
    free(payload);
    return NBASE_ERR_CORRUPT;
  }

  nbase_ivfpq_t *index = calloc(1, sizeof(*index));
  ivf_list_t *lists = calloc((size_t)hdr.nlist, sizeof(ivf_list_t));
  if (!index || !lists) {
    free(index);
    free(lists);
    nbase_pq_destroy(pq);
    free(list_lens);
    free(coarse);
    free(payload);
    return NBASE_ERR_NOMEM;
  }

  index->dim = hdr.dim;
  index->metric = metric;
  index->nlist = hdr.nlist;
  index->num_subvectors = hdr.num_subvectors;
  index->num_centroids = hdr.num_centroids;
  index->count = hdr.count;
  index->coarse_centroids = coarse;
  index->pq = pq;
  index->lists = lists;

  for (uint32_t c = 0U; c < hdr.nlist; c++) {
    uint32_t len = list_lens[c];
    if (len == 0U) {
      continue;
    }
    size_t id_bytes = (size_t)len * sizeof(uint64_t);
    size_t code_bytes = (size_t)len * (size_t)hdr.num_subvectors;
    lists[c].ids = malloc(id_bytes);
    lists[c].codes = malloc(code_bytes);
    if (!lists[c].ids || !lists[c].codes) {
      free(list_lens);
      free(payload);
      nbase_ivfpq_destroy(index);
      return NBASE_ERR_NOMEM;
    }
    lists[c].len = len;
    lists[c].cap = len;
    memcpy(lists[c].ids, payload + off, id_bytes);
    off += id_bytes;
    memcpy(lists[c].codes, payload + off, code_bytes);
    off += code_bytes;
  }

  free(list_lens);
  free(payload);
  *out_index = index;
  return NBASE_OK;
}

nbase_status_t nbase_ivfpq_stats(const nbase_ivfpq_t *index, nbase_ivfpq_stats_t *out_stats) {
  if (!index || !out_stats) {
    return NBASE_ERR_INVALID_ARG;
  }
  out_stats->dim = index->dim;
  out_stats->metric = index->metric;
  out_stats->nlist = index->nlist;
  out_stats->num_subvectors = index->num_subvectors;
  out_stats->num_centroids = index->num_centroids;
  out_stats->count = index->count;
  return NBASE_OK;
}

void nbase_ivfpq_destroy(nbase_ivfpq_t *index) {
  if (!index) {
    return;
  }
  if (index->lists) {
    for (uint32_t c = 0U; c < index->nlist; c++) {
      free(index->lists[c].ids);
      free(index->lists[c].codes);
    }
    free(index->lists);
  }
  nbase_pq_destroy(index->pq);
  free(index->coarse_centroids);
  free(index);
}
