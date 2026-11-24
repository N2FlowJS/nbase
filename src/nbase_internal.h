#ifndef NBASE_INTERNAL_H
#define NBASE_INTERNAL_H

#include "nbase.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define NBASE_INVALID_SLOT UINT32_MAX

#define NBASE_SLOT_EMPTY 0U
#define NBASE_SLOT_ACTIVE 1U
#define NBASE_SLOT_TOMBSTONE 2U

#define NBASE_NBV_MAGIC "NBASEVC1"
#define NBASE_NBV_VERSION 1U

#define NBASE_NBI_MAGIC "NBASEHN1"
#define NBASE_NBI_VERSION 1U

#define NBASE_NBM_MAGIC "NBASEMT1"
#define NBASE_NBM_VERSION 1U

#define NBASE_NBQ_MAGIC "NBASEPQ1"
#define NBASE_NBQ_VERSION 1U

#define NBASE_NBP_MAGIC "NBASEPM1"
#define NBASE_NBP_VERSION 1U

#define NBASE_NBW_MAGIC "NBASEWL1"
#define NBASE_NBW_VERSION 1U

#define NBASE_NBIV_MAGIC "NBASEIV1"
#define NBASE_NBIV_VERSION 1U

#define NBASE_NBSQ_MAGIC "NBASESQ1"
#define NBASE_NBSQ_VERSION 1U

#define NBASE_NBLH_MAGIC "NBASELH1"
#define NBASE_NBLH_VERSION 1U

#define NBASE_NBBQ_MAGIC "NBASEBQ1"
#define NBASE_NBBQ_VERSION 1U

#define NBASE_NBBM_MAGIC "NBASEBM1"
#define NBASE_NBBM_VERSION 1U

#define NBASE_NBCL_MAGIC "NBASECL1"
#define NBASE_NBCL_VERSION 1U

typedef float (*nbase_dist_fn)(const float *a, const float *b, uint32_t dim);

typedef struct nbv_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t dim;
  uint32_t metric;
  uint32_t flags;
  uint64_t slot_count;
  uint64_t active_count;
  uint64_t vectors_offset;
  uint64_t ids_offset;
  uint64_t slot_flags_offset;
  uint64_t payload_bytes;
  uint32_t payload_crc32;
  uint32_t header_crc32;
  uint8_t reserved[48];
} nbv_header_t;

_Static_assert(sizeof(nbv_header_t) == 128, "nbv_header_t must be 128 bytes");
_Static_assert(offsetof(nbv_header_t, header_crc32) == 76, "nbv_header_t CRC offset mismatch");

typedef struct nbi_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t m;
  uint32_t ef_construction;
  uint32_t ef_search;
  int32_t max_level;
  uint32_t entry_point;
  uint64_t node_count;
  uint64_t active_nodes;
  uint64_t deleted_nodes;
  uint64_t rng_state;
  uint64_t payload_bytes;
  uint32_t payload_crc32;
  uint32_t header_crc32;
  uint8_t reserved[48];
} nbi_header_t;

_Static_assert(sizeof(nbi_header_t) == 128, "nbi_header_t must be 128 bytes");
_Static_assert(offsetof(nbi_header_t, header_crc32) == 76, "nbi_header_t CRC offset mismatch");

typedef struct nbm_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t flags;
  uint64_t record_count;
  uint64_t payload_bytes;
  uint32_t payload_crc32;
  uint32_t header_crc32;
  uint8_t reserved[88];
} nbm_header_t;

_Static_assert(sizeof(nbm_header_t) == 128, "nbm_header_t must be 128 bytes");
_Static_assert(offsetof(nbm_header_t, header_crc32) == 36, "nbm_header_t CRC offset mismatch");

typedef struct nbq_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t dim;
  uint32_t num_subvectors;
  uint32_t num_centroids;
  uint64_t payload_bytes;
  uint32_t payload_crc32;
  uint32_t header_crc32;
  uint8_t reserved[88];
} nbq_header_t;

_Static_assert(sizeof(nbq_header_t) == 128, "nbq_header_t must be 128 bytes");
_Static_assert(offsetof(nbq_header_t, header_crc32) == 36, "nbq_header_t CRC offset mismatch");

typedef struct nbp_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t dim;
  uint32_t metric;
  uint32_t partition_capacity;
  uint32_t max_active_partitions;
  uint32_t enable_hnsw;
  uint32_t hnsw_m;
  uint32_t hnsw_ef_construction;
  uint32_t hnsw_ef_search;
  uint32_t next_partition_id;
  uint32_t active_partition_id;
  uint32_t partition_count;
  uint64_t seed;
  uint64_t total_vectors;
  uint64_t payload_bytes;
  uint32_t payload_crc32;
  uint32_t header_crc32;
  uint8_t reserved[40];
} nbp_header_t;

_Static_assert(sizeof(nbp_header_t) == 128, "nbp_header_t must be 128 bytes");
_Static_assert(offsetof(nbp_header_t, header_crc32) == 84, "nbp_header_t CRC offset mismatch");

typedef struct nbw_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t flags;
  uint64_t next_seq;
  uint32_t header_crc32;
  uint8_t reserved[36];
} nbw_header_t;

_Static_assert(sizeof(nbw_header_t) == 64, "nbw_header_t must be 64 bytes");
_Static_assert(offsetof(nbw_header_t, header_crc32) == 24, "nbw_header_t CRC offset mismatch");

typedef enum nbase_wal_op {
  NBASE_WAL_OP_ADD = 1U,
  NBASE_WAL_OP_UPSERT = 2U,
  NBASE_WAL_OP_DELETE = 3U,
  NBASE_WAL_OP_SET_META = 4U,
  NBASE_WAL_OP_UPDATE_META = 5U,
  NBASE_WAL_OP_DELETE_META = 6U,
  NBASE_WAL_OP_COMPACT = 7U,
  NBASE_WAL_OP_SET_TEXT = 8U,
  NBASE_WAL_OP_DELETE_TEXT = 9U
} nbase_wal_op_t;

typedef struct nbw_frame_hdr {
  uint32_t frame_crc32;
  uint32_t payload_len;
  uint32_t op;
  uint32_t reserved;
} nbw_frame_hdr_t;

_Static_assert(sizeof(nbw_frame_hdr_t) == 16, "nbw_frame_hdr_t must be 16 bytes");

typedef struct nbiv_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t dim;
  uint32_t metric;
  uint32_t nlist;
  uint32_t num_subvectors;
  uint32_t num_centroids;
  uint64_t count;
  uint64_t payload_bytes;
  uint32_t payload_crc32;
  uint32_t header_crc32;
  uint8_t reserved[72];
} nbiv_header_t;

_Static_assert(sizeof(nbiv_header_t) == 128, "nbiv_header_t must be 128 bytes");
_Static_assert(offsetof(nbiv_header_t, header_crc32) == 52, "nbiv_header_t CRC offset mismatch");

typedef struct nbsq_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t dim;
  uint64_t payload_bytes;
  uint32_t payload_crc32;
  uint32_t header_crc32;
  uint8_t reserved[96];
} nbsq_header_t;

_Static_assert(sizeof(nbsq_header_t) == 128, "nbsq_header_t must be 128 bytes");
_Static_assert(offsetof(nbsq_header_t, header_crc32) == 28, "nbsq_header_t CRC offset mismatch");

typedef struct nblh_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t dim;
  uint32_t metric;
  uint32_t num_tables;
  uint32_t num_bits;
  float bucket_width;
  uint64_t seed;
  uint64_t active_vectors;
  uint64_t payload_bytes;
  uint32_t payload_crc32;
  uint32_t header_crc32;
  uint8_t reserved[64];
} nblh_header_t;

_Static_assert(sizeof(nblh_header_t) == 128, "nblh_header_t must be 128 bytes");
_Static_assert(offsetof(nblh_header_t, header_crc32) == 60, "nblh_header_t CRC offset mismatch");

typedef struct nbbq_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t dim;
  uint64_t payload_bytes;
  uint32_t payload_crc32;
  uint32_t header_crc32;
  uint8_t reserved[96];
} nbbq_header_t;

_Static_assert(sizeof(nbbq_header_t) == 128, "nbbq_header_t must be 128 bytes");
_Static_assert(offsetof(nbbq_header_t, header_crc32) == 28, "nbbq_header_t CRC offset mismatch");

typedef struct nbbm_header {
  uint8_t magic[8];
  uint32_t version;
  float k1;
  float b;
  uint32_t flags;
  uint64_t doc_count;
  uint64_t payload_bytes;
  uint32_t payload_crc32;
  uint32_t header_crc32;
  uint8_t reserved[80];
} nbbm_header_t;

_Static_assert(sizeof(nbbm_header_t) == 128, "nbbm_header_t must be 128 bytes");
_Static_assert(offsetof(nbbm_header_t, header_crc32) == 44, "nbbm_header_t CRC offset mismatch");

typedef struct nbcl_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t dim;
  uint32_t metric;
  uint32_t target_cluster_size;
  uint32_t max_clusters;
  float distance_threshold;
  uint32_t kmeans_max_iters;
  uint32_t cluster_count;
  uint64_t seed;
  uint64_t active_vectors;
  uint64_t payload_bytes;
  uint32_t payload_crc32;
  uint32_t header_crc32;
  uint8_t reserved[56];
} nbcl_header_t;

_Static_assert(sizeof(nbcl_header_t) == 128, "nbcl_header_t must be 128 bytes");
_Static_assert(offsetof(nbcl_header_t, header_crc32) == 68, "nbcl_header_t CRC offset mismatch");

typedef struct nbase_meta_store nbase_meta_store_t;
typedef struct nbase_wal nbase_wal_t;

typedef struct nbase_id_map {
  size_t cap;
  size_t count;
  size_t tombstones;
  uint64_t *keys;
  uint32_t *slots;
  uint8_t *states;
} nbase_id_map_t;

typedef struct nbase_store {
  uint32_t dim;
  nbase_metric_t metric;
  float *vectors;
  uint64_t *ids;
  uint8_t *slot_flags;
  uint32_t slot_count;
  uint32_t capacity;
  uint64_t active_count;
  uint64_t tombstone_count;
  nbase_id_map_t id_map;
  bool mmap_backed;
  void *mmap_base;
  size_t mmap_size;
} nbase_store_t;

typedef struct nbase_adj {
  uint32_t *neighbors;
  uint32_t len;
  uint32_t cap;
} nbase_adj_t;

typedef struct nbase_hnsw {
  uint32_t m;
  uint32_t m_max;
  uint32_t m_max0;
  uint32_t ef_construction;
  uint32_t ef_search;
  double ml;
  int32_t max_level;
  uint32_t entry_point;
  uint64_t rng_state;
  uint64_t active_nodes;
  uint64_t deleted_nodes;
  uint32_t node_count;
  uint32_t node_capacity;
  int32_t *levels;
  uint8_t *states;
  nbase_adj_t **adj;
} nbase_hnsw_t;

typedef struct nbase_heap_item {
  uint32_t slot;
  float dist;
} nbase_heap_item_t;

typedef struct nbase_heap {
  nbase_heap_item_t *data;
  size_t len;
  size_t cap;
  bool is_max;
} nbase_heap_t;

typedef struct nbase_io_chunk {
  const void *ptr;
  size_t len;
} nbase_io_chunk_t;

static inline uint64_t nbase_rng_next_u64(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static inline double nbase_rng_next_double(uint64_t *state) {
  uint64_t r = nbase_rng_next_u64(state) >> 11;
  if (r == 0U) {
    r = 1U;
  }
  return (double)r * (1.0 / 9007199254740992.0);
}

static inline bool nbase_vec_is_finite(const float *vec, uint32_t dim) {
  for (uint32_t i = 0; i < dim; i++) {
    if (!isfinite(vec[i])) {
      return false;
    }
  }
  return true;
}

static inline bool nbase_metric_is_valid(nbase_metric_t metric) {
  return metric >= NBASE_METRIC_EUCLIDEAN && metric <= NBASE_METRIC_HAMMING;
}

static inline bool nbase_heap_higher(bool is_max, nbase_heap_item_t a, nbase_heap_item_t b) {
  if (is_max) {
    return a.dist > b.dist || (a.dist == b.dist && a.slot > b.slot);
  }
  return a.dist < b.dist || (a.dist == b.dist && a.slot < b.slot);
}

static inline void nbase_heap_init(nbase_heap_t *h, bool is_max) {
  h->data = NULL;
  h->len = 0;
  h->cap = 0;
  h->is_max = is_max;
}

static inline void nbase_heap_free(nbase_heap_t *h) {
  free(h->data);
  h->data = NULL;
  h->len = 0;
  h->cap = 0;
}

static inline bool nbase_heap_push(nbase_heap_t *h, nbase_heap_item_t item) {
  if (h->len == h->cap) {
    size_t new_cap = h->cap == 0 ? 16 : h->cap * 2;
    if (new_cap > SIZE_MAX / sizeof(nbase_heap_item_t)) {
      return false;
    }
    nbase_heap_item_t *next = realloc(h->data, new_cap * sizeof(nbase_heap_item_t));
    if (!next) {
      return false;
    }
    h->data = next;
    h->cap = new_cap;
  }

  size_t idx = h->len++;
  while (idx > 0) {
    size_t parent = (idx - 1) / 2;
    if (!nbase_heap_higher(h->is_max, item, h->data[parent])) {
      break;
    }
    h->data[idx] = h->data[parent];
    idx = parent;
  }
  h->data[idx] = item;
  return true;
}

static inline nbase_heap_item_t nbase_heap_pop(nbase_heap_t *h) {
  nbase_heap_item_t root = h->data[0];
  h->len--;
  if (h->len > 0) {
    nbase_heap_item_t last = h->data[h->len];
    size_t idx = 0;
    while (true) {
      size_t left = 2 * idx + 1;
      if (left >= h->len) {
        break;
      }
      size_t right = left + 1;
      size_t best = left;
      if (right < h->len && nbase_heap_higher(h->is_max, h->data[right], h->data[left])) {
        best = right;
      }
      if (!nbase_heap_higher(h->is_max, h->data[best], last)) {
        break;
      }
      h->data[idx] = h->data[best];
      idx = best;
    }
    h->data[idx] = last;
  }
  return root;
}

static inline int nbase_cmp_result_asc(const void *pa, const void *pb) {
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

/* crc32.c */
uint32_t nbase_crc32_update(uint32_t crc, const void *data, size_t len);
uint32_t nbase_crc32(const void *data, size_t len);

/* distance.c */
nbase_dist_fn nbase_get_distance_fn(nbase_metric_t metric);
bool nbase_cpu_has_avx2_fma(void);

/* store.c */
nbase_status_t nbase_ensure_dir(const char *dir_path);
nbase_status_t nbase_write_all_fd(int fd, const void *buf, size_t len);
nbase_status_t nbase_write_file_atomic(const char *file_path, const void *header, size_t header_len, const nbase_io_chunk_t *chunks, size_t chunk_count);
nbase_status_t nbase_read_exact(int fd, void *buf, size_t len);

nbase_status_t nbase_store_create(uint32_t dim, nbase_metric_t metric, uint32_t initial_capacity, nbase_store_t **out_store);
void nbase_store_destroy(nbase_store_t *store);
nbase_status_t nbase_store_add(nbase_store_t *store, uint64_t id, const float *vec, uint32_t *out_slot);
nbase_status_t nbase_store_update_inplace(nbase_store_t *store, uint32_t slot, const float *vec);
nbase_status_t nbase_store_delete(nbase_store_t *store, uint64_t id, uint32_t *out_slot);
void nbase_store_rollback_slot(nbase_store_t *store, uint32_t slot);
nbase_status_t nbase_store_compact(nbase_store_t *store);
bool nbase_store_find_slot(const nbase_store_t *store, uint64_t id, uint32_t *out_slot);
const float *nbase_store_get_vector(const nbase_store_t *store, uint32_t slot);
nbase_status_t nbase_store_exact_search(const nbase_store_t *store, const float *query, uint32_t k, nbase_metric_t metric, nbase_filter_fn filter, void *filter_ctx, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_store_exact_search_radius(const nbase_store_t *store, const float *query, uint32_t k, float max_distance, nbase_metric_t metric, nbase_filter_fn filter, void *filter_ctx, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_store_save(const nbase_store_t *store, const char *file_path);
nbase_status_t nbase_store_load(const char *file_path, bool mmap_mode, nbase_store_t **out_store);
uint64_t nbase_store_memory_bytes(const nbase_store_t *store);

/* meta.c */
nbase_status_t nbase_meta_create(nbase_meta_store_t **out_meta);
void nbase_meta_destroy(nbase_meta_store_t *meta);
uint64_t nbase_meta_count(const nbase_meta_store_t *meta);
uint64_t nbase_meta_memory_bytes(const nbase_meta_store_t *meta);
nbase_status_t nbase_meta_validate_fields(const nbase_meta_field_t *fields, uint32_t count);
nbase_status_t nbase_meta_set(nbase_meta_store_t *meta, uint64_t id, const nbase_meta_field_t *fields, uint32_t count);
nbase_status_t nbase_meta_update(nbase_meta_store_t *meta, uint64_t id, const nbase_meta_field_t *fields, uint32_t count);
nbase_status_t nbase_meta_get(const nbase_meta_store_t *meta, uint64_t id, const char *key, nbase_meta_value_t *out_val, char *str_buf, size_t str_buf_len);
nbase_status_t nbase_meta_delete(nbase_meta_store_t *meta, uint64_t id);
nbase_status_t nbase_meta_foreach(const nbase_meta_store_t *meta, uint64_t id, nbase_meta_iter_fn cb, void *user_ctx);
bool nbase_meta_matches(const nbase_meta_store_t *meta, uint64_t id, const nbase_meta_filter_t *filter);
nbase_status_t nbase_meta_save(const nbase_meta_store_t *meta, const char *file_path);
nbase_status_t nbase_meta_load(const char *file_path, nbase_meta_store_t **out_meta);

/* hnsw.c */
nbase_status_t nbase_hnsw_create(uint32_t m, uint32_t ef_construction, uint32_t ef_search, uint64_t seed, nbase_hnsw_t **out_hnsw);
void nbase_hnsw_destroy(nbase_hnsw_t *hnsw);
nbase_status_t nbase_hnsw_insert(nbase_hnsw_t *hnsw, const nbase_store_t *store, uint32_t slot);
nbase_status_t nbase_hnsw_insert_batch_mt(nbase_hnsw_t *hnsw, const nbase_store_t *store, const uint32_t *slots, uint32_t count, uint32_t num_threads);
void nbase_hnsw_mark_deleted(nbase_hnsw_t *hnsw, uint32_t slot);
nbase_status_t nbase_hnsw_search(const nbase_hnsw_t *hnsw, const nbase_store_t *store, const float *query, uint32_t k, uint32_t ef_override, nbase_metric_t metric, nbase_filter_fn filter, void *filter_ctx, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_hnsw_save(const nbase_hnsw_t *hnsw, const char *file_path);
nbase_status_t nbase_hnsw_load(const char *file_path, nbase_hnsw_t **out_hnsw);
uint64_t nbase_hnsw_memory_bytes(const nbase_hnsw_t *hnsw);

/* bm25.c */
uint64_t nbase_bm25_memory_bytes(const nbase_bm25_t *bm25);

/* pq.c internal helpers */
const float *nbase_pq_codebooks(const nbase_pq_t *pq);
nbase_status_t nbase_pq_create_from_codebooks(uint32_t dim, uint32_t num_subvectors, uint32_t num_centroids, const float *codebooks, nbase_pq_t **out_pq);

/* wal.c */
nbase_status_t nbase_wal_open(const char *wal_path, nbase_wal_t **out_wal);
void nbase_wal_close(nbase_wal_t *wal);
nbase_status_t nbase_wal_reset(nbase_wal_t *wal);
nbase_status_t nbase_wal_append_vec(nbase_wal_t *wal, nbase_wal_op_t op, uint64_t id, const float *vec, uint32_t dim);
nbase_status_t nbase_wal_append_vec_batch(nbase_wal_t *wal, const uint64_t *ids, const float *vectors, uint32_t count, uint32_t dim);
nbase_status_t nbase_wal_append_vec_batch_op(nbase_wal_t *wal, nbase_wal_op_t op, const uint64_t *ids, const float *vectors, uint32_t count, uint32_t dim);
nbase_status_t nbase_wal_append_id(nbase_wal_t *wal, nbase_wal_op_t op, uint64_t id);
nbase_status_t nbase_wal_append_meta(nbase_wal_t *wal, nbase_wal_op_t op, uint64_t id, const nbase_meta_field_t *fields, uint32_t count);
nbase_status_t nbase_wal_append_meta_batch(nbase_wal_t *wal, nbase_wal_op_t op, const nbase_meta_batch_item_t *items, uint32_t count);
nbase_status_t nbase_wal_append_text(nbase_wal_t *wal, uint64_t id, const char *text);
nbase_status_t nbase_wal_append_text_batch(nbase_wal_t *wal, const uint64_t *ids, const char *const *texts, uint32_t count);
nbase_status_t nbase_wal_append_compact(nbase_wal_t *wal);
nbase_status_t nbase_wal_replay(nbase_wal_t *wal, nbase_db_t *db, uint64_t *out_applied);

/* nbase.c internal helpers */
void nbase_db_rdlock(const nbase_db_t *db);
void nbase_db_unlock(const nbase_db_t *db);
const nbase_store_t *nbase_db_store(const nbase_db_t *db);
const nbase_hnsw_t *nbase_db_hnsw(const nbase_db_t *db);
const nbase_meta_store_t *nbase_db_meta(const nbase_db_t *db);
bool nbase_db_hnsw_enabled(const nbase_db_t *db);

nbase_status_t nbase_db_replay_vec(nbase_db_t *db, nbase_wal_op_t op, uint64_t id, const float *vec, uint32_t dim);
nbase_status_t nbase_db_replay_delete(nbase_db_t *db, uint64_t id);
nbase_status_t nbase_db_replay_meta(nbase_db_t *db, nbase_wal_op_t op, uint64_t id, const nbase_meta_field_t *fields, uint32_t count);
nbase_status_t nbase_db_replay_text(nbase_db_t *db, nbase_wal_op_t op, uint64_t id, const char *text);
nbase_status_t nbase_db_replay_compact(nbase_db_t *db);

#endif /* NBASE_INTERNAL_H */
