#ifndef NBASE_H
#define NBASE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NBASE_VERSION_MAJOR 1
#define NBASE_VERSION_MINOR 14
#define NBASE_VERSION_PATCH 0
#define NBASE_MAX_DIM 65536U

typedef enum nbase_status {
  NBASE_OK = 0,
  NBASE_ERR_INVALID_ARG = -1,
  NBASE_ERR_NOMEM = -2,
  NBASE_ERR_NOT_FOUND = -3,
  NBASE_ERR_ALREADY_EXISTS = -4,
  NBASE_ERR_DIM_MISMATCH = -5,
  NBASE_ERR_IO = -6,
  NBASE_ERR_CORRUPT = -7,
  NBASE_ERR_UNSUPPORTED = -8,
  NBASE_ERR_INVALID_VECTOR = -9
} nbase_status_t;

typedef enum nbase_metric {
  NBASE_METRIC_EUCLIDEAN = 0,
  NBASE_METRIC_SQUARED_EUCLIDEAN = 1,
  NBASE_METRIC_COSINE = 2,
  NBASE_METRIC_INNER_PRODUCT = 3,
  NBASE_METRIC_MANHATTAN = 4,
  NBASE_METRIC_CHEBYSHEV = 5,
  NBASE_METRIC_HAMMING = 6
} nbase_metric_t;

typedef enum nbase_search_mode {
  NBASE_SEARCH_AUTO = 0,
  NBASE_SEARCH_EXACT = 1,
  NBASE_SEARCH_HNSW = 2
} nbase_search_mode_t;

typedef struct nbase_result {
  uint64_t id;
  float distance;
} nbase_result_t;

typedef struct nbase_scored_id {
  uint64_t id;
  float score;
} nbase_scored_id_t;

typedef struct nbase_edge {
  uint64_t src_id;
  uint64_t dst_id;
  float weight;
} nbase_edge_t;

typedef struct nbase_rerank_weight {
  const char *key;
  float weight;
} nbase_rerank_weight_t;

typedef bool (*nbase_filter_fn)(uint64_t id, void *user_ctx);

/* --- Typed Metadata & Predicate Filtering --- */

typedef enum nbase_meta_type {
  NBASE_META_INT64 = 0,
  NBASE_META_DOUBLE = 1,
  NBASE_META_BOOL = 2,
  NBASE_META_STRING = 3
} nbase_meta_type_t;

typedef struct nbase_meta_value {
  nbase_meta_type_t type;
  union {
    int64_t i64;
    double f64;
    bool b;
    const char *str;
  } as;
} nbase_meta_value_t;

typedef struct nbase_meta_field {
  const char *key;
  nbase_meta_value_t value;
} nbase_meta_field_t;

typedef struct nbase_meta_batch_item {
  uint64_t id;
  const nbase_meta_field_t *fields;
  uint32_t field_count;
} nbase_meta_batch_item_t;

typedef enum nbase_meta_op {
  NBASE_OP_EQ = 0,
  NBASE_OP_NE = 1,
  NBASE_OP_GT = 2,
  NBASE_OP_GTE = 3,
  NBASE_OP_LT = 4,
  NBASE_OP_LTE = 5,
  NBASE_OP_EXISTS = 6,
  NBASE_OP_CONTAINS = 7
} nbase_meta_op_t;

typedef struct nbase_meta_cond {
  const char *key;
  nbase_meta_op_t op;
  nbase_meta_value_t value;
} nbase_meta_cond_t;

typedef struct nbase_meta_filter {
  const nbase_meta_cond_t *conds;
  uint32_t count;
} nbase_meta_filter_t;

typedef struct nbase_search_opts {
  nbase_search_mode_t mode;
  nbase_metric_t metric;
  bool use_metric_override;
  uint32_t ef_search;
  nbase_filter_fn filter;
  void *filter_ctx;
  const nbase_meta_filter_t *meta_filter;
  uint32_t num_threads;
  bool skip_cache;
} nbase_search_opts_t;

typedef struct nbase_config {
  uint32_t dim;
  nbase_metric_t metric;
  uint32_t initial_capacity;
  bool enable_hnsw;
  uint32_t hnsw_m;
  uint32_t hnsw_ef_construction;
  uint32_t hnsw_ef_search;
  uint64_t seed;
  uint32_t cache_capacity;
} nbase_config_t;

typedef struct nbase_stats {
  uint32_t dim;
  nbase_metric_t metric;
  uint64_t active_vectors;
  uint64_t total_slots;
  uint64_t tombstones;
  bool mmap_backed;
  bool hnsw_enabled;
  int32_t hnsw_max_level;
  uint64_t hnsw_active_nodes;
  uint64_t hnsw_deleted_nodes;
  uint64_t text_docs;
  uint64_t memory_bytes;
  uint32_t cache_capacity;
  uint32_t cache_entries;
  uint64_t cache_hits;
  uint64_t cache_misses;
} nbase_stats_t;

typedef struct nbase_db nbase_db_t;

/* --- Core & Distance --- */

const char *nbase_version(void);
const char *nbase_strerror(nbase_status_t status);
const char *nbase_simd_backend(void);

nbase_config_t nbase_config_default(uint32_t dim);
nbase_search_opts_t nbase_search_opts_default(void);

float nbase_distance(nbase_metric_t metric, const float *a, const float *b, uint32_t dim);
float nbase_distance_scalar(nbase_metric_t metric, const float *a, const float *b, uint32_t dim);
nbase_status_t nbase_distance_batch(
    nbase_metric_t metric,
    const float *query,
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    float *out_distances);

/* --- Embedded Single-Partition DB --- */

nbase_status_t nbase_create(const nbase_config_t *cfg, nbase_db_t **out_db);
nbase_status_t nbase_open(const char *dir_path, nbase_db_t **out_db);
nbase_status_t nbase_open_mmap(const char *dir_path, nbase_db_t **out_db);
nbase_status_t nbase_open_wal(const char *dir_path, nbase_db_t **out_db);
nbase_status_t nbase_wal_enable(nbase_db_t *db, const char *dir_path);
nbase_status_t nbase_wal_checkpoint(nbase_db_t *db);
nbase_status_t nbase_save(const nbase_db_t *db, const char *dir_path);
void nbase_close(nbase_db_t *db);

nbase_status_t nbase_add(nbase_db_t *db, uint64_t id, const float *vec, uint32_t dim);
nbase_status_t nbase_add_batch(nbase_db_t *db, const uint64_t *ids, const float *vectors, uint32_t count, uint32_t dim, uint32_t num_threads);
nbase_status_t nbase_upsert(nbase_db_t *db, uint64_t id, const float *vec, uint32_t dim);
nbase_status_t nbase_upsert_batch(nbase_db_t *db, const uint64_t *ids, const float *vectors, uint32_t count, uint32_t dim, uint32_t num_threads);
nbase_status_t nbase_delete(nbase_db_t *db, uint64_t id);
nbase_status_t nbase_delete_batch(nbase_db_t *db, const uint64_t *ids, uint32_t count, uint32_t *out_deleted);
nbase_status_t nbase_delete_by_filter(nbase_db_t *db, const nbase_meta_filter_t *filter, uint32_t *out_deleted);
nbase_status_t nbase_get(const nbase_db_t *db, uint64_t id, float *out_vec, uint32_t dim);
nbase_status_t nbase_get_batch(const nbase_db_t *db, const uint64_t *ids, uint32_t count, uint32_t dim, float *out_vectors, bool *out_found);
bool nbase_contains(const nbase_db_t *db, uint64_t id);
nbase_status_t nbase_count(const nbase_db_t *db, const nbase_meta_filter_t *filter, uint64_t *out_count);
nbase_status_t nbase_list_ids(const nbase_db_t *db, uint64_t offset, uint32_t limit, uint64_t *out_ids, uint32_t *out_count, uint64_t *out_total);

typedef bool (*nbase_meta_iter_fn)(const char *key, const nbase_meta_value_t *val, void *user_ctx);

nbase_status_t nbase_set_meta(nbase_db_t *db, uint64_t id, const nbase_meta_field_t *fields, uint32_t count);
nbase_status_t nbase_set_meta_batch(nbase_db_t *db, const nbase_meta_batch_item_t *items, uint32_t count);
nbase_status_t nbase_update_meta(nbase_db_t *db, uint64_t id, const nbase_meta_field_t *fields, uint32_t count);
nbase_status_t nbase_update_meta_batch(nbase_db_t *db, const nbase_meta_batch_item_t *items, uint32_t count);
nbase_status_t nbase_get_meta(const nbase_db_t *db, uint64_t id, const char *key, nbase_meta_value_t *out_val, char *str_buf, size_t str_buf_len);
nbase_status_t nbase_delete_meta(nbase_db_t *db, uint64_t id);
nbase_status_t nbase_delete_meta_batch(nbase_db_t *db, const uint64_t *ids, uint32_t count, uint32_t *out_deleted);
nbase_status_t nbase_foreach_meta(const nbase_db_t *db, uint64_t id, nbase_meta_iter_fn cb, void *user_ctx);
nbase_status_t nbase_query_meta(const nbase_db_t *db, const nbase_meta_filter_t *filter, uint32_t limit, uint64_t *out_ids, uint32_t *out_count);

nbase_status_t nbase_search(const nbase_db_t *db, const float *query, uint32_t dim, uint32_t k, const nbase_search_opts_t *opts, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_search_radius(const nbase_db_t *db, const float *query, uint32_t dim, float max_distance, uint32_t max_results, const nbase_search_opts_t *opts, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_find_similar(const nbase_db_t *db, uint64_t id, uint32_t k, const nbase_search_opts_t *opts, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_search_batch(const nbase_db_t *db, const float *queries, uint32_t query_count, uint32_t dim, uint32_t k, const nbase_search_opts_t *opts, nbase_result_t *out_results, uint32_t *out_counts);

nbase_status_t nbase_set_text(nbase_db_t *db, uint64_t id, const char *text);
nbase_status_t nbase_set_text_batch(nbase_db_t *db, const uint64_t *ids, const char *const *texts, uint32_t count);
nbase_status_t nbase_get_text(const nbase_db_t *db, uint64_t id, char *out_buf, size_t buf_len, size_t *out_len);
nbase_status_t nbase_delete_text(nbase_db_t *db, uint64_t id);
nbase_status_t nbase_delete_text_batch(nbase_db_t *db, const uint64_t *ids, uint32_t count, uint32_t *out_deleted);
nbase_status_t nbase_search_text(const nbase_db_t *db, const char *query, uint32_t k, const nbase_meta_filter_t *meta_filter, nbase_scored_id_t *out_results, uint32_t *out_count);
nbase_status_t nbase_search_text_batch(const nbase_db_t *db, const char *const *queries, uint32_t query_count, uint32_t k, uint32_t num_threads, const nbase_meta_filter_t *meta_filter, nbase_scored_id_t *out_results, uint32_t *out_counts);
nbase_status_t nbase_search_hybrid(const nbase_db_t *db, const float *query_vec, uint32_t dim, const char *query_text, uint32_t k, float alpha, float rrf_k, const nbase_search_opts_t *opts, nbase_scored_id_t *out_results, uint32_t *out_count);
nbase_status_t nbase_search_hybrid_batch(const nbase_db_t *db, const float *query_vectors, uint32_t dim, const char *const *query_texts, uint32_t query_count, uint32_t k, float alpha, float rrf_k, const nbase_search_opts_t *opts, nbase_scored_id_t *out_results, uint32_t *out_counts);

nbase_status_t nbase_rebuild_hnsw(nbase_db_t *db);
nbase_status_t nbase_rebuild_hnsw_mt(nbase_db_t *db, uint32_t num_threads);
nbase_status_t nbase_compact(nbase_db_t *db);
nbase_status_t nbase_cache_configure(nbase_db_t *db, uint32_t max_entries);
nbase_status_t nbase_cache_clear(nbase_db_t *db);
nbase_status_t nbase_stats(const nbase_db_t *db, nbase_stats_t *out_stats);

/* --- K-Means++ & Product Quantization (PQ) --- */

nbase_status_t nbase_kmeans_train(const float *vectors, uint32_t count, uint32_t dim, uint32_t k, uint32_t max_iters, uint64_t seed, float *out_centroids, uint32_t *out_assignments);

typedef struct nbase_pq nbase_pq_t;

nbase_status_t nbase_pq_train(const float *vectors, uint32_t count, uint32_t dim, uint32_t num_subvectors, uint32_t num_centroids, uint32_t max_iters, uint64_t seed, nbase_pq_t **out_pq);
void nbase_pq_destroy(nbase_pq_t *pq);

uint32_t nbase_pq_dim(const nbase_pq_t *pq);
uint32_t nbase_pq_num_subvectors(const nbase_pq_t *pq);
uint32_t nbase_pq_num_centroids(const nbase_pq_t *pq);

nbase_status_t nbase_pq_encode(const nbase_pq_t *pq, const float *vectors, uint32_t count, uint8_t *out_codes);
nbase_status_t nbase_pq_decode(const nbase_pq_t *pq, const uint8_t *codes, uint32_t count, float *out_vectors);
nbase_status_t nbase_pq_build_lut(const nbase_pq_t *pq, const float *query, float *out_lut);
float nbase_pq_adc_distance(const nbase_pq_t *pq, const float *lut, const uint8_t *code);
nbase_status_t nbase_pq_adc_distance_batch(const nbase_pq_t *pq, const float *lut, const uint8_t *codes, uint32_t count, float *out_distances);

nbase_status_t nbase_pq_save(const nbase_pq_t *pq, const char *file_path);
nbase_status_t nbase_pq_load(const char *file_path, nbase_pq_t **out_pq);

/* --- Partitioned Multi-DB with Durable LRU Cache --- */

typedef struct nbase_pdb nbase_pdb_t;

typedef struct nbase_pdb_config {
  uint32_t dim;
  nbase_metric_t metric;
  uint32_t partition_capacity;
  uint32_t max_active_partitions;
  bool enable_hnsw;
  uint32_t hnsw_m;
  uint32_t hnsw_ef_construction;
  uint32_t hnsw_ef_search;
  uint64_t seed;
} nbase_pdb_config_t;

typedef struct nbase_pdb_search_opts {
  nbase_search_opts_t search;
  const uint32_t *partition_ids;
  uint32_t partition_id_count;
  bool search_all_partitions;
} nbase_pdb_search_opts_t;

typedef struct nbase_pdb_stats {
  uint32_t dim;
  nbase_metric_t metric;
  uint64_t total_vectors;
  uint32_t total_partitions;
  uint32_t loaded_partitions;
  uint32_t max_active_partitions;
  uint32_t active_partition_id;
  uint64_t eviction_count;
  uint64_t cache_hits;
  uint64_t cache_misses;
  uint64_t memory_bytes;
} nbase_pdb_stats_t;

nbase_pdb_config_t nbase_pdb_config_default(uint32_t dim);
nbase_pdb_search_opts_t nbase_pdb_search_opts_default(void);

nbase_status_t nbase_pdb_open(const char *dir_path, const nbase_pdb_config_t *cfg, nbase_pdb_t **out_pdb);
nbase_status_t nbase_pdb_save(nbase_pdb_t *pdb);
nbase_status_t nbase_pdb_close(nbase_pdb_t *pdb);

nbase_status_t nbase_pdb_add(nbase_pdb_t *pdb, uint64_t id, const float *vec, uint32_t dim);
nbase_status_t nbase_pdb_add_batch(nbase_pdb_t *pdb, const uint64_t *ids, const float *vectors, uint32_t count, uint32_t dim, uint32_t num_threads);
nbase_status_t nbase_pdb_upsert(nbase_pdb_t *pdb, uint64_t id, const float *vec, uint32_t dim);
nbase_status_t nbase_pdb_upsert_batch(nbase_pdb_t *pdb, const uint64_t *ids, const float *vectors, uint32_t count, uint32_t dim, uint32_t num_threads);
nbase_status_t nbase_pdb_delete(nbase_pdb_t *pdb, uint64_t id);
nbase_status_t nbase_pdb_delete_batch(nbase_pdb_t *pdb, const uint64_t *ids, uint32_t count, uint32_t *out_deleted);
nbase_status_t nbase_pdb_delete_by_filter(nbase_pdb_t *pdb, const nbase_meta_filter_t *filter, uint32_t *out_deleted);
nbase_status_t nbase_pdb_get(nbase_pdb_t *pdb, uint64_t id, float *out_vec, uint32_t dim);
nbase_status_t nbase_pdb_get_batch(nbase_pdb_t *pdb, const uint64_t *ids, uint32_t count, uint32_t dim, float *out_vectors, bool *out_found);
bool nbase_pdb_contains(nbase_pdb_t *pdb, uint64_t id);
nbase_status_t nbase_pdb_count(nbase_pdb_t *pdb, const nbase_meta_filter_t *filter, uint64_t *out_count);
nbase_status_t nbase_pdb_list_ids(nbase_pdb_t *pdb, uint64_t offset, uint32_t limit, uint64_t *out_ids, uint32_t *out_count, uint64_t *out_total);

nbase_status_t nbase_pdb_set_meta(nbase_pdb_t *pdb, uint64_t id, const nbase_meta_field_t *fields, uint32_t count);
nbase_status_t nbase_pdb_set_meta_batch(nbase_pdb_t *pdb, const nbase_meta_batch_item_t *items, uint32_t count);
nbase_status_t nbase_pdb_update_meta(nbase_pdb_t *pdb, uint64_t id, const nbase_meta_field_t *fields, uint32_t count);
nbase_status_t nbase_pdb_update_meta_batch(nbase_pdb_t *pdb, const nbase_meta_batch_item_t *items, uint32_t count);
nbase_status_t nbase_pdb_get_meta(nbase_pdb_t *pdb, uint64_t id, const char *key, nbase_meta_value_t *out_val, char *str_buf, size_t str_buf_len);
nbase_status_t nbase_pdb_delete_meta(nbase_pdb_t *pdb, uint64_t id);
nbase_status_t nbase_pdb_delete_meta_batch(nbase_pdb_t *pdb, const uint64_t *ids, uint32_t count, uint32_t *out_deleted);
nbase_status_t nbase_pdb_foreach_meta(nbase_pdb_t *pdb, uint64_t id, nbase_meta_iter_fn cb, void *user_ctx);
nbase_status_t nbase_pdb_query_meta(nbase_pdb_t *pdb, const nbase_meta_filter_t *filter, uint32_t limit, uint64_t *out_ids, uint32_t *out_count);

nbase_status_t nbase_pdb_set_text(nbase_pdb_t *pdb, uint64_t id, const char *text);
nbase_status_t nbase_pdb_set_text_batch(nbase_pdb_t *pdb, const uint64_t *ids, const char *const *texts, uint32_t count);
nbase_status_t nbase_pdb_get_text(nbase_pdb_t *pdb, uint64_t id, char *out_buf, size_t buf_len, size_t *out_len);
nbase_status_t nbase_pdb_delete_text(nbase_pdb_t *pdb, uint64_t id);
nbase_status_t nbase_pdb_delete_text_batch(nbase_pdb_t *pdb, const uint64_t *ids, uint32_t count, uint32_t *out_deleted);

nbase_status_t nbase_pdb_search(nbase_pdb_t *pdb, const float *query, uint32_t dim, uint32_t k, const nbase_pdb_search_opts_t *opts, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_pdb_search_radius(nbase_pdb_t *pdb, const float *query, uint32_t dim, float max_distance, uint32_t max_results, const nbase_pdb_search_opts_t *opts, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_pdb_search_batch(nbase_pdb_t *pdb, const float *queries, uint32_t query_count, uint32_t dim, uint32_t k, const nbase_pdb_search_opts_t *opts, nbase_result_t *out_results, uint32_t *out_counts);
nbase_status_t nbase_pdb_search_text(nbase_pdb_t *pdb, const char *query, uint32_t k, const nbase_pdb_search_opts_t *opts, nbase_scored_id_t *out_results, uint32_t *out_count);
nbase_status_t nbase_pdb_search_text_batch(nbase_pdb_t *pdb, const char *const *queries, uint32_t query_count, uint32_t k, const nbase_pdb_search_opts_t *opts, nbase_scored_id_t *out_results, uint32_t *out_counts);
nbase_status_t nbase_pdb_search_hybrid(nbase_pdb_t *pdb, const float *query_vec, uint32_t dim, const char *query_text, uint32_t k, float alpha, float rrf_k, const nbase_pdb_search_opts_t *opts, nbase_scored_id_t *out_results, uint32_t *out_count);
nbase_status_t nbase_pdb_search_hybrid_batch(nbase_pdb_t *pdb, const float *query_vectors, uint32_t dim, const char *const *query_texts, uint32_t query_count, uint32_t k, float alpha, float rrf_k, const nbase_pdb_search_opts_t *opts, nbase_scored_id_t *out_results, uint32_t *out_counts);
nbase_status_t nbase_pdb_find_similar(nbase_pdb_t *pdb, uint64_t id, uint32_t k, const nbase_pdb_search_opts_t *opts, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_pdb_rerank_mmr(nbase_pdb_t *pdb, const nbase_result_t *candidates, uint32_t candidate_count, uint32_t k, float lambda, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_pdb_rerank_weighted(nbase_pdb_t *pdb, const nbase_result_t *candidates, uint32_t candidate_count, const nbase_rerank_weight_t *weights, uint32_t weight_count, uint32_t k, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_pdb_extract_relationships(nbase_pdb_t *pdb, uint32_t k_neighbors, float max_distance, const nbase_pdb_search_opts_t *opts, nbase_edge_t **out_edges, uint64_t *out_edge_count);
nbase_status_t nbase_pdb_compact(nbase_pdb_t *pdb);
nbase_status_t nbase_pdb_rebuild_hnsw_mt(nbase_pdb_t *pdb, uint32_t num_threads);
nbase_status_t nbase_pdb_stats(nbase_pdb_t *pdb, nbase_pdb_stats_t *out_stats);

/* --- Dynamic Online Clustered Vector Index (.nbcl) --- */

typedef struct nbase_clustered nbase_clustered_t;

typedef struct nbase_clustered_config {
  uint32_t dim;
  nbase_metric_t metric;
  uint32_t target_cluster_size;
  uint32_t max_clusters;
  float distance_threshold;
  uint32_t kmeans_max_iters;
  uint64_t seed;
} nbase_clustered_config_t;

typedef struct nbase_clustered_stats {
  uint32_t dim;
  nbase_metric_t metric;
  uint32_t target_cluster_size;
  uint32_t max_clusters;
  uint32_t cluster_count;
  uint64_t active_vectors;
  uint64_t memory_bytes;
} nbase_clustered_stats_t;

nbase_clustered_config_t nbase_clustered_config_default(uint32_t dim);
nbase_status_t nbase_clustered_create(const nbase_clustered_config_t *cfg, nbase_clustered_t **out_cl);
void nbase_clustered_destroy(nbase_clustered_t *cl);
nbase_status_t nbase_clustered_add(nbase_clustered_t *cl, uint64_t id, const float *vec, uint32_t dim);
nbase_status_t nbase_clustered_add_batch(nbase_clustered_t *cl, const uint64_t *ids, const float *vectors, uint32_t count, uint32_t dim);
nbase_status_t nbase_clustered_upsert(nbase_clustered_t *cl, uint64_t id, const float *vec, uint32_t dim);
nbase_status_t nbase_clustered_upsert_batch(
    nbase_clustered_t *cl,
    const uint64_t *ids,
    const float *vectors,
    uint32_t count,
    uint32_t dim);
nbase_status_t nbase_clustered_delete(nbase_clustered_t *cl, uint64_t id);
nbase_status_t nbase_clustered_delete_batch(nbase_clustered_t *cl, const uint64_t *ids, uint32_t count, uint32_t *out_deleted);
nbase_status_t nbase_clustered_get(const nbase_clustered_t *cl, uint64_t id, float *out_vec, uint32_t dim);
nbase_status_t nbase_clustered_get_batch(
    const nbase_clustered_t *cl,
    const uint64_t *ids,
    uint32_t count,
    uint32_t dim,
    float *out_vectors,
    bool *out_found);
bool nbase_clustered_contains(const nbase_clustered_t *cl, uint64_t id);
nbase_status_t nbase_clustered_list_ids(
    const nbase_clustered_t *cl,
    uint64_t offset,
    uint32_t limit,
    uint64_t *out_ids,
    uint32_t *out_count,
    uint64_t *out_total);
nbase_status_t nbase_clustered_recluster(nbase_clustered_t *cl, uint32_t num_clusters);
nbase_status_t nbase_clustered_compact(nbase_clustered_t *cl);
nbase_status_t nbase_clustered_search(const nbase_clustered_t *cl, const float *query, uint32_t dim, uint32_t k, uint32_t nprobe, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_clustered_search_filtered(const nbase_clustered_t *cl, const float *query, uint32_t dim, uint32_t k, uint32_t nprobe, nbase_filter_fn filter, void *filter_ctx, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_clustered_search_radius(
    const nbase_clustered_t *cl,
    const float *query,
    uint32_t dim,
    float max_distance,
    uint32_t max_results,
    uint32_t nprobe,
    nbase_filter_fn filter,
    void *filter_ctx,
    nbase_result_t *out_results,
    uint32_t *out_count);
nbase_status_t nbase_clustered_search_batch(
    const nbase_clustered_t *cl,
    const float *queries,
    uint32_t query_count,
    uint32_t dim,
    uint32_t k,
    uint32_t nprobe,
    uint32_t num_threads,
    nbase_result_t *out_results,
    uint32_t *out_counts);
nbase_status_t nbase_clustered_stats(const nbase_clustered_t *cl, nbase_clustered_stats_t *out_stats);
nbase_status_t nbase_clustered_save(const nbase_clustered_t *cl, const char *file_path);
nbase_status_t nbase_clustered_load(const char *file_path, nbase_clustered_t **out_cl);

/* --- IVF-PQ Compressed Approximate Index (.nbiv) --- */

typedef struct nbase_ivfpq nbase_ivfpq_t;

typedef struct nbase_ivfpq_config {
  uint32_t dim;
  nbase_metric_t metric;
  uint32_t nlist;
  uint32_t num_subvectors;
  uint32_t num_centroids;
  uint32_t max_iters;
  uint64_t seed;
} nbase_ivfpq_config_t;

typedef struct nbase_ivfpq_stats {
  uint32_t dim;
  nbase_metric_t metric;
  uint32_t nlist;
  uint32_t num_subvectors;
  uint32_t num_centroids;
  uint64_t count;
} nbase_ivfpq_stats_t;

nbase_ivfpq_config_t nbase_ivfpq_config_default(uint32_t dim, uint32_t nlist, uint32_t num_subvectors);
nbase_status_t nbase_ivfpq_train(const nbase_ivfpq_config_t *cfg, const float *train_vectors, uint32_t train_count, nbase_ivfpq_t **out_index);
nbase_status_t nbase_ivfpq_add(nbase_ivfpq_t *index, const uint64_t *ids, const float *vectors, uint32_t count);
nbase_status_t nbase_ivfpq_upsert(nbase_ivfpq_t *index, uint64_t id, const float *vec, uint32_t dim);
nbase_status_t nbase_ivfpq_delete(nbase_ivfpq_t *index, uint64_t id);
nbase_status_t nbase_ivfpq_delete_batch(nbase_ivfpq_t *index, const uint64_t *ids, uint32_t count, uint32_t *out_deleted);
bool nbase_ivfpq_contains(const nbase_ivfpq_t *index, uint64_t id);
nbase_status_t nbase_ivfpq_list_ids(
    const nbase_ivfpq_t *index,
    uint64_t offset,
    uint32_t limit,
    uint64_t *out_ids,
    uint32_t *out_count,
    uint64_t *out_total);
nbase_status_t nbase_ivfpq_compact(nbase_ivfpq_t *index);
nbase_status_t nbase_ivfpq_search(const nbase_ivfpq_t *index, const float *query, uint32_t k, uint32_t nprobe, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_ivfpq_search_filtered(const nbase_ivfpq_t *index, const float *query, uint32_t k, uint32_t nprobe, nbase_filter_fn filter, void *filter_ctx, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_ivfpq_search_batch(
    const nbase_ivfpq_t *index,
    const float *queries,
    uint32_t query_count,
    uint32_t k,
    uint32_t nprobe,
    uint32_t num_threads,
    nbase_result_t *out_results,
    uint32_t *out_counts);
nbase_status_t nbase_ivfpq_save(const nbase_ivfpq_t *index, const char *file_path);
nbase_status_t nbase_ivfpq_load(const char *file_path, nbase_ivfpq_t **out_index);
nbase_status_t nbase_ivfpq_stats(const nbase_ivfpq_t *index, nbase_ivfpq_stats_t *out_stats);
void nbase_ivfpq_destroy(nbase_ivfpq_t *index);

/* --- MMR Diversity Reranking & Graph Analytics --- */

nbase_status_t nbase_rerank_mmr(const nbase_db_t *db, const nbase_result_t *candidates, uint32_t candidate_count, uint32_t k, float lambda, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_rerank_weighted(const nbase_db_t *db, const nbase_result_t *candidates, uint32_t candidate_count, const nbase_rerank_weight_t *weights, uint32_t weight_count, uint32_t k, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_extract_relationships(const nbase_db_t *db, uint32_t k_neighbors, float max_distance, nbase_edge_t **out_edges, uint64_t *out_edge_count);
nbase_status_t nbase_extract_relationships_filtered(const nbase_db_t *db, uint32_t k_neighbors, float max_distance, const nbase_meta_filter_t *meta_filter, nbase_edge_t **out_edges, uint64_t *out_edge_count);
nbase_status_t nbase_extract_communities(const uint64_t *ids, uint32_t node_count, const nbase_edge_t *edges, uint64_t edge_count, uint32_t *out_labels, uint32_t *out_community_count);
void nbase_free_edges(nbase_edge_t *edges);

/* --- Scalar Quantization (SQ8, .nbsq) --- */

typedef struct nbase_sq8 nbase_sq8_t;

nbase_status_t nbase_sq8_train(const float *vectors, uint32_t count, uint32_t dim, nbase_sq8_t **out_sq8);
void nbase_sq8_destroy(nbase_sq8_t *sq8);
uint32_t nbase_sq8_dim(const nbase_sq8_t *sq8);
nbase_status_t nbase_sq8_encode(const nbase_sq8_t *sq8, const float *vectors, uint32_t count, uint8_t *out_codes);
nbase_status_t nbase_sq8_decode(const nbase_sq8_t *sq8, const uint8_t *codes, uint32_t count, float *out_vectors);
float nbase_sq8_l2_sq_asymmetric(const nbase_sq8_t *sq8, const float *query, const uint8_t *code);
float nbase_sq8_l2_sq_symmetric(const nbase_sq8_t *sq8, const uint8_t *code_a, const uint8_t *code_b);
nbase_status_t nbase_sq8_l2_sq_asymmetric_batch(const nbase_sq8_t *sq8, const float *query, const uint8_t *codes, uint32_t count, float *out_distances);
nbase_status_t nbase_sq8_l2_sq_symmetric_batch(
    const nbase_sq8_t *sq8,
    const uint8_t *query_code,
    const uint8_t *codes,
    uint32_t count,
    float *out_distances);
nbase_status_t nbase_sq8_save(const nbase_sq8_t *sq8, const char *file_path);
nbase_status_t nbase_sq8_load(const char *file_path, nbase_sq8_t **out_sq8);

/* --- Locality-Sensitive Hashing (LSH, .nblh) --- */

typedef struct nbase_lsh nbase_lsh_t;

typedef struct nbase_lsh_config {
  uint32_t dim;
  nbase_metric_t metric;
  uint32_t num_tables;
  uint32_t num_bits;
  float bucket_width;
  uint64_t seed;
} nbase_lsh_config_t;

typedef struct nbase_lsh_stats {
  uint32_t dim;
  nbase_metric_t metric;
  uint32_t num_tables;
  uint32_t num_bits;
  float bucket_width;
  uint64_t active_vectors;
} nbase_lsh_stats_t;

nbase_lsh_config_t nbase_lsh_config_default(uint32_t dim);
nbase_status_t nbase_lsh_create(const nbase_lsh_config_t *cfg, nbase_lsh_t **out_lsh);
void nbase_lsh_destroy(nbase_lsh_t *lsh);
nbase_status_t nbase_lsh_add(nbase_lsh_t *lsh, uint64_t id, const float *vec, uint32_t dim);
nbase_status_t nbase_lsh_add_batch(nbase_lsh_t *lsh, const uint64_t *ids, const float *vectors, uint32_t count, uint32_t dim);
nbase_status_t nbase_lsh_upsert(nbase_lsh_t *lsh, uint64_t id, const float *vec, uint32_t dim);
nbase_status_t nbase_lsh_upsert_batch(nbase_lsh_t *lsh, const uint64_t *ids, const float *vectors, uint32_t count, uint32_t dim);
nbase_status_t nbase_lsh_delete(nbase_lsh_t *lsh, uint64_t id);
nbase_status_t nbase_lsh_delete_batch(nbase_lsh_t *lsh, const uint64_t *ids, uint32_t count, uint32_t *out_deleted);
nbase_status_t nbase_lsh_get(const nbase_lsh_t *lsh, uint64_t id, float *out_vec, uint32_t dim);
nbase_status_t nbase_lsh_get_batch(const nbase_lsh_t *lsh, const uint64_t *ids, uint32_t count, uint32_t dim, float *out_vectors, bool *out_found);
bool nbase_lsh_contains(const nbase_lsh_t *lsh, uint64_t id);
nbase_status_t nbase_lsh_list_ids(
    const nbase_lsh_t *lsh,
    uint64_t offset,
    uint32_t limit,
    uint64_t *out_ids,
    uint32_t *out_count,
    uint64_t *out_total);
nbase_status_t nbase_lsh_compact(nbase_lsh_t *lsh);
nbase_status_t nbase_lsh_search(const nbase_lsh_t *lsh, const float *query, uint32_t dim, uint32_t k, uint32_t multi_probe, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_lsh_search_filtered(const nbase_lsh_t *lsh, const float *query, uint32_t dim, uint32_t k, uint32_t multi_probe, nbase_filter_fn filter, void *filter_ctx, nbase_result_t *out_results, uint32_t *out_count);
nbase_status_t nbase_lsh_search_batch(
    const nbase_lsh_t *lsh,
    const float *queries,
    uint32_t query_count,
    uint32_t dim,
    uint32_t k,
    uint32_t multi_probe,
    uint32_t num_threads,
    nbase_result_t *out_results,
    uint32_t *out_counts);
nbase_status_t nbase_lsh_stats(const nbase_lsh_t *lsh, nbase_lsh_stats_t *out_stats);
nbase_status_t nbase_lsh_save(const nbase_lsh_t *lsh, const char *file_path);
nbase_status_t nbase_lsh_load(const char *file_path, nbase_lsh_t **out_lsh);

/* --- Binary Quantization (BQ, .nbbq) --- */

typedef struct nbase_bq nbase_bq_t;

nbase_status_t nbase_bq_train(const float *vectors, uint32_t count, uint32_t dim, nbase_bq_t **out_bq);
void nbase_bq_destroy(nbase_bq_t *bq);
uint32_t nbase_bq_dim(const nbase_bq_t *bq);
uint32_t nbase_bq_words_per_vec(const nbase_bq_t *bq);
nbase_status_t nbase_bq_encode(const nbase_bq_t *bq, const float *vectors, uint32_t count, uint64_t *out_words);
uint32_t nbase_bq_hamming(const uint64_t *a, const uint64_t *b, uint32_t num_words);
nbase_status_t nbase_bq_hamming_batch(const uint64_t *query_words, const uint64_t *codes, uint32_t count, uint32_t num_words, uint32_t *out_distances);
nbase_status_t nbase_bq_save(const nbase_bq_t *bq, const char *file_path);
nbase_status_t nbase_bq_load(const char *file_path, nbase_bq_t **out_bq);

/* --- BM25 Full-Text & Hybrid Reciprocal Rank Fusion (.nbbm) --- */

typedef struct nbase_bm25 nbase_bm25_t;

nbase_status_t nbase_bm25_create(float k1, float b, nbase_bm25_t **out_bm25);
void nbase_bm25_destroy(nbase_bm25_t *bm25);
uint64_t nbase_bm25_doc_count(const nbase_bm25_t *bm25);
nbase_status_t nbase_bm25_index(nbase_bm25_t *bm25, uint64_t id, const char *text);
nbase_status_t nbase_bm25_index_batch(
    nbase_bm25_t *bm25,
    const uint64_t *ids,
    const char *const *texts,
    uint32_t count);
nbase_status_t nbase_bm25_get(const nbase_bm25_t *bm25, uint64_t id, char *out_buf, size_t buf_len, size_t *out_len);
bool nbase_bm25_contains(const nbase_bm25_t *bm25, uint64_t id);
nbase_status_t nbase_bm25_delete(nbase_bm25_t *bm25, uint64_t id);
nbase_status_t nbase_bm25_delete_batch(nbase_bm25_t *bm25, const uint64_t *ids, uint32_t count, uint32_t *out_deleted);
nbase_status_t nbase_bm25_list_ids(
    const nbase_bm25_t *bm25,
    uint64_t offset,
    uint32_t limit,
    uint64_t *out_ids,
    uint32_t *out_count,
    uint64_t *out_total);
nbase_status_t nbase_bm25_compact(nbase_bm25_t *bm25);
nbase_status_t nbase_bm25_search(const nbase_bm25_t *bm25, const char *query, uint32_t k, nbase_scored_id_t *out_results, uint32_t *out_count);
nbase_status_t nbase_bm25_search_filtered(const nbase_bm25_t *bm25, const char *query, uint32_t k, nbase_filter_fn filter, void *filter_ctx, nbase_scored_id_t *out_results, uint32_t *out_count);
nbase_status_t nbase_bm25_search_batch(
    const nbase_bm25_t *bm25,
    const char *const *queries,
    uint32_t query_count,
    uint32_t k,
    uint32_t num_threads,
    nbase_scored_id_t *out_results,
    uint32_t *out_counts);
nbase_status_t nbase_bm25_save(const nbase_bm25_t *bm25, const char *file_path);
nbase_status_t nbase_bm25_load(const char *file_path, nbase_bm25_t **out_bm25);
nbase_status_t nbase_hybrid_rrf(const nbase_result_t *vec_results, uint32_t vec_count, const nbase_scored_id_t *text_results, uint32_t text_count, float alpha, float rrf_k, uint32_t k, nbase_scored_id_t *out_results, uint32_t *out_count);

/* --- Embeddable Multi-Threaded HTTP/1.1 Server --- */

typedef struct nbase_http_server nbase_http_server_t;

typedef struct nbase_http_config {
  const char *bind_host;
  uint16_t port;
  uint32_t num_workers;
  size_t max_body_bytes;
  uint32_t socket_timeout_ms;
  const char *api_key;
  const char *save_dir;
  uint32_t max_requests_per_sec;
  const char *cors_origin;
} nbase_http_config_t;

nbase_http_config_t nbase_http_config_default(void);
nbase_status_t nbase_http_server_start(nbase_db_t *db, const nbase_http_config_t *cfg, nbase_http_server_t **out_srv);
uint16_t nbase_http_server_port(const nbase_http_server_t *srv);
void nbase_http_server_stop(nbase_http_server_t *srv);

#ifdef __cplusplus
}
#endif

#endif /* NBASE_H */
