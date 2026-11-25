# API Reference (C11 `include/nbase.h` & C++17 `include/nbase.hpp`)

All functions in [`include/nbase.h`](../include/nbase.h) return `nbase_status_t` (with `NBASE_OK == 0` on success) unless noted otherwise.

## Status Codes & Enums

- **`nbase_status_t`**:
  `NBASE_OK (0)`, `NBASE_ERR_INVALID_ARG (-1)`, `NBASE_ERR_NOMEM (-2)`, `NBASE_ERR_NOT_FOUND (-3)`, `NBASE_ERR_ALREADY_EXISTS (-4)`, `NBASE_ERR_DIM_MISMATCH (-5)`, `NBASE_ERR_IO (-6)`, `NBASE_ERR_CORRUPT (-7)`, `NBASE_ERR_UNSUPPORTED (-8)`, `NBASE_ERR_INVALID_VECTOR (-9)`.
- **`nbase_metric_t`**:
  `NBASE_METRIC_EUCLIDEAN`, `NBASE_METRIC_SQUARED_EUCLIDEAN`, `NBASE_METRIC_COSINE`, `NBASE_METRIC_INNER_PRODUCT`, `NBASE_METRIC_MANHATTAN`, `NBASE_METRIC_CHEBYSHEV`, `NBASE_METRIC_HAMMING`.
- **`nbase_search_mode_t`**: `NBASE_SEARCH_AUTO`, `NBASE_SEARCH_EXACT`, `NBASE_SEARCH_HNSW`.
- **`nbase_meta_type_t`**: `NBASE_META_INT64`, `NBASE_META_DOUBLE`, `NBASE_META_BOOL`, `NBASE_META_STRING`.
- **`nbase_meta_op_t`**: `NBASE_OP_EQ`, `NBASE_OP_NE`, `NBASE_OP_GT`, `NBASE_OP_GTE`, `NBASE_OP_LT`, `NBASE_OP_LTE`, `NBASE_OP_EXISTS`, `NBASE_OP_CONTAINS`.

## Version, Diagnostics & Distance

- `const char *nbase_version(void);` -> `"1.14.0"`
- `const char *nbase_strerror(nbase_status_t status);`
- `const char *nbase_simd_backend(void);` -> `"avx2+fma"`, `"neon"`, or `"scalar"`
- `nbase_config_t nbase_config_default(uint32_t dim);` (`initial_capacity=64`, `enable_hnsw=true`, `hnsw_m=16`, `hnsw_ef_construction=200`, `hnsw_ef_search=50`, `cache_capacity=0`)
- `nbase_search_opts_t nbase_search_opts_default(void);`
- `float nbase_distance(nbase_metric_t metric, const float *a, const float *b, uint32_t dim);`
- `float nbase_distance_scalar(nbase_metric_t metric, const float *a, const float *b, uint32_t dim);`
- `nbase_status_t nbase_distance_batch(nbase_metric_t metric, const float *query, const float *vectors, uint32_t count, uint32_t dim, float *out_distances);`

## Single-Partition Embedded DB (`nbase_db_t`)

- **Lifecycle & WAL**:
  - `nbase_create`, `nbase_open`, `nbase_open_mmap`, `nbase_open_wal`
  - `nbase_wal_enable`, `nbase_wal_checkpoint`, `nbase_save`, `nbase_close`
- **Vectors & Batch Operations**:
  - `nbase_add`, `nbase_add_batch`, `nbase_upsert`, `nbase_upsert_batch`, `nbase_delete`, `nbase_delete_batch`, `nbase_delete_by_filter`, `nbase_get`, `nbase_get_batch`, `nbase_contains`, `nbase_count`, `nbase_list_ids`
- **Metadata**:
  - `nbase_set_meta`, `nbase_set_meta_batch`, `nbase_update_meta`, `nbase_update_meta_batch`, `nbase_get_meta`, `nbase_delete_meta`, `nbase_delete_meta_batch`, `nbase_foreach_meta`, `nbase_query_meta` (`nbase_meta_batch_item_t`)
- **Vector, Text & Hybrid Search**:
  - `nbase_search`, `nbase_search_radius`, `nbase_find_similar`, `nbase_search_batch`
  - `nbase_set_text`, `nbase_set_text_batch`, `nbase_get_text`, `nbase_delete_text`, `nbase_delete_text_batch`, `nbase_search_text`, `nbase_search_text_batch`, `nbase_search_hybrid`, `nbase_search_hybrid_batch`
- **Index Maintenance, Query Cache & Telemetry**:
  - `nbase_rebuild_hnsw`, `nbase_rebuild_hnsw_mt`, `nbase_compact`
  - `nbase_cache_configure(db, max_entries)`, `nbase_cache_clear(db)`
  - `nbase_stats(db, &out_stats)` populates `dim`, `metric`, `active_vectors`, `total_slots`, `tombstones`, `mmap_backed`, `hnsw_*`, `text_docs`, `memory_bytes`, `cache_capacity`, `cache_entries`, `cache_hits`, `cache_misses`.

## Reranking & Graph Analytics

- `nbase_rerank_mmr(db, candidates, candidate_count, k, lambda, out_results, &out_count)`
- `nbase_rerank_weighted(db, candidates, candidate_count, weights, weight_count, k, out_results, &out_count)` (`nbase_rerank_weight_t { const char *key; float weight; }`)
- `nbase_extract_relationships(db, k_neighbors, max_distance, &out_edges, &out_edge_count)` (`nbase_free_edges(edges)` to free)
- `nbase_extract_relationships_filtered(db, k_neighbors, max_distance, meta_filter, &out_edges, &out_edge_count)`
- `nbase_extract_communities(ids, node_count, edges, edge_count, out_labels, &out_community_count)`

## Multi-Partition DB (`nbase_pdb_t`)

- `nbase_pdb_config_default`, `nbase_pdb_search_opts_default`, `nbase_pdb_open`, `nbase_pdb_save`, `nbase_pdb_close`
- `nbase_pdb_add`, `nbase_pdb_add_batch`, `nbase_pdb_upsert`, `nbase_pdb_upsert_batch`, `nbase_pdb_delete`, `nbase_pdb_delete_batch`, `nbase_pdb_delete_by_filter`, `nbase_pdb_get`, `nbase_pdb_get_batch`, `nbase_pdb_contains`, `nbase_pdb_count`, `nbase_pdb_list_ids`
- `nbase_pdb_set_meta`, `nbase_pdb_set_meta_batch`, `nbase_pdb_update_meta`, `nbase_pdb_update_meta_batch`, `nbase_pdb_get_meta`, `nbase_pdb_delete_meta`, `nbase_pdb_delete_meta_batch`, `nbase_pdb_foreach_meta`, `nbase_pdb_query_meta`
- `nbase_pdb_set_text`, `nbase_pdb_set_text_batch`, `nbase_pdb_get_text`, `nbase_pdb_delete_text`, `nbase_pdb_delete_text_batch`
- `nbase_pdb_search`, `nbase_pdb_search_radius`, `nbase_pdb_search_batch`, `nbase_pdb_search_text`, `nbase_pdb_search_text_batch`, `nbase_pdb_search_hybrid`, `nbase_pdb_search_hybrid_batch`, `nbase_pdb_find_similar`, `nbase_pdb_rerank_mmr`, `nbase_pdb_rerank_weighted`
- `nbase_pdb_extract_relationships` (supports `opts->search.meta_filter` and predicate filters), `nbase_pdb_compact`, `nbase_pdb_rebuild_hnsw_mt`, `nbase_pdb_stats` (`total_vectors`, `total_partitions`, `loaded_partitions`, `eviction_count`, `cache_hits`, `cache_misses`, `memory_bytes`)

## Dynamic Online Clustered DB (`nbase_clustered_t`, `.nbcl`)

- `nbase_clustered_config_default`, `nbase_clustered_create`, `nbase_clustered_destroy`
- `nbase_clustered_add`, `nbase_clustered_add_batch`, `nbase_clustered_upsert`, `nbase_clustered_upsert_batch`, `nbase_clustered_delete`, `nbase_clustered_delete_batch`, `nbase_clustered_get`, `nbase_clustered_get_batch`, `nbase_clustered_contains`, `nbase_clustered_list_ids`
- `nbase_clustered_recluster`, `nbase_clustered_compact`, `nbase_clustered_search`, `nbase_clustered_search_filtered`, `nbase_clustered_search_radius`, `nbase_clustered_search_batch`, `nbase_clustered_stats` (includes `memory_bytes`), `nbase_clustered_save`, `nbase_clustered_load`

## Quantization, Approximate Indexes & Standalone BM25

- **K-Means++**: `nbase_kmeans_train`
- **Product Quantization (`nbase_pq_t`)**: `nbase_pq_train`, `destroy`, `dim`, `num_subvectors`, `num_centroids`, `encode`, `decode`, `build_lut`, `adc_distance`, `adc_distance_batch`, `save`, `load`
- **IVF-PQ (`nbase_ivfpq_t`, `.nbiv`)**: `nbase_ivfpq_config_default`, `train`, `add`, `upsert`, `delete`, `delete_batch`, `contains`, `list_ids`, `compact`, `search`, `search_filtered`, `search_batch`, `save`, `load`, `stats`, `destroy`
- **Scalar Quantization SQ8 (`nbase_sq8_t`, `.nbsq`)**: `nbase_sq8_train`, `destroy`, `dim`, `encode`, `decode`, `l2_sq_asymmetric`, `l2_sq_symmetric`, `l2_sq_asymmetric_batch`, `l2_sq_symmetric_batch`, `save`, `load`
- **Binary Quantization BQ (`nbase_bq_t`, `.nbbq`)**: `nbase_bq_train`, `destroy`, `dim`, `words_per_vec`, `encode`, `hamming`, `hamming_batch`, `save`, `load`
- **Multi-Probe LSH (`nbase_lsh_t`, `.nblh`)**: `nbase_lsh_config_default`, `create`, `destroy`, `add`, `add_batch`, `upsert`, `upsert_batch`, `get`, `get_batch`, `delete`, `delete_batch`, `contains`, `list_ids`, `compact`, `search`, `search_filtered`, `search_batch`, `stats`, `save`, `load`
- **Standalone BM25 & RRF (`nbase_bm25_t`, `.nbbm`)**: `nbase_bm25_create`, `destroy`, `doc_count`, `index`, `index_batch`, `get`, `contains`, `delete`, `delete_batch`, `list_ids`, `compact`, `search`, `search_filtered`, `search_batch`, `save`, `load`, `nbase_hybrid_rrf`

## Embeddable HTTP/1.1 Server (`nbase_http_server_t`)

- `nbase_http_config_default`, `nbase_http_server_start`, `nbase_http_server_port`, `nbase_http_server_stop`

## C++17 Header-Only Wrapper (`include/nbase.hpp`)

All handles wrap C ownership with move semantics, RAII cleanup, and `nbase::Error` exceptions:
- `nbase::Database` (`create`, `open`, `open_mmap`, `open_wal`, `add`, `add_batch`, `upsert`, `upsert_batch`, `remove`, `delete_batch`, `delete_by_filter`, `get`, `get_batch`, `contains`, `count`, `list_ids`, `set_meta`, `set_meta_batch`, `update_meta`, `update_meta_batch`, `delete_meta`, `delete_meta_batch`, `query_meta`, `set_text`, `set_text_batch`, `get_text`, `delete_text`, `delete_text_batch`, `search`, `search_radius`, `find_similar`, `search_batch`, `search_text`, `search_text_batch`, `search_hybrid`, `search_hybrid_batch`, `rerank_mmr`, `rerank_weighted`, `extract_relationships`, `extract_communities`, `compact`, `rebuild_hnsw_mt`, `cache_configure`, `cache_clear`, `stats`)
- `nbase::PartitionedDatabase` (`open`, `save`, `add`, `add_batch`, `upsert`, `upsert_batch`, `remove`, `delete_batch`, `delete_by_filter`, `get`, `get_batch`, `contains`, `count`, `list_ids`, `set_meta`, `set_meta_batch`, `update_meta`, `update_meta_batch`, `delete_meta`, `delete_meta_batch`, `query_meta`, `set_text`, `set_text_batch`, `get_text`, `delete_text`, `delete_text_batch`, `search`, `search_radius`, `search_batch`, `search_text`, `search_text_batch`, `search_hybrid`, `search_hybrid_batch`, `find_similar`, `rerank_mmr`, `rerank_weighted`, `extract_relationships`, `compact`, `rebuild_hnsw_mt`, `stats`)
- `nbase::distance_batch(metric, query, vectors)`
- `nbase::ClusteredDatabase` (`upsert_batch`, `get_batch`, `list_ids`, `compact`, `search_radius`, `search_batch`), `nbase::Pq` (`adc_distance_batch`), `nbase::IvfPq` (`upsert`, `remove`, `delete_batch`, `contains`, `list_ids`, `compact`, `search_filtered`, `search_batch`, `stats`), `nbase::Lsh` (`add_batch`, `upsert`, `upsert_batch`, `get`, `get_batch`, `delete_batch`, `contains`, `list_ids`, `compact`, `search_filtered`, `search_batch`, `stats`), `nbase::Sq8` (`l2_sq_symmetric`, `l2_sq_asymmetric_batch`, `l2_sq_symmetric_batch`), `nbase::Bq` (`hamming_batch`), `nbase::Bm25` (`index_batch`, `delete_batch`, `contains`, `list_ids`, `compact`, `search_batch`), `nbase::HttpServer`
