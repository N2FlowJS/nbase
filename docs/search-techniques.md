# Search, Metadata Filtering, Reranking & Graph Analytics

## Vector Search Modes & Metrics

- **`NBASE_SEARCH_AUTO`**: Uses HNSW when enabled, falling back to exact SIMD scan when HNSW is disabled.
- **`NBASE_SEARCH_EXACT`**: Brute-force SIMD scan returning 100% ground-truth results.
- **`NBASE_SEARCH_HNSW`**: HNSW beam search controlled by `opts.ef_search`. You can override the metric per query via `opts.use_metric_override = true` and `opts.metric`.
- **`nbase_search_batch`**: Parallel work-stealing search across `opts.num_threads` POSIX worker threads.

## Metadata Filtering

Store typed per-vector fields (`int64`, `double`, `bool`, `string`) with `nbase_set_meta` / `nbase_update_meta`. Supply `nbase_meta_filter_t` inside `nbase_search_opts_t` (or combine with a C predicate `opts.filter`):
- Comparison operators: `NBASE_OP_EQ`, `NBASE_OP_NE`, `NBASE_OP_GT`, `NBASE_OP_GTE`, `NBASE_OP_LT`, `NBASE_OP_LTE`
- Presence & substring search: `NBASE_OP_EXISTS`, `NBASE_OP_CONTAINS`

## BM25 Full-Text & Hybrid RRF Search

- Index per-vector documents via `nbase_set_text(db, id, text)` and retrieve via `nbase_search_text(db, query, k, meta_filter, ...)`.
- Fuse vector k-NN rankings and BM25 rankings using Reciprocal Rank Fusion via `nbase_search_hybrid(db, query_vec, dim, query_text, k, alpha, rrf_k, &opts, ...)`:

$$\text{score}_{\text{RRF}}(d) = \frac{\alpha}{k_{\text{rrf}} + r_{\text{vec}}(d)} + \frac{1 - \alpha}{k_{\text{rrf}} + r_{\text{text}}(d)}$$

## Diversity & Metadata-Weighted Reranking

- **MMR Reranking (`nbase_rerank_mmr`)**: Iteratively selects diverse top-`k` results controlled by `lambda` in `[0.0, 1.0]` (`1.0` = pure relevance, `0.0` = pure diversity).
- **Weighted Reranking (`nbase_rerank_weighted`)**: Adjusts candidate distance using numeric metadata attributes (`adjusted = dist - sum(value_f * weight_f)`) and sorts ascending.

## Graph Analytics (`src/rerank.c`)

- **`nbase_extract_relationships`**: Builds canonical undirected k-NN edges (`src_id < dst_id`, `weight <= max_distance`) across all active vectors (or across partitions via `nbase_pdb_extract_relationships`).
- **`nbase_extract_communities`**: Iterative path-compressed Union-Find clustering labeling connected components in O(V + E * alpha(V)).
