# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.17.0] - 2026-11-26

Embedded & partitioned batch text retrieval (`nbase_get_text_batch`, `nbase_pdb_get_text_batch`), C++17 `Database::get_text_batch` & `PartitionedDatabase::get_text_batch` sharing `detail::unpack_texts_batch`, AVX2 integer-accumulated Hamming single/batch kernels (`hamming_avx2`, `hamming_batch_avx2`), 4-row BQ `popcnt` unrolling for `num_words == 4` (`dim=256`), HTTP `PATCH`/`POST /api/vectors/patch` atomic batch patching & batch text lookup upgrade, and CLI `--json`/`--include-*` flags on `search` and `find-similar`.

### Added & Changed
- **Core C11 & C++17 Batch Text Retrieval (`include/nbase.h`, `src/nbase.c`, `src/partitioned.c`, `include/nbase.hpp`)**:
  - Added `nbase_get_text_batch(const nbase_db_t *, const uint64_t *, uint32_t, char **, bool *)` and `nbase_pdb_get_text_batch(nbase_pdb_t *, const uint64_t *, uint32_t, char **, bool *)` (with `nbase_free_texts` ownership), grouped by partition ID under a single lock acquisition.
  - Added `nbase::detail::unpack_texts_batch` RAII helper shared across `Database::get_text_batch`, `PartitionedDatabase::get_text_batch`, and `Bm25::get_batch`.
- **SIMD & Search Performance (`src/distance.c`, `src/bq.c`, `bench/bench_nbase.c`)**:
  - Added `hsum256_epi32` and upgraded `hamming_avx2` + `hamming_batch_avx2` to 2-row 2-way (`i += 16`) pure 256-bit SIMD integer accumulation replacing per-lane `movemask + popcnt`.
  - Extended `bq_hamming_batch_popcnt` `num_words == 4U` (`dim=256`) to a 4-row unrolled loop (`i += 4`) matching the `num_words == 1U` and `2U` fast paths.
  - Added `nbase_get_text_batch` (`text_get_batch(2000)`) benchmark coverage in `bench/bench_nbase.c`.
- **HTTP/1.1 Server & CLI Parity (`src/http_server.c`, `apps/nbase_cli.c`)**:
  - Refactored single/batch vector patching around shared `parse_and_validate_patch_item` / `apply_patch_item` helpers and added atomic `PATCH` / `POST /api/vectors/patch`.
  - Switched `POST /api/vectors/text/get` and `nbase-cli get-text-batch` onto `nbase_get_text_batch`.
  - Added `--json`, `--include-vectors`, `--include-meta`, and `--include-text` to `nbase-cli search` and `find-similar` (plus `--metric <M>` on `find-similar`) backed by `cli_print_search_results_json`.
- **Tests (`tests/test_db.c`, `tests/test_partitioned.c`, `tests/test_distance.c`, `tests/test_bq.c`, `tests/test_server.c`)**:
  - Updated `nbase_version()` assertion to `"1.17.0"`, added `test_db_get_text_batch_and_concurrent_reads`, `test_pdb_get_text_batch_multi_partition_and_eviction`, `test_hamming_avx2_and_signed_zero_edges`, `num_words == 4` unroll edges in `test_bq.c`, and `test_http_server_patch_batch_and_text_get_edges`.

## [1.16.0] - 2026-11-26

BM25 telemetry (`nbase_bm25_stats`), `nbase_bm25_get_batch` / `nbase_free_texts`, per-worker generation-stamped `score_map_t` scratch & single-term search fast path, C++17 `Bm25::{stats, get_batch}`, 2-row AVX2 Manhattan batch (`manhattan_batch_avx2`), HTTP `POST /api/vectors/{metadata,text}/get` & `parse_include_flags` deduplication, and CLI `patch-batch`.

### Added & Changed
- **BM25 Telemetry, Batch Retrieval & Search Scratch (`include/nbase.h`, `src/bm25.c`, `include/nbase.hpp`)**:
  - Added `nbase_bm25_stats_t` (`k1`, `b`, `doc_count`, `total_slots`, `tombstones`, `unique_terms`, `total_postings`, `avg_doc_len`, `memory_bytes`) and `nbase_bm25_stats(const nbase_bm25_t *, nbase_bm25_stats_t *)`.
  - Added `nbase_bm25_get_batch` and `nbase_free_texts`, plus C++17 `nbase::Bm25::{stats, get_batch}`.
  - Upgraded `score_map_t` with generation stamping (`bm25_search_scratch_t`) reused across queries per thread inside `nbase_bm25_search_batch`, eliminated duplicate `term_map_find` lookups, and added a direct single-term fast path bypassing `score_map_t`.
- **SIMD Distance, HTTP/1.1 Server & CLI Parity (`src/distance.c`, `src/http_server.c`, `apps/nbase_cli.c`)**:
  - Updated `manhattan_batch_avx2` to a cache-friendly 2-row 2-way (`i += 16`) `_mm256_andnot_ps` kernel matching `chebyshev_batch_avx2`.
  - Refactored `handle_get_vectors_batch`, `handle_search`, `handle_search_metadata`, and `handle_rerank` in `src/http_server.c` to reuse `parse_include_flags` while preserving handler defaults.
  - Added `POST /api/vectors/metadata/get` and `POST /api/vectors/text/get` sharing `handle_get_ids_attr_batch`.
  - Added `nbase-cli patch-batch <dir> <file.ndjson|->` with upfront NDJSON & ID existence validation.
- **Tests (`tests/test_db.c`, `tests/test_bm25.c`, `tests/test_server.c`)**:
  - Updated `nbase_version()` assertion to `"1.16.0"`, added `test_bm25_stats_get_batch_and_search_scratch_tsan`, and added `test_http_server_attr_batch_get_and_include_defaults`.

## [1.15.0] - 2026-11-25

IVF-PQ atomic `nbase_ivfpq_upsert_batch` & `nbase_ivfpq_search_radius`, LSH `nbase_lsh_search_radius` & per-worker generation-stamped `search_batch` scratch, C++17 `IvfPq::{upsert_batch, search_radius}` & `Lsh::search_radius`, 2-row AVX2 Chebyshev batch optimization (`chebyshev_batch_avx2`), HTTP `handle_post_similar` `parse_include_flags` cleanup, and CLI `get-meta-batch` / `get-text-batch`.

### Added & Changed
- **IVF-PQ & LSH API Parity (`include/nbase.h`, `src/ivfpq.c`, `src/lsh.c`, `include/nbase.hpp`)**:
  - Added `nbase_ivfpq_upsert_batch` sharing `ivfpq_write_batch_unlocked` and `ivfpq_delete_sorted_unlocked` with `nbase_ivfpq_add` / `nbase_ivfpq_delete_batch`: validates finiteness and duplicate input IDs before locking, encodes and reserves capacity upfront (zero partial mutation on OOM), then replaces pre-existing IDs and appends new codes atomically under one `wrlock`.
  - Added `nbase_ivfpq_search_radius` and `nbase_lsh_search_radius` supporting `max_distance`, `max_results`, and optional `nbase_filter_fn` predicates, alongside C++17 `nbase::IvfPq::{upsert_batch, search_radius}` and `nbase::Lsh::search_radius`.
  - Added per-worker generation-stamped `lsh_search_scratch_t` reuse in `lsh_search_batch_worker` eliminating per-query `seen` + heap allocations inside `nbase_lsh_search_batch`.
- **SIMD Kernel, Server & CLI Parity (`src/distance.c`, `src/http_server.c`, `apps/nbase_cli.c`, `bench/bench_nbase.c`)**:
  - Replaced the 4-stream `chebyshev_batch_avx2` loop with a 2-row 2-way (`i += 16`) `_mm256_andnot_ps` kernel reducing register/cache contention while keeping scalar parity, and updated `bench_nbase` `nbase_extract_relationships` threshold (`12.0f`) to emit non-empty graph edges on the 64-d benchmark dataset.
  - Refactored `handle_post_similar` in `src/http_server.c` to reuse `parse_include_flags`.
  - Added `nbase-cli get-meta-batch <dir> <id1,id2,...>` and `nbase-cli get-text-batch <dir> <id1,id2,...>` emitting per-ID JSON arrays.
- **Tests (`tests/test_db.c`, `tests/test_ivfpq.c`, `tests/test_lsh.c`, `tests/test_distance.c`)**:
  - Updated `nbase_version()` assertion to `"1.15.0"`, expanded `test_distance_batch_unrolled_edges` across dims `1..129` and counts `0..17`, and added `test_ivfpq_upsert_batch_and_search_radius_tsan` and `test_lsh_search_radius_and_batch_scratch_tsan`.

## [1.14.0] - 2026-11-25

Standalone index batch CRUD parity (`nbase_ivfpq_delete_batch`, `nbase_lsh_upsert_batch`, `nbase_lsh_delete_batch`, `nbase_lsh_get_batch`, `nbase_bm25_delete_batch` + C++17 `nbase.hpp` methods), AVX2 8-lane bit-exact `sq8_encode_avx2`, 4-thread parallel `nbase_extract_relationships_filtered`, HTTP `POST /api/search/batch` `include*` flags, and CLI `set-meta-batch` / `set-text-batch`.

### Added
- **Standalone Index Batch CRUD Parity (`include/nbase.h`, `src/lsh.c`, `src/ivfpq.c`, `src/bm25.c`, `include/nbase.hpp`)**:
  - Added `nbase_ivfpq_delete_batch` using sorted+deduped ID matching via `bsearch` across inverted lists and swap-delete compaction under `wrlock`, plus C++17 `nbase::IvfPq::delete_batch`.
  - Added `nbase_lsh_upsert_batch`, `nbase_lsh_delete_batch`, and `nbase_lsh_get_batch` (sharing `lsh_validate_write_batch` with `nbase_lsh_add_batch`), plus C++17 `nbase::Lsh::{upsert_batch, delete_batch, get_batch}`.
  - Added `nbase_bm25_delete_batch` using `bm25_delete_unlocked` under `wrlock`, plus C++17 `nbase::Bm25::delete_batch`.
- **AVX2 SQ8 Encode & Parallel k-NN Relationship Extraction (`src/sq8.c`, `src/rerank.c`, `bench/bench_nbase.c`)**:
  - Added AVX2 8-lane `sq8_encode_avx2` inside `nbase_sq8_encode` (`dim >= 8`) matching exact scalar `roundf` clamping semantics.
  - Parallelized `nbase_extract_relationships_filtered` across 4 `pthreads` when `active_cnt >= 128` using disjoint per-worker k-NN buffers and deterministic in-place edge compaction, plus relationship extraction timing in `bench_nbase`.
- **HTTP/1.1 Server & CLI Parity (`src/http_server.c`, `apps/nbase_cli.c`)**:
  - Updated `POST /api/search/batch` to honor `includeMetadata`, `includeVectors`, and `includeText` flags via `parse_include_flags` and `append_result_entries_json`.
  - Added `nbase-cli set-meta-batch <dir> <file.ndjson|-> [--update]` and `nbase-cli set-text-batch <dir> <file.ndjson|->` sharing a common NDJSON record reader (`cli_read_ndjson_id_records`).
- **Tests (`tests/test_db.c`, `tests/test_lsh.c`, `tests/test_ivfpq.c`, `tests/test_bm25.c`, `tests/test_sq8.c`, `tests/test_rerank.c`, `tests/test_server.c`)**:
  - Updated `nbase_version()` assertion to `"1.14.0"` and added `test_lsh_upsert_delete_get_batch_and_tsan`, `test_ivfpq_delete_batch_and_tsan`, `test_bm25_delete_batch_and_errors`, `test_sq8_avx2_encode_exact_edge_cases`, `test_extract_relationships_parallel_hnsw_and_exact_tsan`, and HTTP `POST /api/search/batch` `include*` flag assertions.

## [1.13.0] - 2026-11-25

Batch metadata deletion (`nbase_delete_meta_batch`), full Partitioned Multi-DB batch parity (`nbase_pdb_delete_meta_batch`, `nbase_pdb_delete_text_batch`, `nbase_pdb_search_text_batch`, `nbase_pdb_search_hybrid_batch`), parallel PQ encoding & batched parallel IVF-PQ ingestion, AVX2 SQ8 decoding (`sq8_decode_avx2`), HTTP `DELETE /api/vectors/metadata`, and CLI `delete-meta-batch`.

### Added
- **Single-DB & Partitioned Multi-DB Parity (`include/nbase.h`, `src/nbase.c`, `src/partitioned.c`, `include/nbase.hpp`)**:
  - Added `nbase_delete_meta_batch` (with single `wrlock`, query cache invalidation, and `NBASE_WAL_OP_DELETE_META` logging) and C++17 `nbase::Database::delete_meta_batch`.
  - Added `nbase_pdb_delete_meta_batch`, `nbase_pdb_delete_text_batch` (sharing `pdb_delete_attr_batch`), partition-outer `nbase_pdb_search_text_batch` and `nbase_pdb_search_hybrid_batch` (sharing `pdb_alloc_scored_batch_scratch`, `pdb_append_scored_batch_slice`, `pdb_finalize_scored_batch`), plus C++17 `nbase::PartitionedDatabase::{delete_meta_batch, delete_text_batch, search_text_batch, search_hybrid_batch}`.
- **Parallel PQ Encoding, Batched IVF-PQ Add & AVX2 SQ8 Decode (`src/pq.c`, `src/ivfpq.c`, `src/sq8.c`)**:
  - Parallelized `nbase_pq_encode` across disjoint row slices using `pthreads` while preserving bit-exact output.
  - Upgraded `nbase_ivfpq_add` (`count >= 32`) to run parallel coarse assignment + residual extraction, one parallel batch `nbase_pq_encode`, and upfront capacity reservation (`ivfpq_reserve_list`) with atomic zero-partial-mutation rollback on OOM.
  - Added AVX2 8-lane `sq8_decode_avx2` inside `nbase_sq8_decode`.
- **HTTP/1.1 Server & CLI Enhancements (`src/http_server.c`, `apps/nbase_cli.c`)**:
  - Added `DELETE /api/vectors/metadata` and `POST /api/vectors/metadata/delete` sharing `handle_delete_ids_attr_batch` with `handle_delete_text_batch`.
  - Added `nbase-cli delete-meta-batch <dir> <id1,id2,...>` unified cleanly with `delete-text-batch`.
- **Tests (`tests/test_db.c`, `tests/test_meta.c`, `tests/test_partitioned.c`, `tests/test_pq.c`, `tests/test_ivfpq.c`, `tests/test_sq8.c`, `tests/test_concurrency.c`, `tests/test_server.c`)**:
  - Added `test_db_delete_meta_batch_cache_and_wal`, `test_pdb_batch_delete_meta_text_and_search_text_hybrid_batch`, `test_pq_encode_parallel_determinism_tsan`, `test_ivfpq_parallel_batch_add_mt_and_cosine`, `test_sq8_avx2_decode_dims_and_accuracy`, concurrent `nbase_delete_meta_batch` TSan coverage, and HTTP `/api/vectors/metadata` batch delete tests.

## [1.12.0] - 2026-11-25

Single-partition parallel batch text/hybrid search & text deletion (`nbase_search_text_batch`, `nbase_search_hybrid_batch`, `nbase_delete_text_batch`), persistent pthread generation-barrier K-Means++ training (`nbase_kmeans_train_mt`), C++17 `nbase::Database` batch parity, HTTP `DELETE /api/vectors/text` + multi-threaded batch search, and CLI `delete-text-batch` / `search-text-batch` / `search-hybrid-batch`.

### Added
- **Core C11 Engine & C++17 Parity (`include/nbase.h`, `src/nbase.c`, `include/nbase.hpp`)**:
  - Added `nbase_delete_text_batch`, single-`rdlock` parallel `nbase_search_text_batch`, and single-`rdlock` parallel `nbase_search_hybrid_batch`, refactoring `nbase_search_text` and `nbase_search_hybrid` onto shared internal helpers (`db_search_text_unlocked`, `db_search_hybrid_unlocked`).
  - Added `nbase::Database::{search_batch, delete_text_batch, search_text_batch, search_hybrid_batch}` in `include/nbase.hpp` sharing a `detail::unpack_batch_results` helper.
- **Deterministic Parallel K-Means++ (`src/kmeans.c`, `src/pq.c`, `src/nbase_internal.h`, `bench/bench_nbase.c`)**:
  - Added `nbase_kmeans_train_mt` with a persistent generation-barrier `pthreads` worker pool for parallel Lloyd assignment and strictly deterministic sequential `0..count-1` centroid accumulation, while keeping multi-subvector `nbase_pq_train` free of nested thread oversubscription.
  - Added `search_text_batch(4t)` and `search_hybrid_batch(4t)` measurements to `bench/bench_nbase.c`.
- **HTTP/1.1 Server & CLI Enhancements (`src/http_server.c`, `apps/nbase_cli.c`)**:
  - Added `DELETE /api/vectors/text` and `POST /api/vectors/text/delete` using `nbase_delete_text_batch` (sharing `parse_json_u64_id_array` with `handle_delete_vectors`).
  - Upgraded `POST /api/search/text/batch` and `POST /api/search/hybrid/batch` to accept `"threads": 0..64` and dispatch via parallel `nbase_search_text_batch` / `nbase_search_hybrid_batch`.
  - Added `nbase-cli` commands `delete-text-batch`, `search-text-batch`, and `search-hybrid-batch`.
- **Tests (`tests/test_db.c`, `tests/test_meta.c`, `tests/test_pq.c`, `tests/test_concurrency.c`, `tests/test_server.c`)**:
  - Added `test_db_text_and_hybrid_batch_search_and_delete`, `test_kmeans_parallel_determinism_tsan`, concurrent TSan stress coverage for batch text/hybrid search & deletion, and HTTP route tests for `DELETE /api/vectors/text` + threaded batch search.

## [1.11.0] - 2026-11-25

Thread-safe BM25 concurrency, O(N log k) top-k heap & multi-threaded batch search/lifecycle parity (`index_batch`, `contains`, `list_ids`, `compact`, `search_batch`), AVX2 `bq_encode` + hardware 4-row `popcnt` Hamming kernels, HTTP `/api/search/{text,hybrid}/batch`, and CLI `patch` & `search-batch`.

### Added
- **BM25 Concurrency, Performance & Lifecycle Parity (`src/bm25.c`, `include/nbase.h`, `include/nbase.hpp`)**:
  - Added `pthread_rwlock_t` concurrency, cached per-posting `doc_len` (zero struct size growth), hoisted BM25 IDF/norm arithmetic, pre-sized query score maps, and bounded O(N log k) top-k min-heap extraction.
  - Added `nbase_bm25_index_batch`, `nbase_bm25_contains`, sorted paginated `nbase_bm25_list_ids`, `nbase_bm25_compact` (reclaiming doc tombstones & dead terms + shrinking posting allocations), and work-stealing `pthreads` `nbase_bm25_search_batch`, plus C++17 `Bm25::{index_batch, contains, list_ids, compact, search_batch}`.
- **Binary Quantization AVX2 Encode & 4-Row POPCNT (`src/bq.c`)**:
  - Added runtime-dispatched AVX2 `_mm256_cmp_ps`/`_mm256_movemask_ps` bit-packing in `bq_encode_avx2` and 4-row unrolled hardware `popcnt` (`bq_hamming_popcnt`, `bq_hamming_batch_popcnt`).
- **HTTP/1.1 Server, CLI & Benchmark Enhancements (`src/http_server.c`, `apps/nbase_cli.c`, `bench/bench_nbase.c`)**:
  - Added `POST /api/search/text/batch` and `POST /api/search/hybrid/batch` sharing `append_scored_entries_json`.
  - Added `nbase-cli patch <dir> <id>` (`--vector`, `--meta`, `--text`, `--clear-text`) and `nbase-cli search-batch <dir> <file.ndjson|->`.
  - Added `search_batch(4t)` QPS timing to `bench_nbase`.
- **Tests (`tests/test_bm25.c`, `tests/test_bq.c`, `tests/test_server.c`, `tests/test_db.c`)**:
  - Added `test_bm25_crud_batch_compact_and_tsan`, `test_bq_avx2_encode_and_popcnt_batch_edges`, and `test_http_server_text_and_hybrid_batch_search`.

## [1.10.0] - 2026-11-25

IVF-PQ mutable CRUD & lifecycle parity (`upsert`, `delete`, `contains`, `list_ids`, `compact`), 4-row unrolled AVX2+FMA SQ8 asymmetric/symmetric batch kernels, 4-code interleaved PQ ADC batch lookup with prefetch, HTTP `PATCH /api/vectors/:id`, `HEAD /api/vectors/:id/{metadata,text}`, selective `ids` export, and CLI `import --upsert` / `export --ids`.

### Added
- **IVF-PQ Mutable CRUD & Compaction Parity (`src/ivfpq.c`, `include/nbase.h`, `include/nbase.hpp`)**:
  - Added thread-safe (`pthread_rwlock_t`) `nbase_ivfpq_upsert`, `nbase_ivfpq_delete` (O(1) swap-with-last inside inverted lists), `nbase_ivfpq_contains`, sorted paginated `nbase_ivfpq_list_ids`, and `nbase_ivfpq_compact`.
  - Exposed corresponding C++17 `nbase::IvfPq` methods: `upsert` (pointer and `std::vector` overloads), `remove`, `contains`, `list_ids`, and `compact`.
- **SIMD SQ8 & PQ ADC Batch Acceleration (`src/sq8.c`, `src/pq.c`)**:
  - Upgraded `sq8_l2_sq_asymmetric_batch_avx2` and `sq8_l2_sq_symmetric_batch_avx2` to 4-row unrolled AVX2+FMA accumulation sharing query/scale YMM register loads across 4 codes.
  - Added `pq_adc_four_256` 4-code interleaved LUT accumulation with guarded software prefetch (`__builtin_prefetch`) and bit-exact summation order inside `nbase_pq_adc_distance_batch`.
- **HTTP/1.1 Server & CLI Enhancements (`src/http_server.c`, `apps/nbase_cli.c`)**:
  - Added `PATCH /api/vectors/:id` for partial updates (`vector`, partial-merge `metadata`, and `text` set/`null` clear), plus `HEAD /api/vectors/:id/metadata` and `HEAD /api/vectors/:id/text`.
  - Extended `POST /api/vectors/export` to support optional `"ids": [...]` selection alongside `"filters"`.
  - Added `--upsert` to `nbase-cli import` and `--ids <id1,id2,...>` to `nbase-cli export` (combinable with `--filter`).
- **Tests (`tests/test_ivfpq.c`, `tests/test_sq8.c`, `tests/test_pq.c`, `tests/test_server.c`, `tests/test_db.c`)**:
  - Added `test_ivfpq_crud_upsert_delete_list_compact_and_tsan` (CRUD, cross-cluster upsert, pagination, compact + save/load, TSan concurrent readers/writer).
  - Extended `test_sq8_l2_sq_{asymmetric,symmetric}_batch` and `test_pq_adc_distance_batch` to cover 4-row/4-code unroll tail lengths (`0..7`, `13`) and additional unaligned dimensions (`17`, `35`).
  - Added `test_http_server_patch_vector_head_subresources_and_export_ids`.

## [1.9.0] - 2026-11-25

Full 7/7 AVX2+FMA batch distance coverage (`chebyshev_batch_avx2`, `hamming_batch_avx2`), HNSW neighbor software prefetching, LSH CRUD & compaction parity (`upsert`, `get`, `list_ids`, `compact`), HTTP `HEAD` & filtered `POST /api/vectors/export`, CLI `exists` & `export --filter`, and cross-format binary fuzz testing.

### Added
- **SIMD Distance Kernels & HNSW Search (`src/distance.c`, `src/hnsw.c`)**:
  - Added 4-row unrolled AVX2+FMA `chebyshev_batch_avx2` and `hamming_batch_avx2` batch kernels, completing 7/7 metrics with SIMD unrolling.
  - Added guarded next-neighbor software prefetching (`__builtin_prefetch`) on visited marks and flat vector slots in HNSW `search_layer`.
- **LSH CRUD & Compaction Parity (`src/lsh.c`, `include/nbase.h`, `include/nbase.hpp`)**:
  - Added `nbase_lsh_upsert`, `nbase_lsh_get`, sorted paginated `nbase_lsh_list_ids`, and `nbase_lsh_compact` (compacting flat store tombstones and re-indexing buckets in place), with C++17 `Lsh::{upsert, get, list_ids, compact}`.
- **HTTP/1.1 & CLI Tooling (`src/http_server.c`, `apps/nbase_cli.c`)**:
  - Added `HEAD /health`, `HEAD /api/vectors/:id`, CORS `HEAD` method support, and filtered/selective `POST /api/vectors/export`.
  - Added `nbase-cli exists <dir> <id1,id2,...>` and `--filter <json>` option for `nbase-cli export`.
- **Fuzz, Sanitizer & Benchmark Coverage (`tests/*.c`, `bench/bench_nbase.c`)**:
  - Added `test_binary_formats_fuzz_corruption_and_truncation` across DB (`.nbv`, `.nbi`, `.nbm`, `.nbbm`), `.nblh`, `.nbcl`, `.nbiv`, `.nbsq`, `.nbbq`, plus `test_lsh_upsert_get_list_compact_and_tsan`, `test_http_server_head_and_filtered_export`, extended `test_distance_batch_unrolled_edges`, and `Batch Chebyshev` / `Batch Hamming` benchmarks.

## [1.8.0] - 2026-11-25

Cosine & Manhattan 4-row AVX2+FMA batch unrolling, Clustered DB compaction & multi-threaded batch search, SIMD-batched graph relationship extraction fallback, HTTP batch existence / dedicated radius search endpoints, and CLI `search-meta` & `rerank` commands.

### Added
- **AVX2+FMA Cosine & Manhattan Batch Kernels + Graph Fallback (`src/distance.c`, `src/rerank.c`)**:
  - Upgraded `cosine_batch_avx2` and added `manhattan_batch_avx2` with 4-row unrolled accumulation loops sharing query register loads across 4 candidates.
  - Upgraded `nbase_extract_relationships_filtered` non-HNSW path to score contiguous candidate batches via `nbase_distance_batch`.
- **Clustered Index Compaction & Multi-Threaded Batch Search (`src/clustered.c`, `include/nbase.h`, `include/nbase.hpp`)**:
  - Added `nbase_clustered_compact` (reclaiming store tombstones, shrinking `slot_cluster` & empty cluster member buffers) and work-stealing `nbase_clustered_search_batch` under shared `pthread_rwlock_rdlock`.
  - Updated C++17 `ClusteredDatabase::compact` and `ClusteredDatabase::search_batch`.
- **HTTP/1.1 & CLI Tooling (`src/http_server.c`, `apps/nbase_cli.c`)**:
  - Added `POST /api/vectors/exists` (`ids` batch presence check) and `POST /api/search/radius` (dedicated radius search with metadata filters and include flags).
  - Added `nbase-cli search-meta` and `nbase-cli rerank` (`diversity`, `weighted`, `standard`).
- **Tests & Benchmarks (`tests/test_distance.c`, `tests/test_clustered.c`, `tests/test_rerank.c`, `tests/test_server.c`, `bench/bench_nbase.c`)**:
  - Extended unrolled batch kernel parity tests to Cosine (with zero-norm candidate) and Manhattan, added Clustered `compact` + TSan `search_batch` tests, added no-HNSW relationship extraction test, added HTTP `/api/vectors/exists` & `/api/search/radius` tests, and added `Batch Manhattan` + Clustered `search_batch(4t)` benchmarks.

## [1.7.0] - 2026-11-24

SIMD distance unrolling, LSH batching & multi-threaded query search, Clustered DB batch/radius parity, standalone `POST /api/rerank`, and CLI graph/distance tooling increment.

### Added
- **SIMD Distance Kernels & LSH Multi-Threaded Batch Search (`src/distance.c`, `src/lsh.c`, `include/nbase.h`, `include/nbase.hpp`)**:
  - Upgraded AVX2+FMA `squared_euclidean_batch_avx2` and `inner_product_batch_avx2` with 4-row unrolled accumulation loops sharing query register loads across 4 candidate vectors.
  - Replaced per-hyperplane scalar/callback projection hashing in `src/lsh.c` with single-call `nbase_distance_batch(NBASE_METRIC_INNER_PRODUCT, ...)` across all tables and bits.
  - Added thread-pooled work-stealing `nbase_lsh_search_batch` under shared `pthread_rwlock_rdlock` and C++17 `Lsh::search_batch`.
- **Clustered Index Batch & Radius Parity (`src/clustered.c`, `include/nbase.h`, `include/nbase.hpp`)**:
  - Added `nbase_clustered_get_batch` and `nbase_clustered_search_radius` (with C++17 `ClusteredDatabase::get_batch` and `ClusteredDatabase::search_radius`).
- **HTTP/1.1 Standalone Reranking & CLI Subcommands (`src/http_server.c`, `apps/nbase_cli.c`)**:
  - Added `POST /api/rerank` supporting `"diversity"` (MMR), `"weighted"` (metadata-weighted), and `"standard"` candidate reranking with optional `includeMetadata`, `includeVectors`, and `includeText`.
  - Added `nbase-cli communities <dir> [--k <K>] [--max-distance <D>] [--filter <json>]` and `nbase-cli distance <metric> <q0,q1,...> <v0,v1,...> [...]`.
- **Tests & Benchmarks (`tests/test_distance.c`, `tests/test_clustered.c`, `tests/test_lsh.c`, `tests/test_server.c`, `bench/bench_nbase.c`)**:
  - Added edge-dimension/count parity tests for unrolled batch kernels, ASan/TSan concurrency coverage for `nbase_clustered_get_batch` / `search_radius` and `nbase_lsh_search_batch`, HTTP tests for `POST /api/rerank`, and expanded `bench_nbase` coverage.

## [1.6.0] - 2026-11-24

Performance, batching, reliability, and NDJSON tooling increment adding SIMD-batched exact store scanning, symmetric SQ8 batch quantization distance (`nbase_sq8_l2_sq_symmetric_batch`), multi-threaded IVF-PQ batch search (`nbase_ivfpq_search_batch`), HTTP NDJSON streaming import/export (`GET /api/vectors/export`, `POST /api/vectors/import`), CLI filter/metric extensions (`delete-by-filter`, `--filter`, `--metric`), and deterministic WAL truncation & CRC bitflip fuzz verification.

### Added
- **SIMD Exact Store Scan & Quantization/IVF-PQ Acceleration (`src/store.c`, `src/sq8.c`, `src/ivfpq.c`)**:
  - Upgraded unfiltered `nbase_store_exact_search` and `nbase_store_exact_search_radius` to evaluate contiguous active slot runs of up to 256 vectors via `nbase_distance_batch`.
  - Added `nbase_sq8_l2_sq_symmetric_batch` with 16-way unrolled AVX2+FMA integer-difference quantization distance kernel.
  - Added thread-pooled work-stealing `nbase_ivfpq_search_batch` under shared `pthread_rwlock_rdlock` and upgraded IVF-PQ coarse quantization (`nbase_ivfpq_add` + search) and `build_dot_lut` to evaluate batches via `nbase_distance_batch`.
- **HTTP/1.1 NDJSON Import/Export, CLI & C++17 Wrapper (`src/http_server.c`, `apps/nbase_cli.c`, `include/nbase.hpp`)**:
  - Added `GET /api/vectors/export` (`application/x-ndjson` paginated serialization) and `POST /api/vectors/import` (NDJSON ingest with `?upsert=true`, `?threads=N`, batch metadata, and batch text).
  - Added `nbase-cli delete-by-filter <dir> --filter <json>` and optional `--filter <json>` / `--metric <M>` flags on `search`, `search-text`, and `search-hybrid`.
  - Updated `include/nbase.hpp` (`Sq8::l2_sq_symmetric_batch`, `IvfPq::search_batch`).
- **WAL Fuzz & Concurrency Coverage (`tests/test_wal.c`, `tests/test_ivfpq.c`, `tests/test_sq8.c`, `tests/test_server.c`)**:
  - Added deterministic WAL truncation-sweep and payload/header CRC bitflip fuzz verification alongside `test_ivfpq_search_batch_mt_and_tsan`, `test_sq8_l2_sq_symmetric_batch`, and `test_http_server_ndjson_import_export`.

## [1.5.0] - 2026-11-24

Performance, batching, and API ergonomics increment introducing SIMD 1-to-N batch distance (`nbase_distance_batch`), faster K-Means++ training, deterministic multi-threaded PQ codebook training, clustered batch upsert & paginated ID listing, `POST /api/vectors/:id/similar` & `POST /api/distance`, and `nbase-cli find-similar` / `relationships`.

### Added
- **SIMD 1-to-N Batch Distance & Faster PQ/K-Means++ (`src/distance.c`, `src/kmeans.c`, `src/pq.c`)**:
  - Added `nbase_distance_batch(metric, query, vectors, count, dim, out_distances)` across all 7 metrics with AVX2+FMA / NEON / scalar paths, once-per-query norm precomputation for Cosine, dedicated `dim == 8` AVX2 fast path, and 16-float loop unrolling.
  - Upgraded `nbase_kmeans_train`, `nbase_pq_encode`, and `nbase_pq_build_lut` to evaluate squared L2 batches via `nbase_distance_batch`.
  - Parallelized `nbase_pq_train` across independent subvectors (`pthread_create` work-stealing with up to 4 threads when `num_subvectors > 1 && count * num_centroids >= 16384`), preserving bit-for-bit deterministic codebooks.
- **Clustered Index Batch Upsert & ID Pagination (`src/clustered.c`)**:
  - Added `nbase_clustered_upsert_batch` with upfront vector finiteness and duplicate-ID validation.
  - Added `nbase_clustered_list_ids` returning sorted active vector IDs (`offset`, `limit`, `out_total`).
- **HTTP/1.1 Server, CLI & C++17 Wrapper (`src/http_server.c`, `apps/nbase_cli.c`, `include/nbase.hpp`)**:
  - Added `POST /api/vectors/:id/similar` supporting `k`, `filters`, `metric`, `mode`, `efSearch`, `skipCache`, and `include*`.
  - Added `POST /api/distance` supporting 1-to-N distance calculation (`metric`, `query`, `vectors`).
  - Added `nbase-cli find-similar <dir> <id> [--k <K>] [--ef <EF>] [--exact] [--filter <json>]` and `relationships <dir> [--k <K>] [--max-distance <D>] [--filter <json>]`.
  - Updated `include/nbase.hpp` (`nbase::distance_batch`, `ClusteredDatabase::upsert_batch`, `ClusteredDatabase::list_ids`).

## [1.4.0] - 2026-11-24

Performance, thread-safety, and API completeness increment adding filtered vector counting (`nbase_count` / `nbase_pdb_count`), SIMD-batched SQ8 & BQ distance kernels, thread-safe SIMD-accelerated LSH & IVF-PQ indexes, full HTTP `/api/vectors/count` and `/api/vectors/:id/{metadata,text}` REST methods, CLI `count`/`get-text`/`delete-meta`/`delete-text` commands, deterministic JSON fuzz coverage, and C++17 `nbase::Sq8`/`Bq`/`Lsh` batch wrappers.

### Added
- **Filtered & Unfiltered Counting (`nbase_count`, `nbase_pdb_count`)**:
  - Added O(1) unfiltered counting (`pdb_count` without reloading evicted partitions) and allocation-free metadata-filtered counting across `nbase_db_t` and `nbase_pdb_t`.
- **SIMD-Batched Quantization Distance Kernels (`src/sq8.c`, `src/bq.c`)**:
  - Added `nbase_sq8_l2_sq_asymmetric_batch` with query shift precomputation and 16-way AVX2+FMA unrolling.
  - Added `nbase_bq_hamming_batch` with specialized `1`/`2`/`4`-word fast paths and 4x unrolling for arbitrary bit widths.
- **Thread-Safe & SIMD-Accelerated LSH & IVF-PQ (`src/lsh.c`, `src/ivfpq.c`)**:
  - Added internal `pthread_rwlock_t` reader/writer synchronization across `nbase_lsh_t` and `nbase_ivfpq_t`.
  - Replaced scalar inner-product projection loops in `lsh_hash_table` and IVF-PQ `build_dot_lut` with runtime-dispatched AVX2+FMA/NEON `nbase_get_distance_fn` kernels.
  - Added atomic `nbase_lsh_add_batch` and thread-safe `nbase_lsh_contains`.
- **HTTP/1.1 Server, CLI, C++17 & Fuzz Tests (`src/http_server.c`, `apps/nbase_cli.c`, `include/nbase.hpp`, `tests/test_json.c`)**:
  - Added `GET`/`POST /api/vectors/count` (with optional `"filters"`), `GET`/`PUT`/`DELETE /api/vectors/:id/metadata`, and `GET /api/vectors/:id/text`.
  - Added `nbase-cli count <dir> [--filter <json>]`, `delete-meta`, `get-text`, and `delete-text`.
  - Updated `include/nbase.hpp` (`Database::count`, `PartitionedDatabase::count`, `Sq8::l2_sq_symmetric`, `Sq8::l2_sq_asymmetric_batch`, `Bq::hamming_batch`, `Lsh::add_batch`, `Lsh::contains`, `Lsh::stats`).
  - Added 8 new tests including TSan concurrency on LSH & IVF-PQ and a 1,000-iteration deterministic JSON mutation fuzz suite.

## [1.3.0] - 2026-11-24

Feature and throughput increment adding atomic single-`fsync` batch metadata & text APIs (`nbase_db_t` & `nbase_pdb_t`), filtered k-NN relationship & community extraction, batch PQ ADC distance evaluation with IVF-PQ acceleration, HTTP bulk metadata/text endpoints, CLI metadata management, and C++17 `nbase::Pq` RAII wrapper.

### Added
- **Atomic Batch Metadata & Text Indexing (`nbase_db_t`, `nbase_wal_t`, `nbase_pdb_t`)**:
  - Added `nbase_meta_batch_item_t`, `nbase_set_meta_batch`, `nbase_update_meta_batch`, and `nbase_set_text_batch` with upfront field/ID validation (zero partial mutation on missing IDs), single query cache invalidation, and single-`fsync` WAL batching (`nbase_wal_append_meta_batch`, `nbase_wal_append_text_batch`).
  - Added partitioned multi-DB parity APIs `nbase_pdb_set_meta_batch`, `nbase_pdb_update_meta_batch`, and `nbase_pdb_set_text_batch` with partition-grouped locality.
- **Filtered Relationship & Community Extraction (`src/rerank.c`, `src/partitioned.c`)**:
  - Added `nbase_extract_relationships_filtered(db, k_neighbors, max_distance, meta_filter, &edges, &edge_count)` supporting both exact and HNSW graph paths.
  - Enabled `meta_filter` and predicate filter enforcement inside `nbase_pdb_extract_relationships`.
- **Batch PQ ADC Scan & IVF-PQ Acceleration (`src/pq.c`, `src/ivfpq.c`)**:
  - Added `nbase_pq_adc_distance_batch` with a 4x-unrolled branchless fast path for `256`-centroid codebooks and exact scalar parity across `<256`-centroid codebooks.
  - Upgraded unfiltered `nbase_ivfpq_search` / `nbase_ivfpq_search_filtered` inner list scans to evaluate PQ distances in batch via `nbase_pq_adc_distance_batch`.
- **HTTP/1.1 Server, CLI & C++17 Wrapper (`src/http_server.c`, `apps/nbase_cli.c`, `include/nbase.hpp`)**:
  - Added `POST /api/vectors/metadata` (batch replace/set, optional `"mode"`), `PATCH /api/vectors/metadata` (batch partial update), `POST /api/vectors/text` (batch text indexing), and optional `"filters"` on `POST /api/search/relationships` and `POST /api/search/communities`.
  - Upgraded bulk vector `POST /api/vectors` / `POST /api/vectors/upsert` and `nbase-cli import` flushes to use `nbase_set_meta_batch` and `nbase_set_text_batch`.
  - Added `nbase-cli set-meta <dir> <id> <json-object> [--update]` and `nbase-cli get-meta <dir> <id>`.
  - Added `nbase::Pq` RAII class plus `set_meta_batch`, `update_meta_batch`, `set_text_batch`, and filtered `extract_relationships` methods in `include/nbase.hpp`.

## [1.2.0] - 2026-11-24

Feature increment adding radius/range search, paginated deterministic ID listing, HTTP `GET /api/vectors` pagination, CLI `list` & `--max-distance`, and filtered/SIMD approximate indexes.

### Added
- **Radius / Range Search (`nbase_search_radius`, `nbase_pdb_search_radius`)**:
  - Added exact SIMD-accelerated and HNSW-mode radius search (`nbase_store_exact_search_radius`, `nbase_search_radius`, `nbase_pdb_search_radius`) supporting metadata/predicate filters, metric overrides, and negative finite inner-product radii.
  - Added optional `"maxDistance"` support in `POST /api/search` and `--max-distance <D>` in `nbase-cli search`.
- **Paginated ID Enumeration (`nbase_list_ids`, `nbase_pdb_list_ids`)**:
  - Added sorted active vector ID pagination (`offset`, `limit`, `total`) for both single and partitioned DBs (without reloading evicted partitions from disk).
  - Added `GET /api/vectors?offset=0&limit=50&includeVectors=...&includeMetadata=...&includeText=...` in `src/http_server.c` and `nbase-cli list <dir> [--offset <N>] [--limit <M>]`.
- **Filtered Approximate Indexes & SIMD Coarse Quantization (`src/ivfpq.c`, `src/lsh.c`)**:
  - Added `nbase_ivfpq_search_filtered` and `nbase_lsh_search_filtered`, plus corresponding C++17 methods in `include/nbase.hpp` (`Database`, `PartitionedDatabase`, `IvfPq`, `Lsh`).
  - Replaced scalar IVF-PQ coarse L2/dot loops with runtime-dispatched AVX2+FMA/NEON `nbase_get_distance_fn` kernels.

## [1.1.0] - 2026-11-23

Feature completeness update adding thread-safe clustered indexing enhancements, high-throughput batch upsert/get & partitioned reranking, HTTP bulk upsert/get endpoints, CLI commands, and aligned shared library `SOVERSION`.

### Added
- **Thread-Safe Dynamic Clustered Index (`src/clustered.c`)**:
  - Added internal `pthread_rwlock_t` synchronization across all mutating and read operations on `nbase_clustered_t`.
  - Added `nbase_clustered_add_batch`, `nbase_clustered_delete_batch`, `nbase_clustered_search_filtered`, and `memory_bytes` footprint accounting in `nbase_clustered_stats_t`.
- **Batch Upsert, Batch Get & Partitioned Reranking (`src/nbase.c`, `src/wal.c`, `src/partitioned.c`)**:
  - Added `nbase_upsert_batch` with single-`fsync` WAL batch logging (`nbase_wal_append_vec_batch_op`), duplicate-ID validation, cache invalidation, and parallel HNSW batch insertion.
  - Added `nbase_get_batch` for single-lock multi-ID retrieval.
  - Added multi-partition parity APIs `nbase_pdb_upsert_batch`, `nbase_pdb_get_batch`, `nbase_pdb_rerank_mmr`, and `nbase_pdb_rerank_weighted` with partition-grouped batch locality.
- **HTTP/1.1 Server, CLI & C++17 Wrapper (`src/http_server.c`, `apps/nbase_cli.c`, `include/nbase.hpp`)**:
  - Added `POST /api/vectors/upsert` (single and bulk `vectors` array upsert) and `POST /api/vectors/get` (bulk fetch by `ids` with `includeVectors`, `includeMetadata`, `includeText`).
  - Added `nbase-cli upsert <dir> <id> <v0,v1,...>` and `nbase-cli get-batch <dir> <id1,id2,...>`.
  - Added matching `Database`, `PartitionedDatabase`, and `ClusteredDatabase` methods in `include/nbase.hpp`.

### Changed
- Aligned `libnbase` shared library `SOVERSION` in `CMakeLists.txt` from `0` to `1` (`libnbase.so.1 -> libnbase.so.1.1.0`) matching major version `1`.

## [1.0.0] - 2026-11-20

Complete transition from Node.js/TypeScript to the pure-C11/C++17 `nbase` engine promoted to the repository root.

### Added
- **Pure-C11 Core Engine (`include/nbase.h`, `src/`) & Header-Only C++17 RAII Wrapper (`include/nbase.hpp`)**:
  - Zero runtime dependencies (`libm` + POSIX `pthreads`), compiled strictly with `-std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror`.
  - Runtime-dispatched **AVX2+FMA** (`x86_64`), **NEON** (`ARM64`), and scalar distance kernels across 7 metrics (`euclidean`, `squared_euclidean`, `cosine`, `inner_product`, `manhattan`, `chebyshev`, `hamming`).
  - Contiguous flat vector storage (`src/store.c`) with open-addressing `uint64_t` lookup, zero-copy `nbase_open_mmap`, tombstones, and physical `nbase_compact`.
  - Slot-indexed HNSW (`src/hnsw.c`) with Algorithm 4 heuristic pruning, 256-stripe node locks for multi-threaded `nbase_add_batch` / `nbase_rebuild_hnsw_mt`, generation-stamped per-query search state, `ef_search` dial, and per-query metric override.
  - Typed binary metadata (`int64`, `double`, `bool`, `string`) & predicate filtering (`EQ`, `NE`, `GT`, `GTE`, `LT`, `LTE`, `EXISTS`, `CONTAINS`), `nbase_delete_batch`, and `nbase_delete_by_filter`.
  - CRC32-framed Write-Ahead Log (`wal.nbw` / `NBASEWL1`), torn-tail truncation, auto-replay, and atomic `nbase_wal_checkpoint`.
  - Thread-safe LRU query result cache (`cache_capacity`, `skip_cache`, `nbase_cache_configure`, `nbase_cache_clear`, automatic invalidation on mutations, and `memory_bytes` footprint accounting).
  - Integrated Okapi BM25 full-text index (`.nbbm`) and Reciprocal Rank Fusion (`nbase_search_hybrid`, `nbase_hybrid_rrf`).
  - Maximal Marginal Relevance (`nbase_rerank_mmr`), metadata-weighted reranking (`nbase_rerank_weighted`), k-NN graph extraction (`nbase_extract_relationships`), and iterative Union-Find community detection (`nbase_extract_communities`).
  - Quantization & approximate indexes: K-Means++, PQ (`.nbq`), IVF-PQ (`.nbiv`), SIMD SQ8 (`.nbsq`), 1-bit Binary Quantization (`.nbbq`), Multi-Probe LSH (`.nblh`), and Dynamic Online Clustered Vector Index (`.nbcl`).
  - Multi-Partition DB (`nbase_pdb_t`) with durable LRU partition eviction, cross-partition batch insert/delete/filter-delete/search/text/hybrid/graph APIs, and `cache_hits`/`cache_misses`/`memory_bytes` stats.
- **Standalone HTTP/1.1 Keep-Alive Daemon (`nbased`) & Streaming CLI (`nbase-cli`)**:
  - Pure-POSIX `poll` worker pool HTTP/1.1 server (`src/http_server.c`) with strict RFC 8259 JSON parser (`src/json.c`), RFC 9112 lingering close on early rejects, Bearer authentication, CORS, per-second rate limiting, `/stats`, Prometheus `/metrics` (rolling 256-sample P50/P95/P99 latency, `nbase_memory_bytes`, `nbase_cache_*`), and `/api/admin/{save,checkpoint,compact,rebuild,cache}`.
  - Backward-compatible `nbased` CLI/env fallbacks (`HOST`, `PORT`, `NBASE_DB_PATH`, `NBASE_VECTOR_SIZE`, `NBASE_API_KEYS`, `NBASE_CORS_ORIGINS`, plus `NBASE_CACHE_SIZE` / `--cache-size`).
  - Offline CLI `nbase-cli` with NDJSON streaming `import`/`export`, `delete-batch`, `search-hybrid`, and `info`.
- **Docker, Packaging & CI**:
  - Multi-stage `debian:bookworm-slim` [Dockerfile](Dockerfile) exposing port `1307`, [docker-compose.yml](docker-compose.yml), CPack `.tar.gz` trial/release packaging, and GitHub Actions CMake CI (`Release`, `ASan+UBSan`, `TSan`).

### Removed
- Removed the legacy Node.js/TypeScript implementation (`src/*.ts`, `test/*.ts`, `package.json`, `tsconfig*.json`, `eslint.config.mjs`, `nodemon.json`, `.github/workflows/npm.yml`).

## [0.3.1] - 2026-10-20

Five defects found reviewing the 0.3.0 release: two silent data-loss paths, one
process-killing event, one erased field, and a cache that served the wrong
payload. Each has a regression test that fails on 0.3.0 and passes here.

### Fixed

- **Evicting a partition deleted its unpersisted vectors.** The LRU disposer
  closed the partition with `save: false`, and `save()` only walks the *loaded*
  partitions — so once a partition left the cache nothing would ever write it
  again. On `partitionCapacity: 2, maxActivePartitions: 1` with 6 vectors, only
  **2 of 6** survived on disk. Eviction now persists before releasing, and
  `close()` awaits the write: lru-cache v11 never awaits a disposer
  (`Disposer` is typed `(value, key, reason) => void` and the return value is
  dropped), which is also what the old comment claimed otherwise.
- **A `partitionIds` search silently skipped the partitions it named.**
  `findNearest` filtered the caller's list against `loadedPartitions`, so a
  request for a non-resident partition returned results from whatever happened
  to be resident, or `[]` — indistinguishable from "no matches". An explicit
  list is now loaded and searched; unknown ids are reported and dropped. A
  search with no list still covers only the resident partitions, and now logs
  that it did so instead of passing it off as a full search.
- **`VectorDBMonitor` emitted `error` with no listener.** `recordError` emits
  `error`, which Node treats specially: with no listener the emit throws
  `ERR_UNHANDLED_ERROR`. Every `partition:error` is routed into it, so the
  first partition error threw out of whatever code path raised it. The monitor's
  errors are now forwarded to the `Database`'s own `error` event.
- **`updateVector` erased the vector's metadata.** `ClusteredVectorDB` updated
  by deleting and re-adding the vector, and `deleteVector` removes the metadata
  entry; the re-add did not supply it again. Every update silently dropped that
  vector's metadata, which then also excluded it from any metadata filter. It
  now updates in place, so metadata and dimension survive and an update no
  longer emits `vector:delete`.
- **The search cache ignored `includeMetadata` and shared its result objects.**
  The key was hand-listed and omitted the flag, and the cache is consulted
  *before* `Search` attaches metadata, so a query primed with
  `includeMetadata: false` satisfied a later `includeMetadata: true` query with
  metadata-free results. The key is now derived from the whole option object, so
  a newly added option is in the key by default, and both the stored entries and
  the returned results are copied — a caller mutating what it got back could
  rewrite the cached payload.

- **An absolute `persistence.dbPath` was rewritten into the project directory.**
  `Database` joined it onto `process.cwd()`, and `path.join(cwd, '/tmp/my-db')`
  is `<cwd>/tmp/my-db`, so an absolute path produced a directory inside the
  package instead of the one that was asked for. Now `path.resolve`, which
  honours an absolute path and still resolves a relative one against the cwd.

### Fixed

- **`efSearch` was never read.** `ef` was always `max(k, this.efSearch)`, so the
  documented accuracy dial did nothing: `efSearch: 1` and `efSearch: 500`
  returned the same list. It now sets the beam width — on a 4000x128 index,
  200 queries went from 0.37 ms to 32.35 ms across that range.
- **`distanceMetric` was ignored on the HNSW path.** The distance function is
  fixed when the index is built, so a `cosine` request was answered with
  Euclidean distances in Euclidean order: measured recall against the true
  cosine ranking was 0.80, and it is 1.00 now. The per-search function is
  threaded through the traversal rather than swapping shared state, so
  concurrent searches cannot change each other's metric.
- **`exactDimensions: true` returned `[]` after every `buildIndex()`.** The
  per-dimension entry point was only written by the `dimensionSet.size === 1`
  branch of `addVector`, and the build's dimension pre-scan had already filled
  every group before the first `addVector` ran. The pre-scan seeds it now.
- **A query could be answered from the wrong dimension group.** With
  `dimensionAware` on, the descent started at the global entry point, which
  belongs to whichever dimension was indexed first, so a query in another
  dimension returned confidently wrong-dimension hits whenever those were
  closer in raw distance. It now starts at that dimension's entry point.
- **An id of `0` disabled the index.** `if (!this.entryPointId)` and
  `if (dimEntryPoint)` are truthiness tests, and `0` is a valid id:
  `findNearest` silently answered from the linear fallback — which returns
  exact results, so nothing looked wrong — and the per-dimension entry point
  was never promoted past the level-0 node it started on.
- **Distance kernels are typed against `DistanceMetric`** instead of `string`,
  and the metric a per-search function implements is detected by comparison.
- **HNSW prunes with Algorithm 4 instead of nearest-M.** Keeping the `M` nearest
  neighbours builds a kNN graph — every node points inward at its own cluster,
  with no long-range links, which is the property a small-world graph needs to
  be navigable. Recall@10 over six seeded queries on 8000 vectors of dimension
  64, both variants built from the same data in the same process, reproduced
  over two runs: **0.917 average / 0.800 worst before, 0.967 / 0.900 after.**
  `keepPrunedConnections` is off; refilling with the rejected candidates
  re-introduces the clustering the heuristic removes. This is a benchmark
  result rather than a CI gate, for the `Math.random()` reason above — the suite
  pins the recall floor instead.
- **Tombstones survive a save/load.** `serialize()` omitted `deletedNodes`, so a
  reloaded index reported `deletedNodesCount: 0` while the graph still carried
  the deleted nodes and their edges, and every query kept paying to traverse
  them.

Recall@10 measured with the new oracle: 1.00 at 500 vectors of dimension 16
(every run), and 0.95-0.98 average / 0.90 worst at 2000x128.

### Known

- **`_randomLevel()` uses `Math.random()`, so two builds of the same data
  produce different graphs** and one build's recall@10 is not reproducible.
  Every figure quoted here is several seeded queries against one build, and
  comparing *builds* of two variants has to happen in the same process or the
  delta is noise. That is not hypothetical: the first measurement of the
  pruning heuristic below said it was worse, and only a same-process comparison
  showed the opposite.

### Added

- **`Database.batchSearch(queries, options)` and `POST /api/search/batch`.**
  Both have been documented in `README.md` and `docs/api-reference.md` since
  0.1.x with nothing behind them. The method runs the ordinary `findNearest`
  path per query — so the cache, the concurrency limiter, reranking and
  metadata all apply — with a bounded worker pool, and reports a failed query
  in `errors` with its index instead of flattening it into an empty result list.
  `BatchEngineSearch` was not reused: it bypasses all of the above and turns a
  failure or timeout into an empty array.
- `DatabaseBatchSearchOptions`; the duplicate `BatchSearchOptions` declaration
  in `types.ts` was removed, since interface merging had been making its
  effective shape the union of two documents.

- `test/test-helpers/search-oracle.ts` — a seeded PRNG, a float64 reference
  distance, a brute-force top-k and a recall metric. Every existing search
  assertion was `length > 0`, so an implementation returning `k` arbitrary
  vectors passed.
- `test/tests/search_oracle.test.ts` — the exact paths must match the reference
  id for id; the approximate ones must reach a stated recall; the library's
  kernels must agree with the reference; a deleted vector never comes back.
- `test/tests/hnsw_correctness.test.ts` — `efSearch`, the metric override, the
  per-dimension entry point, `exactDimensions` after a build, and id `0`.

### Security

Hardening of the optional REST API. `createServer` was the only code that
assembled the real Express app, and it had no tests, so none of this was
covered.

- **The server bound every interface.** `app.listen(PORT)` ignores the
  configured host, so a process configured for `localhost` — the default — was
  reachable from the network while logging `http://localhost:PORT`. The host is
  passed through now.
- **CORS is opt-in.** It was mounted as a bare `cors()`, answering every origin
  with `Access-Control-Allow-Origin: *`. That is not access control, and it
  defeated an api-key middleware added through `middleware` — a key held by a
  browser app is readable by any site that can reach the API. Pass
  `cors: ['https://app.example']`, or `cors: true` to restore the old
  behaviour.
- **API keys (opt-in).** `auth: { apiKeys: [...] }` protects `/api/*` via
  `Authorization: Bearer` or `x-api-key`, compared with `timingSafeEqual` over
  SHA-256 digests. With no keys configured the API stays open and the server logs
  a warning saying so; `auth: { requireAuth: true }` makes it refuse to start
  instead.
- **A 500 no longer leaks internal messages.** Every route answered
  `res.status(500).json({ error: error.message })`, and those strings come from
  the storage layer — `ENOENT: /home/user/.../partitions/vec.bin` is what a
  client got, which is a map of the host. Outside `NODE_ENV=development` the
  client now gets `Internal server error`.
- **`k` is validated.** `k: "abc"` produced `NaN`, HNSW derives its beam width as
  `Math.max(k, efSearch)`, and `results.size >= NaN` is never true — so one
  request scanned the whole index. `k: 1e9` did the same directly. `k`,
  `efSearch`, `limit` and `rerankLambda` are now bounded integers/numbers on
  `/search`, `/similar` and `/search/metadata`.
- **`threshold` is capped on the O(n^2) endpoints.** `/relationships` and
  `/communities` only checked `> 0`; a threshold of 1e9 makes every pair related,
  so 10k vectors meant 50M result objects and an OOM kill — which also loses
  whatever the database had not saved yet.
- **Vector payloads are checked.** `new Float32Array(["1", null])` stored NaN and
  answered 200, poisoning every later distance for that vector.
- **Ids are checked.** An object or `null` id became a `Map` key across
  `memoryStorage`, `metadata` and `vectorDimensions`, and because
  `typeof id === 'number'` was false the auto-id counter never advanced, so the
  next generated id silently overwrote it.
- **`__proto__` is stripped from metadata** before it is persisted and echoed.
- **A metadata criteria no longer matches inherited keys.** `field in meta` walks
  the prototype chain, so `{"criteria": "constructor"}` matched every stored
  record. `Object.hasOwn` now.
- **`/stats` is rate-limited.** It is mounted at `/`, outside the `/api` limiter,
  and it is an O(all vectors) call that stringifies every metadata object.
- **`limit` is honoured by the metadata search.** Only the object-criteria branch
  applied it; the string/array branch returned every match from every loaded
  partition.
- **Middleware order.** The 404 handler now runs before the error handler, so a
  throw after `createServer` returns reaches the app's handler instead of
  Express's default one, which answers with a stack trace. A malformed JSON body
  is reported as 400 instead of 500.

### Added

- `test/api/server_defaults.test.ts` — the assembled app: headers, CORS, auth,
  error handling, payload validation, limiter scope.
- `test/api/validation.test.ts` — the validators, including the NaN and
  out-of-range cases that were denial of service.
- `src/server/utils/validation.ts` and `src/server/utils/http_error.ts`.
- `trustProxy` option; without it the rate limiter trusts a client-supplied
  `X-Forwarded-For` by default, which was the case before it existed.

- `test/database/database.test.ts` — the `Database` facade had no tests at all;
  nothing in the suite ever constructed one, and the coverage tool reported it
  as 100%.
- `test/tests/partitioned_vector_db.eviction.test.ts` — eviction durability,
  restart, and `partitionIds` coverage.

## [0.3.0] - 2026-10-02

Breaking renames plus the removal of every explicit `any` from `src/`. The
published package was still 0.1.10, so everything in 0.2.0 and this release
arrives together.

### Changed

- **BREAKING: `UnifiedSearch` is now `Search`**, in `src/search/search.ts`
  (was `src/search/unified_search.ts`). The public export, the class name and
  every JSDoc reference changed with it.
- **BREAKING: `UnifiedSearchOptions` is now `SearchOptions`.** The old
  `SearchOptions` was a thin wrapper over it, so the two were merged: the
  backward-compatibility aliases `limit`, `offset` and `stopEarly` moved into
  the base interface. `HybridSearchOptions` and `BatchQuery.options` follow
  the merged type.
- **BREAKING: `UnifiedSearchPartitionedStats` is now `SearchStats`**, and
  `Database.getUnifiedSearch()` is now `Database.getSearch()`.
- **BREAKING: metadata is `Record<string, unknown>`**, not
  `Record<string, any>`, everywhere on the public surface (`SearchResult`,
  `addVector`/`updateMetadata`/`getMetadata`, filter predicates, the server
  routes, `RerankingOptions.metadata`/`metadataMap`). Reading a metadata value
  now requires narrowing, which is the point: the values came from `JSON.parse`
  and were never checked.
- Query helpers take `values?: unknown` instead of `any | any[]`, and
  `FilterConfig.value` is `unknown`.
- `PartitionedVectorDBInterface` is fully typed: `search`/`findNearest` take
  `SearchOptions`, `createPartition` takes the new `CreatePartitionOptions`,
  `getActivePartition()` returns `ClusteredVectorDB | null` and
  `getPartitionConfigs()` returns `PartitionConfig[]`.
- `TypedEventEmitter`'s constraint is `object` rather than
  `Record<string, unknown>`, which interfaces without an index signature
  cannot satisfy.
- The `search:complete` payload has one shape (`SearchCompleteEvent`) shared by
  the engine and `Database`, instead of two that disagreed.
- The `Search` engine's `search:complete` event no longer carries `timeMs`: the
  underlying `partition:indexed` event never had it, so it was always
  `undefined`.

### Fixed

- **The monitor recorded the search method as `undefined`.** `Database`
  read `data.dbMethodUsed` off the engine's payload, which is called `method`,
  so every `methodUsage` entry was keyed `"undefined"`.
- **Filters compared across types.** `$gt`/`$gte`/`$lt`/`$lte` used the raw
  `>` operator, so a string metadata value was coerced to a number and compared
  against a numeric filter. Comparisons are now type-bracketed (both numbers or
  both strings), the way MongoDB does it; mixed types never match.
- **`recordError()` spread a primitive `extraData`** character by character
  into the event payload. Only objects are spread now.
- **`partition:indexProgress.progress` is documented as a fraction (0-1)**, not
  a percentage, which is what HNSW reports.
- The HNSW worker restored its state through `as any`; it now assigns the
  private fields directly, and the compiler checks the field names.

### Removed

- `UnifiedSearchStats` — superseded by `SearchStats` and referenced nowhere.
- 227 `any` occurrences across 19 files, and the 2 unused
  `eslint-disable @typescript-eslint/no-unused-vars` directives. `npm run lint`
  now reports 0 problems, down from 229 warnings.
- Dead code: `VectorDB.getStats()` re-checked for a `clusters.dimensions` field
  its own return type requires, `HybridEngineSearch` re-derived `progress`/`id`/
  `error` from payloads that are already typed, and
  `getMetadataWithFieldAcrossPartitions` called `getVector(null)`.

### Added

- Plain repository/npm/stars/forks/issues/license links in place of the five
  shields.io badges. npmjs.com rendered the badge images as alt text on the
  package page, so every one of them was a broken image for anyone reading the
  package there.
- `utils/errors.ts` — `toError`, `errorMessage` and `errorCode`, so
  `useUnknownInCatchVariables` no longer forces an `any` annotation on every
  catch block (`error.code === 'ENOENT'` in particular).
- `LICENSE` (MIT, 2025 n2flowjs). `package.json` declared `"license": "MIT"`
  and shipped `LICENSE` in `files`, and the README links to it, but the file
  was never committed — npm warns on publish and TypeDoc could not resolve the
  link.

### Fixed

- **The release workflow could not publish.** `.github/workflows/npm.yml` ran
  its "Check npm authentication" step under `set -euo pipefail` and echoed
  `$GITHUB_WORKFLOW_PATH` inside the diagnostic message. That variable does not
  exist in the Actions environment, so `set -u` aborted the step on its own
  error output, before `npm publish` was ever reached. Build, tests and the
  tarball checks all passed first. The message now uses `$GITHUB_WORKFLOW` and
  an explicit `WORKFLOW_FILE`.
- `npm run docs` reported 21 warnings, now 0: 17 uses of the unsupported
  `@fires` block tag (TypeDoc's tag is `@event`, so the emitted-event list was
  silently dropped from the API docs), plus an `@description` block tag, a
  `@constructor` tag, and a `@param options.force` that did not match the
  `save(options: SaveOptions)` signature.
- 64 tracked files carried CRLF in the working tree while `.gitattributes`
  pins `* text=auto eol=lf` and every committed blob is LF, so `git status` and
  `npm run format:check` disagreed with the repository. Working tree
  normalized to LF; `.prettierrc` was committed with CRLF and is now LF too.
- `npm run format:check` now passes. 42 files had drifted from the prettier
  config in `.prettierrc` — quote style, wrapping, trailing commas — and
  nothing caught it because CI never runs the check. Formatting only, no
  behaviour change.

## [0.2.0]

Correctness fixes, a smaller published package, and a much stricter build.

### Fixed

- **`close()` silently discarded all data.** `VectorDB.close()` and
  `PartitionedVectorDB.close()` set their closing flag before awaiting the final
  `save()`, and `save()` refused to run once that flag was set — so the "final
  save" never wrote a byte. Three separate layers were involved: the flag
  ordering, an inverted `if (this.isInitialized) return` guard in
  `PartitionedVectorDB.close()` that made closing a healthy database a no-op,
  and `save()` calling `_ensureInitialized()`, which throws while closing.
  `close({ save })` is new for callers that have already flushed state.
- **Filtered HNSW searches always returned no results.** The traversal invoked
  the caller's `filter` with the id only, so a metadata predicate received
  `undefined` and rejected every node. Metadata is now resolved through
  `VectorProvider.getMetadata`.
- **HNSW returned deleted and filtered-out vectors.** The entry point was
  seeded into the result set unconditionally. It is now only seeded when it is
  actually eligible; excluded nodes are still traversed, never returned.
- **HNSW's linear fallback returned soft-deleted vectors** and enforced an
  exact dimension match regardless of the `exactDimensions` option.
- **Deleting one vector destroyed its whole cluster.**
  `ClusteredVectorDB._removeVectorFromCluster` re-read the vector from storage
  after it had already been deleted, always got `undefined`, and took the
  "cluster is empty" branch — orphaning every sibling so they became
  permanently unreachable. `updateVector` also left the id a member of two
  clusters at once.
- **The search cache served results for the wrong filter.** The cache key
  flattened every filter to the string `'present'`, so one predicate's cached
  result was returned for any other. Filters now get distinct identities, and
  the vector hash covers the full vector instead of 16 sampled dimensions.
  The caller's `partitionIds` array is no longer sorted in place.
- **`searchTimeoutMs` was documented but ignored.** It was destructured and
  left as a TODO, so callers who set it got an unbounded search. It now
  applies, with the timer always cleared.
- **Async Express 4 handlers never reached the error middleware.** A throw
  produced an unhandled rejection and the request hung instead of returning
  500. All routes are wrapped.
- **`partition:indexProgress` was declared as `{}`** while the code emitted
  `{ id, progress, operation }`, so the payload was lost to consumers.
- **Cluster "pruning" was a no-op** — every cluster was scanned, making the
  search a full linear scan plus one distance per cluster. New `nprobe` option.
- **`bulkAdd` could spin forever** on a zero-batch-size path that retried
  without changing any state. It now throws.
- **Graph extraction could overflow the stack** (recursive DFS) and
  **stringified numeric ids**, so metadata lookups always missed. It is now
  iterative, keeps the original id type, and is bounded — requests over
  `maxGraphExtractionVectors` are rejected with 413.
- **`req.params.id` was never validated** in the vector routes; a missing id is
  now a 400.

### Added

- `VectorProvider.getMetadata?` so HNSW can evaluate metadata predicates.
- `VectorStoreSearchOptions` — one shared search-options type, replacing the
  `SearchOptions & { exactDimensions?: boolean }` intersections that were
  repeated across HNSW and the store classes.
- `VectorDB.readyPromise` and `ClusteredVectorDB.getStats({ includeClusterMembers })`.
- `SaveOptions` / `CloseOptions`, exposed on `save()` and `close()`.
- `nprobe` on `ClusteredVectorDBOptions`, and `maxGraphExtractionVectors` on
  the server options.
- `.env.example`.

### Changed

- The published tarball is **1.6 MB / 101 files**, down from 71.3 MB / 201
  files. It previously shipped `.env`, the whole test suite, all of `src/`,
  and 67 MB of Windows service daemon logs.
- The publish workflow now builds before publishing and asserts the tarball
  actually contains `dist/index.js` and `dist/index.d.ts`. `dist/` is
  gitignored, so without an explicit build the published `main`/`types`
  pointed at files that did not exist.
- `engines.node` is `>=20.11.0`, matching what the dependency tree actually
  requires. It claimed `>=14`.
- `.env` is no longer tracked in git.

### Removed

- The root `index.js`: dead (`main` points at `dist/index.js`), and it used
  ESM `export *` syntax in a CommonJS package, so running it threw.
- `MetadataIndex` — write-only from the store's perspective and never
  unindexed on delete or update, so it only ever grew.
- `utils/constants.ts` and `server/utils/id_helpers.ts` — unreferenced and not
  part of the public API.
- The `[key: string]: any` index signatures on `SearchResult`,
  `RerankingOptions` and `BatchSearchOptions`, which were hiding real fields.

### Build

- TypeScript 5.4.5 → 5.9.3, with `typescript-eslint` 8.28 → 8.71 and `typedoc`
  0.25 → 0.28. TypeScript 7 is not yet usable: both tools declare peers that
  exclude it.
- Enabled `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`,
  `noPropertyAccessFromIndexSignature`, `noImplicitOverride`,
  `useUnknownInCatchVariables`, `allowUnreachableCode: false` and
  `allowUnusedLabels: false`.
- ESLint runs the type-aware rules. These found two more unhandled promises:
  `Database.markAsReady()` is async but was never awaited, and
  `scheduleSaveConfigs()` had no rejection handler.
- `npm run lint` is no longer broken (`eslint .ts` was not a glob) and reports
  0 errors, down from 683.
- CI runs on pull requests across Node 20, 22 and 24.
- `npm audit`: 17 findings (2 critical, 8 high) → 0. `qs` was the only one
  reachable from production code, via express.
- `.gitattributes` pins `* text=auto eol=lf`; 49 files were renormalized so
  whole-file line-ending diffs no longer appear in `git status`.

### Testing

- 15 new tests covering `close()` persistence, HNSW filtering, cluster deletion
  and the packed-tarball smoke path.
- 135 passing (was 120).
- Test fixtures are written to `os.tmpdir()` instead of the repository, which
  unblocks `mocha --parallel`.

[Unreleased]: https://github.com/n2flowjs/nbase/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/n2flowjs/nbase/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/n2flowjs/nbase/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/n2flowjs/nbase/compare/v0.1.10...v0.2.0
