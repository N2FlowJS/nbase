# nbase / libnbase (Pure C11 / C++17 Vector, Graph & Full-Text Engine — v1.18.0)

`nbase` is a standalone, zero-dependency, pure-C11 (with header-only C++17 RAII wrapper [`include/nbase.hpp`](include/nbase.hpp)) embedded vector/graph/full-text database engine (`libnbase.a` / `libnbase.so`), persistent HTTP/1.1 Keep-Alive REST daemon (`nbased`), and streaming NDJSON CLI (`nbase-cli`) with AVX2/NEON SIMD distance & quantization kernels, flat `mmap` storage, CRC32 Write-Ahead Log recovery, thread-safe LRU query result cache, typed binary metadata filtering & batch/filter deletion, DB-integrated & partitioned BM25 full-text + Hybrid RRF batch search, Dynamic Online Clustered Vector DB (`.nbcl`), IVF-PQ (parallel batch encoding), PQ, SIMD SQ8 (AVX2 encode/decode/distance), 1-bit Binary Quantization (`BQ`), multi-probe LSH, lock-striped parallel HNSW build & search, deterministic parallel K-Means++, MMR & metadata-weighted reranking, filtered k-NN graph & Union-Find community analytics, multi-partition durable LRU storage with full batch/rerank/graph parity & `memory_bytes` telemetry, Prometheus P50/P95/P99 `/metrics`, CORS & rate limiting, strict RFC 8259 JSON, installable `pkg-config` packaging, and native Docker deployment.

## Architecture & Core Features (`v1.18.0`)

- **C11 Public API** ([`include/nbase.h`](include/nbase.h)):
  - Single-DB handle (`nbase_db_t`): `create`, `open`, `open_mmap`, `open_wal`, `wal_enable`, `wal_checkpoint`, `save`, `close`, `add`, `add_batch`, `upsert`, `upsert_batch`, `delete`, `delete_batch`, `delete_by_filter`, `get`, `get_batch`, `contains`, `count`, `list_ids`, `set_meta`, `set_meta_batch`, `update_meta`, `update_meta_batch`, `delete_meta`, `delete_meta_batch`, `set_text`, `set_text_batch`, `delete_text`, `delete_text_batch`, `search`, `search_radius`, `find_similar`, `search_batch`, `search_text`, `search_text_batch`, `search_hybrid`, `search_hybrid_batch`, `rebuild_hnsw`, `rebuild_hnsw_mt`, `compact`, `cache_configure`, `cache_clear`, `stats` (with `text_docs`, `memory_bytes`, `cache_capacity`, `cache_entries`, `cache_hits`, `cache_misses`).
  - Thread-safe read/write concurrency (`pthread_rwlock_t`) + mutex-protected LRU query result cache (`cfg.cache_capacity`, `opts.skip_cache`, automatic invalidation on mutations) + parallel `pthreads` work-stealing `nbase_search_batch` (`opts.num_threads`) + lock-striped parallel HNSW construction (`nbase_add_batch`, `nbase_rebuild_hnsw_mt`).
- **Write-Ahead Log & Crash Recovery** (`src/wal.c`): CRC32-framed binary log `wal.nbw` (`NBASEWL1`), auto-replay on `nbase_open`/`nbase_open_wal`, single-`fsync` vector/metadata/text batch logging, torn-write tail detection and truncation, and atomic `nbase_wal_checkpoint`.
- **SIMD Distance Kernels** (`src/distance.c`): All 7 metrics (`euclidean`, `squared_euclidean`, `cosine`, `inner_product`, `manhattan`, `chebyshev`, `hamming`) resolved via direct function pointer (`nbase_get_distance_fn`) plus 1-to-N batch distance (`nbase_distance_batch`) with runtime-dispatched **AVX2+FMA** on x86_64, **NEON** on ARM64, and a portable scalar fallback.
- **Flat Vector Store, Compaction & `mmap`** (`src/store.c`): Contiguous `float32` storage, open-addressing `uint64_t -> slot` hash table supporting full `uint64_t` range (including `0` and `UINT64_MAX`), tombstone deletion, `NaN`/`Inf` validation, physical compaction (`nbase_compact`), and zero-copy read-only `nbase_open_mmap`.
- **HNSW Index** (`src/hnsw.c`): Slot-indexed HNSW with Algorithm 4 heuristic pruning, 256-stripe fine-grained node locks for parallel batch construction, thread-safe per-query generation-stamped traversal state, `ef_search` override, per-search metric override, callback + metadata filter predicates, persistent tombstones, deterministic seeded PRNG, and rebuild.
- **Typed Binary Metadata & Filtering** (`src/meta.c`): Per-vector typed attributes (`int64`, `double`, `bool`, `string`) with `set_meta`, `set_meta_batch`, `update_meta`, `update_meta_batch`, `get_meta`, `delete_meta`, preserved across `upsert`, strict type-bracketed operators (`EQ`, `NE`, `GT`, `GTE`, `LT`, `LTE`, `EXISTS`, `CONTAINS`), and atomic `meta.nbm` persistence.
- **IVF-PQ Compressed Approximate Index, K-Means++ & PQ** (`src/kmeans.c`, `src/pq.c`, `src/ivfpq.c`): Coarse K-Means++ inverted lists + residual subspace PQ encoding (`nbase_ivfpq_train`, `add`, `upsert`, `upsert_batch`, `delete`, `delete_batch`, `contains`, `list_ids`, `compact`), thread-safe `pthread_rwlock_t` concurrency, 4-code batched ADC (`nbase_pq_adc_distance_batch`) & SIMD dot-LUT / coarse multi-probe Asymmetric Distance Computation (`nbase_ivfpq_search`, `nbase_ivfpq_search_filtered`, `nbase_ivfpq_search_radius`, `search_batch`), and atomic `.nbiv` (`NBASEIV1`) persistence, alongside standalone K-Means++ and PQ (`nbq` / `NBASEPQ1`).
- **MMR Diversity & Weighted Reranking & Graph Analytics** (`src/rerank.c`): `nbase_rerank_mmr` (Maximal Marginal Relevance balancing relevance and candidate diversity), `nbase_rerank_weighted` (metadata-weighted candidate score adjustment), parallel `nbase_extract_relationships` / `nbase_extract_relationships_filtered` (k-NN graph edge extraction), and iterative Union-Find `nbase_extract_communities`.
- **Multi-Partition DB with Durable LRU Cache & Full Parity** (`src/partitioned.c`): `nbase_pdb_t` with automatic partition rollover (`partition_capacity`), durable LRU eviction (`max_active_partitions`), `add_batch`, `upsert_batch`, `get_batch`, `delete_batch`, `delete_by_filter`, `count`, `list_ids`, `search`, `search_radius`, `search_batch`, `rerank_mmr`, `rerank_weighted`, `extract_relationships` (filtered), cross-partition metadata (`set_meta_batch`, `update_meta_batch`, `query_meta`, `delete_meta`, `delete_meta_batch`, `foreach_meta`), text & hybrid search (`set_text`, `set_text_batch`, `get_text`, `get_text_batch`, `delete_text`, `delete_text_batch`, `search_text`, `search_text_batch`, `search_hybrid`, `search_hybrid_batch`), `find_similar`, `compact`, `rebuild_hnsw_mt`, cache `cache_hits`/`cache_misses` & `memory_bytes` telemetry, and atomic `manifest.nbp` persistence.
- **Dynamic Online Clustered Vector Index (`.nbcl`)** (`src/clustered.c`): thread-safe (`pthread_rwlock_t`) online distance-threshold & `target_cluster_size` cluster assignment with incremental centroid updates, `add_batch`, `upsert_batch`, `delete_batch`, `get_batch`, `list_ids`, K-Means++ `nbase_clustered_recluster`, `nbase_clustered_compact`, `nprobe` search, `search_filtered`, `search_radius`, `find_similar`, parallel `search_batch`, `memory_bytes` stats, and atomic CRC32-framed `.nbcl` (`NBASECL1`) persistence.
- **Scalar Quantization (`SQ8`), 1-bit Binary Quantization (`BQ`) & Locality-Sensitive Hashing (`LSH`)** (`src/sq8.c`, `src/bq.c`, `src/lsh.c`):
  - Per-dimension uint8 quantization with runtime-dispatched AVX2+FMA single/batched asymmetric (`nbase_sq8_l2_sq_asymmetric_batch`) & symmetric (`nbase_sq8_l2_sq_symmetric`, `nbase_sq8_l2_sq_symmetric_batch`) L2 estimators and `.nbsq` (`NBASESQ1`) persistence.
  - 1-bit mean-threshold packing (`nbase_bq_train`, `encode` AVX2, `nbase_bq_hamming`, `nbase_bq_hamming_batch` 4-row `popcnt`) with 32x compression and `.nbbq` (`NBASEBQ1`) persistence.
  - Thread-safe (`pthread_rwlock_t`) SIMD-batched multi-table multi-probe SimHash (`cosine`) and p-stable (`euclidean`) LSH (`nbase_lsh_create`, `add`, `add_batch`, `upsert`, `upsert_batch`, `get`, `get_batch`, `contains`, `delete`, `delete_batch`, `list_ids`, `compact`, `search`, `search_filtered`, `search_radius`, `find_similar`, parallel `search_batch`) with `.nblh` (`NBASELH1`) persistence.
- **DB-Integrated BM25 Full-Text & Hybrid RRF Search** (`src/bm25.c`, `src/nbase.c`):
  - Thread-safe (`pthread_rwlock_t`) Okapi BM25 inverted index (`nbase_bm25_create`, `index`, `index_batch`, `get`, `get_batch`, `contains`, `delete`, `delete_batch`, `list_ids`, `compact`, `stats`, `search`, `search_filtered`, parallel `search_batch`, `.nbbm` persistence) + Reciprocal Rank Fusion (`nbase_hybrid_rrf`).
  - **Integrated & Parallel Text / Hybrid Search**: DB-integrated full-text & RRF batch search (`nbase_set_text`, `nbase_set_text_batch`, `nbase_get_text`, `nbase_get_text_batch`, `nbase_delete_text`, `nbase_delete_text_batch`, `nbase_search_text`, `nbase_search_text_batch`, `nbase_search_hybrid`, `nbase_search_hybrid_batch`) with WAL durability (`SET_TEXT` / `DELETE_TEXT`) and automatic `bm25.nbbm` save/load.
  - Vector metadata enumeration/queries (`nbase_foreach_meta`, `nbase_query_meta`) and find-similar (`nbase_find_similar`).
- **Embeddable HTTP/1.1 Keep-Alive Server (`src/http_server.c`), Prometheus `/metrics`, Strict JSON (`src/json.c`), `nbased` & `nbase-cli`** (`apps/nbased.c`, `apps/nbase_cli.c`, `Dockerfile`):
  - Pure POSIX threaded HTTP/1.1 REST API (`poll` accept loop + worker pool, persistent HTTP/1.1 Keep-Alive connection reuse, clean non-blocking shutdown, optional Bearer API key, CORS preflight, per-second `/api/*` rate limit `429`, Prometheus `GET /metrics` with rolling P50/P95/P99 latency, `nbase_memory_bytes`, `nbase_cache_{entries,hits_total,misses_total}`, `GET/POST/PUT/PATCH/HEAD/DELETE /api/vectors`, `GET/POST /api/vectors/count`, `POST /api/vectors/exists`, `GET/POST /api/vectors/export`, `POST /api/vectors/import`, `POST /api/vectors/upsert`, `PATCH/POST /api/vectors/patch`, `POST /api/vectors/get`, `POST/PATCH/DELETE /api/vectors/metadata`, `POST /api/vectors/metadata/{get,delete}`, `POST/DELETE /api/vectors/text`, `POST /api/vectors/text/{get,delete}`, `POST /api/vectors/delete` & `DELETE /api/vectors`, `GET/HEAD/PUT/DELETE /api/vectors/:id/text`, `GET/POST /api/vectors/:id/similar`, `GET/HEAD/PUT/PATCH/DELETE /api/vectors/:id/metadata`, `POST /api/distance`, `POST /api/rerank`, `POST /api/search` supporting `maxDistance`/`rerankingMethod`/`rerankWeights`/`skipCache`, `/api/search/batch` (with `include*` flags), `/api/search/similar`, `/api/search/radius`, `/api/search/metadata`, `/api/search/text`, `/api/search/text/batch` (`"threads"`), `/api/search/hybrid`, `/api/search/hybrid/batch` (`"threads"`), `/api/search/relationships` & `/api/search/communities` with `"filters"`, `/api/admin/{save,checkpoint,compact,rebuild,cache}`, `GET/HEAD /health`, `/stats`).
  - Standalone WAL-backed daemon `nbased`, streaming NDJSON `import --upsert`/`export --ids --filter`, `upsert`, `patch`, `patch-batch`, `get-batch`, `exists`, `list --include-*`, `count`, `delete-by-filter`, `set-meta`, `set-meta-batch`, `get-meta`, `get-meta-batch`, `delete-meta`, `delete-meta-batch`, `set-text`, `set-text-batch`, `get-text`, `get-text-batch`, `delete-text`, `delete-text-batch`, `search --max-distance --filter --metric --json --include-*`, `search-batch`, `search-text-batch`, `search-hybrid-batch`, `search-meta`, `rerank`, `find-similar --metric --json --include-*`, `relationships`, `communities`, `distance`, `delete-batch` & offline CLI `nbase-cli`, complete C++17 RAII wrapper [`include/nbase.hpp`](include/nbase.hpp), and multi-stage native Docker image ([`Dockerfile`](Dockerfile)).
- **Crash-Safe Binary Formats (Headers + IEEE 802.3 CRC32)**:
  - `vectors.nbv` (`NBASEVC1`), `hnsw.nbi` (`NBASEHN1`), `meta.nbm` (`NBASEMT1`), `wal.nbw` (`NBASEWL1`), PQ (`NBASEPQ1`), IVF-PQ (`.nbiv` / `NBASEIV1`), SQ8 (`.nbsq` / `NBASESQ1`), BQ (`.nbbq` / `NBASEBQ1`), BM25 (`.nbbm` / `NBASEBM1`), LSH (`.nblh` / `NBASELH1`), Clustered (`.nbcl` / `NBASECL1`), and partition manifests (`manifest.nbp` / `NBASEPM1`).

## Building, Installing & Testing

```bash
# Release build & install (builds libnbase.a, libnbase.so.1 -> libnbase.so.1.18.0, nbased, nbase-cli, test_nbase, bench_nbase, libnbase.pc, nbase.hpp)
cmake -S . -B /tmp/nbase-release -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/nbase-release -j
ctest --test-dir /tmp/nbase-release --output-on-failure
/tmp/nbase-release/bench_nbase
cmake --install /tmp/nbase-release --prefix /tmp/nbase-stage

# Debug build with AddressSanitizer + UndefinedBehaviorSanitizer
cmake -S . -B /tmp/nbase-asan -DCMAKE_BUILD_TYPE=Debug -DNBASE_ENABLE_SANITIZERS=ON
cmake --build /tmp/nbase-asan -j
ctest --test-dir /tmp/nbase-asan --output-on-failure

# Debug build with ThreadSanitizer (mutually exclusive with ASan)
cmake -S . -B /tmp/nbase-tsan -DCMAKE_BUILD_TYPE=Debug -DNBASE_ENABLE_TSAN=ON
cmake --build /tmp/nbase-tsan -j
ctest --test-dir /tmp/nbase-tsan --output-on-failure
```

## Running the REST Server (`nbased`) & Docker

```bash
# Start natively with WAL persistence and LRU query cache
./nbased --host 0.0.0.0 --port 1307 --data-dir ./database --dim 128 --workers 4 --cache-size 256

# Or via Docker Compose (exposes port 1307 with persistent volume /data)
docker compose up -d --build
```

## Example Usage (Metadata Filter & Parallel Batch Search)

```c
#include "nbase.h"
#include <stdio.h>

int main(void) {
  nbase_config_t cfg = nbase_config_default(4);
  nbase_db_t *db = NULL;
  if (nbase_create(&cfg, &db) != NBASE_OK) return 1;

  float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
  nbase_add(db, 1, v1, 4);
  nbase_add(db, 2, v2, 4);

  nbase_meta_field_t meta1[2] = {
      {"category", {.type = NBASE_META_STRING, .as = {.str = "ml"}}},
      {"score", {.type = NBASE_META_INT64, .as = {.i64 = 95}}},
  };
  nbase_set_meta(db, 1, meta1, 2);

  nbase_meta_cond_t cond = {"score", NBASE_OP_GTE, {.type = NBASE_META_INT64, .as = {.i64 = 90}}};
  nbase_meta_filter_t filter = {&cond, 1};
  nbase_search_opts_t opts = nbase_search_opts_default();
  opts.meta_filter = &filter;
  opts.num_threads = 4;

  nbase_result_t hits[2];
  uint32_t count = 0;
  float query[4] = {0.9f, 0.1f, 0.0f, 0.0f};
  if (nbase_search(db, query, 4, 2, &opts, hits, &count) == NBASE_OK) {
    for (uint32_t i = 0; i < count; i++) {
      printf("id=%llu dist=%.4f\n", (unsigned long long)hits[i].id, hits[i].distance);
    }
  }

  nbase_save(db, "/tmp/my_nbase_db");
  nbase_close(db);
  return 0;
}
```
