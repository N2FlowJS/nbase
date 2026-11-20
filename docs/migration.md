# Migration Guide: `@n2flowjs/nbase` (`v0.3.x` Node.js) to `nbase v1.0.0` (Pure C11 / C++17)

`nbase v1.0.0` replaces the Node.js/TypeScript runtime with the zero-dependency C11/C++17 engine (`libnbase`, `nbased`, `nbase-cli`) for native speed, predictable memory bounds, SIMD vector kernels, WAL crash safety, and compact Docker deployment.

## 1. Docker & Environment Variable Mapping

Existing REST server deployments using Node.js environment variables continue to work seamlessly because `nbased` reads legacy fallbacks automatically:

| Node.js (`v0.3.x`) Env Var | `nbased v1.0.0` Primary Var | Notes |
|---|---|---|
| `HOST` | `NBASE_HOST` | Fallback honored (`0.0.0.0` in Docker) |
| `PORT` | `NBASE_PORT` | Fallback honored (`1307` in Docker) |
| `NBASE_DB_PATH` | `NBASE_DATA_DIR` | Fallback honored (`/data` in Docker) |
| `NBASE_VECTOR_SIZE` | `NBASE_DIM` | Fallback honored (default `128`) |
| `NBASE_API_KEYS` | `NBASE_API_KEY` | Single Bearer token in `v1.0.0` |
| `NBASE_CORS_ORIGINS` | `NBASE_CORS_ORIGIN` | Fallback honored |
| *(lru-cache max)* | `NBASE_CACHE_SIZE` / `--cache-size` | LRU query cache entries (`256` in Compose) |

## 2. REST API Differences

- **IDs**: Vector IDs in `nbase v1.0.0` are 64-bit unsigned integers (`uint64_t`, including `0` and `UINT64_MAX`) passed as JSON numbers or unsigned numeric strings (`"123"`).
- **Endpoints preserved**:
  - `GET /health`, `GET /stats`, `GET /metrics` (now Prometheus format with P50/P95/P99 latency and `nbase_memory_bytes` / `nbase_cache_*`)
  - `POST /api/vectors`, `GET/PUT/DELETE /api/vectors/:id`, `GET /api/vectors/:id/exists`, `GET /api/vectors/:id/similar`, `PATCH /api/vectors/:id/metadata`
  - `POST /api/search` (supports `rerank`, `rerankingMethod: "weighted"|"diversity"|"standard"`, `rerankWeights`, `skipCache`), `POST /api/search/batch`, `POST /api/search/metadata`, `POST /api/search/relationships`, `POST /api/search/communities`
- **New REST endpoints in `v1.0.0`**:
  - `POST /api/vectors/delete` & `DELETE /api/vectors` (bulk deletion by `ids` or metadata `filters`)
  - `PUT/DELETE /api/vectors/:id/text`, `POST /api/search/text`, `POST /api/search/hybrid`
  - `POST /api/admin/{save,checkpoint,compact,rebuild,cache}`

## 3. Code API Mapping (TypeScript -> C++17 / C11)

| TypeScript (`v0.3.x`) | C++17 (`include/nbase.hpp`) | C11 (`include/nbase.h`) |
|---|---|---|
| `new Database(opts)` | `nbase::Database::create(cfg)` / `open_wal(dir)` | `nbase_create` / `nbase_open_wal` |
| `db.addVector(id, vec, meta)` | `db.add(id, vec)` + `db.set_meta(...)` | `nbase_add` + `nbase_set_meta` |
| `db.bulkAdd(items)` | `db.add_batch(ids, vecs, dim, threads)` | `nbase_add_batch` |
| `db.findNearest(q, k, opts)` | `db.search(q, k, &opts)` | `nbase_search` |
| `db.batchSearch(queries, opts)` | `pdb.search_batch(...)` | `nbase_search_batch` / `nbase_pdb_search_batch` |
| `new PartitionedVectorDB(...)` | `nbase::PartitionedDatabase::open(...)` | `nbase_pdb_open` |
| `new ClusteredVectorDB(...)` | `nbase::ClusteredDatabase::create(...)` | `nbase_clustered_create` |

## 4. Migrating Data via NDJSON

Export vectors, metadata, and text fields from legacy stores as newline-delimited JSON (`{"id": 1, "vector": [...], "metadata": {...}, "text": "..."}`) and bulk-load into `nbase v1.0.0`:

```bash
nbase-cli init ./data_v1 --dim 128 --metric euclidean
nbase-cli import ./data_v1 export.ndjson --threads 4
```
