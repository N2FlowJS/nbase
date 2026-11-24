# HTTP/1.1 Server (`nbased`) & CLI (`nbase-cli`)

## Standalone HTTP/1.1 Server (`nbased`)

`nbased` wraps `nbase_db_t` + WAL persistence behind a POSIX multi-threaded HTTP/1.1 Keep-Alive server (`src/http_server.c`) using `poll` + bounded worker pool (`NBASE_HTTP_QUEUE_CAP = 256`, `NBASE_HTTP_MAX_KEEPALIVE_REQS = 100`).

### CLI Flags & Environment Variables

| CLI Flag | Env Variable | Legacy Fallback | Default |
|---|---|---|---|
| `--host <ip>` | `NBASE_HOST` | `HOST` | `127.0.0.1` (`0.0.0.0` in Docker) |
| `--port <port>` | `NBASE_PORT` | `PORT` | `3000` (`1307` in Docker) |
| `--data-dir <dir>` | `NBASE_DATA_DIR` | `NBASE_DB_PATH` | *(in-memory)* |
| `--dim <dim>` | `NBASE_DIM` | `NBASE_VECTOR_SIZE` | `128` |
| `--metric <name>` | `NBASE_METRIC` | — | `euclidean` |
| `--workers <n>` | `NBASE_WORKERS` | — | `4` (`1..32`) |
| `--cache-size <n>` | `NBASE_CACHE_SIZE` | — | `0` (LRU disabled) |
| `--api-key <key>` | `NBASE_API_KEY` | `NBASE_API_KEYS` | *(auth off)* |
| `--rate-limit <rps>` | `NBASE_RATE_LIMIT` | — | `0` (off) |
| `--cors-origin <orig>` | `NBASE_CORS_ORIGIN` | `NBASE_CORS_ORIGINS` | *(disabled)* |

Authentication uses constant-time `Authorization: Bearer <key>` comparison on all routes except `GET /health` and `GET /metrics`. `SIGTERM` / `SIGINT` drains active keep-alive connections cleanly and runs `nbase_wal_checkpoint(db)` when `--data-dir` is enabled.

### Endpoints

- **`GET /health`**: returns `{"status":"ok","version":"1.4.0","simd":"..."}`.
- **`GET /stats`**: returns DB stats (`dim`, `metric`, `active_vectors`, `total_slots`, `tombstones`, `mmap_backed`, `hnsw_*`, `text_docs`, `memory_bytes`, `cache_capacity`, `cache_entries`, `cache_hits`, `cache_misses`, `search_p50_us`, `search_p95_us`, `search_p99_us`).
- **`GET /metrics`**: Prometheus exposition format (`nbase_requests_total`, `nbase_errors_total`, `nbase_search_queries_total`, `nbase_search_latency_microseconds_sum`, `nbase_search_latency_p50_microseconds`, `nbase_search_latency_p95_microseconds`, `nbase_search_latency_p99_microseconds`, `nbase_keepalive_requests_total`, `nbase_vectors_active`, `nbase_vectors_tombstones`, `nbase_hnsw_nodes_active`, `nbase_text_docs_active`, `nbase_memory_bytes`, `nbase_cache_entries`, `nbase_cache_hits_total`, `nbase_cache_misses_total`).
- **`GET /api/vectors`**: paginated sorted vector listing (`?offset=0&limit=50&includeVectors=false&includeMetadata=false&includeText=false`).
- **`GET /api/vectors/count` & `POST /api/vectors/count`**: count active vectors (`POST` accepts optional `"filters"`).
- **`POST /api/vectors`**: single insert `{"id": 1, "vector": [...], "metadata": {...}, "text": "..."}` or bulk insert `{"vectors": [...], "threads": 4}`.
- **`POST /api/vectors/upsert`**: single upsert `{"id": 1, "vector": [...], "metadata": {...}, "text": "..."}` or bulk upsert `{"vectors": [...], "threads": 4}`.
- **`POST /api/vectors/get`**: batch fetch `{"ids": [1, 2], "includeVectors": true, "includeMetadata": true, "includeText": false}`.
- **`POST /api/vectors/metadata`** & **`PATCH /api/vectors/metadata`**: atomic bulk metadata replace (`POST` / `"mode":"set"`) or partial update (`PATCH` / `"mode":"update"`) via `{"items": [{"id": 1, "metadata": {...}}]}`.
- **`POST /api/vectors/text`**: atomic bulk text indexing `{"items": [{"id": 1, "text": "..."}]}`.
- **`POST /api/vectors/delete` & `DELETE /api/vectors`**: batch/filter delete via `{"ids": [1, 2]}` or `{"filters": [...]}`.
- **`GET /api/vectors/:id`** (`?includeVector=true&includeMetadata=true&includeText=true`), **`PUT /api/vectors/:id`** (upsert), **`DELETE /api/vectors/:id`**.
- **`GET /api/vectors/:id/exists`**, **`GET /api/vectors/:id/similar?k=10`**, **`GET/PUT/PATCH/DELETE /api/vectors/:id/metadata`**, **`GET/PUT/DELETE /api/vectors/:id/text`**.
- **`POST /api/search`**: vector k-NN / radius search supporting `query`, `k`, `maxDistance`, `efSearch`, `metric`, `mode` (`"auto"`, `"exact"`, `"hnsw"`), `filters`, `skipCache`, `includeMetadata`, `includeVectors`, `includeText`, `rerank`, `rerankingMethod` (`"standard"`, `"diversity"`, `"weighted"`), `rerankLambda`, and `rerankWeights` (`{"field": weight}`).
- **`POST /api/search/batch`**: multi-query search `{"queries": [[...], [...]], "k": 10, "skipCache": false, ...}`.
- **`POST /api/search/metadata`**: metadata predicate scan (`filters`, `limit`, `includeMetadata`, `includeVectors`, `includeText`).
- **`POST /api/search/text`**: BM25 full-text search (`query`, `k`, `filters`, `include*`).
- **`POST /api/search/hybrid`**: Vector + BM25 RRF fusion (`query`, `text`, `k`, `alpha`, `rrfK`, `skipCache`, `include*`).
- **`POST /api/search/relationships`** & **`POST /api/search/communities`**: k-NN edge extraction and Union-Find community labels (`kNeighbors`, `maxDistance`, optional `filters`).
- **`POST /api/admin/save`**, **`POST /api/admin/checkpoint`**, **`POST /api/admin/compact`**, **`POST /api/admin/rebuild`**, **`POST /api/admin/cache`** (`{"max_entries": 256, "clear": false}`).

## CLI Tool (`nbase-cli`)

- `nbase-cli version`
- `nbase-cli init <dir> --dim <D> [--metric <M>] [--no-hnsw]`
- `nbase-cli info <dir>` — Prints DB dimensions, active count, tombstones, HNSW levels, `text_docs`, `memory_bytes`, and `cache_*` JSON
- `nbase-cli add <dir> <id> <v0,v1,...>` / `upsert <dir> <id> <v0,v1,...>` / `get <dir> <id>` / `get-batch <dir> <id1,id2,...>` / `list <dir> [--offset <N>] [--limit <M>]` / `count <dir> [--filter <json>]` / `delete <dir> <id>` / `delete-batch <dir> <id1,id2,...>`
- `nbase-cli set-meta <dir> <id> <json-object> [--update]` / `get-meta <dir> <id>` / `delete-meta <dir> <id>` / `set-text <dir> <id> <text>` / `get-text <dir> <id>` / `delete-text <dir> <id>`
- `nbase-cli search <dir> <q0,q1,...> [--k <K>] [--ef <EF>] [--exact] [--max-distance <D>]`
- `nbase-cli search-text <dir> <query> [--k <K>]` / `search-hybrid <dir> <q0,q1,...|-> <query_text> [--k <K>] [--alpha <A>] [--rrf-k <R>]`
- `nbase-cli rebuild <dir> [--threads <T>]` / `compact <dir>` / `checkpoint <dir>`
- `nbase-cli import <dir> <file.ndjson|-> [--threads <T>]` / `export <dir> [file.ndjson|-]`
