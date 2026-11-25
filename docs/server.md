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

- **`GET /health` & `HEAD /health`**: `GET` returns `{"status":"ok","version":"1.14.0","simd":"..."}`; `HEAD` returns status `200` with empty body.
- **`GET /stats`**: returns DB stats (`dim`, `metric`, `active_vectors`, `total_slots`, `tombstones`, `mmap_backed`, `hnsw_*`, `text_docs`, `memory_bytes`, `cache_capacity`, `cache_entries`, `cache_hits`, `cache_misses`, `search_p50_us`, `search_p95_us`, `search_p99_us`).
- **`GET /metrics`**: Prometheus exposition format (`nbase_requests_total`, `nbase_errors_total`, `nbase_search_queries_total`, `nbase_search_latency_microseconds_sum`, `nbase_search_latency_p50_microseconds`, `nbase_search_latency_p95_microseconds`, `nbase_search_latency_p99_microseconds`, `nbase_keepalive_requests_total`, `nbase_vectors_active`, `nbase_vectors_tombstones`, `nbase_hnsw_nodes_active`, `nbase_text_docs_active`, `nbase_memory_bytes`, `nbase_cache_entries`, `nbase_cache_hits_total`, `nbase_cache_misses_total`).
- **`GET /api/vectors`**: paginated sorted vector listing (`?offset=0&limit=50&includeVectors=false&includeMetadata=false&includeText=false`).
- **`GET /api/vectors/count` & `POST /api/vectors/count`**: count active vectors (`POST` accepts optional `"filters"`).
- **`GET /api/vectors/export` & `POST /api/vectors/export`**: export active, ID-selected, or metadata-filtered (`POST` `{"ids":[...], "filters":..., "includeVectors":..., "includeMetadata":..., "includeText":...}`) records as `application/x-ndjson`.
- **`POST /api/vectors/import`**: ingest NDJSON lines (`?upsert=true`, `?threads=4`) including `vector`, `metadata`, and `text`.
- **`POST /api/vectors`**: single insert `{"id": 1, "vector": [...], "metadata": {...}, "text": "..."}` or bulk insert `{"vectors": [...], "threads": 4}`.
- **`POST /api/vectors/upsert`**: single upsert `{"id": 1, "vector": [...], "metadata": {...}, "text": "..."}` or bulk upsert `{"vectors": [...], "threads": 4}`.
- **`POST /api/vectors/get`**: batch fetch `{"ids": [1, 2], "includeVectors": true, "includeMetadata": true, "includeText": false}`.
- **`POST /api/vectors/exists`**: batch ID existence check `{"ids": [1, 2, 99]}` returning per-ID booleans and `existingCount`.
- **`POST /api/vectors/metadata`**, **`PATCH /api/vectors/metadata`**, & **`DELETE /api/vectors/metadata`** (`POST /api/vectors/metadata/delete`): atomic bulk metadata replace (`POST` / `"mode":"set"`), partial update (`PATCH` / `"mode":"update"`) via `{"items": [{"id": 1, "metadata": {...}}]}`, or batch metadata deletion `{"ids": [1, 2]}`.
- **`POST /api/vectors/text`** & **`DELETE /api/vectors/text`** (`POST /api/vectors/text/delete`): atomic bulk text indexing `{"items": [{"id": 1, "text": "..."}]}` or batch text deletion `{"ids": [1, 2]}`.
- **`POST /api/vectors/delete` & `DELETE /api/vectors`**: batch/filter delete via `{"ids": [1, 2]}` or `{"filters": [...]}`.
- **`GET /api/vectors/:id`** (`?includeVector=true&includeMetadata=true&includeText=true`), **`HEAD /api/vectors/:id`** (`200`/`404`), **`PUT /api/vectors/:id`** (upsert), **`PATCH /api/vectors/:id`** (partial update of `vector`, partial-merge `metadata`, and/or `text` / `null` to clear), **`DELETE /api/vectors/:id`**.
- **`GET /api/vectors/:id/exists`**, **`GET/POST /api/vectors/:id/similar`** (`k`, `filters`, `metric`, `mode`, `efSearch`, `skipCache`, `include*`), **`GET/HEAD/PUT/PATCH/DELETE /api/vectors/:id/metadata`**, **`GET/HEAD/PUT/DELETE /api/vectors/:id/text`**.
- **`POST /api/distance`**: computes 1-to-N distances (`{"metric":"euclidean|cosine|...", "query":[...], "vectors":[[...], ...]}`).
- **`POST /api/rerank`**: standalone reranking over `candidates: [{"id": 1, "distance": 0.1}, ...]` supporting `method` (`"diversity"`, `"weighted"`, `"standard"`), `k`, `lambda`, `weights`, `includeMetadata`, `includeVectors`, and `includeText`.
- **`POST /api/search`**: vector k-NN / radius search supporting `query`, `k`, `maxDistance`, `efSearch`, `metric`, `mode` (`"auto"`, `"exact"`, `"hnsw"`), `filters`, `skipCache`, `includeMetadata`, `includeVectors`, `includeText`, `rerank`, `rerankingMethod` (`"standard"`, `"diversity"`, `"weighted"`), `rerankLambda`, and `rerankWeights` (`{"field": weight}`).
- **`POST /api/search/batch`**: multi-query search `{"queries": [[...], [...]], "k": 10, "skipCache": false, "includeMetadata": false, "includeVectors": false, "includeText": false, ...}`.
- **`POST /api/search/radius`**: dedicated distance-threshold search requiring `query` and `maxDistance` (`maxResults`/`limit`, optional `filters`, `skipCache`, `includeMetadata`, `includeVectors`, `includeText`).
- **`POST /api/search/metadata`**: metadata predicate scan (`filters`, `limit`, `includeMetadata`, `includeVectors`, `includeText`).
- **`POST /api/search/text`**: BM25 full-text search (`query`, `k`, `filters`, `include*`).
- **`POST /api/search/text/batch`**: multi-query parallel BM25 full-text search (`queries` array of strings, `k`, optional `threads` `0..64`, `filters`, `includeMetadata`, `includeVectors`, `includeText`).
- **`POST /api/search/hybrid`**: Vector + BM25 RRF fusion (`query`, `text`, `k`, `alpha`, `rrfK`, `skipCache`, `include*`).
- **`POST /api/search/hybrid/batch`**: multi-query parallel hybrid RRF search (`queries` array of `{"query":[...], "text":"..."}`, `k`, `alpha`, `rrfK`, `metric`, `mode`, `efSearch`, optional `threads` `0..64`, `skipCache`, `filters`, `include*`).
- **`POST /api/search/relationships`** & **`POST /api/search/communities`**: k-NN edge extraction and Union-Find community labels (`kNeighbors`, `maxDistance`, optional `filters`).
- **`POST /api/admin/save`**, **`POST /api/admin/checkpoint`**, **`POST /api/admin/compact`**, **`POST /api/admin/rebuild`**, **`POST /api/admin/cache`** (`{"max_entries": 256, "clear": false}`).

## CLI Tool (`nbase-cli`)

- `nbase-cli version`
- `nbase-cli init <dir> --dim <D> [--metric <M>] [--no-hnsw]`
- `nbase-cli info <dir>` — Prints DB dimensions, active count, tombstones, HNSW levels, `text_docs`, `memory_bytes`, and `cache_*` JSON
- `nbase-cli add <dir> <id> <v0,v1,...>` / `upsert <dir> <id> <v0,v1,...>` / `patch <dir> <id> [--vector <v0,v1,...>] [--meta <json>] [--text <str>|--clear-text]` / `get <dir> <id>` / `get-batch <dir> <id1,id2,...>` / `exists <dir> <id1,id2,...>` / `list <dir> [--offset <N>] [--limit <M>]` / `count <dir> [--filter <json>]` / `delete <dir> <id>` / `delete-batch <dir> <id1,id2,...>` / `delete-by-filter <dir> --filter <json>`
- `nbase-cli set-meta <dir> <id> <json-object> [--update]` / `set-meta-batch <dir> <file.ndjson|-> [--update]` / `get-meta <dir> <id>` / `delete-meta <dir> <id>` / `delete-meta-batch <dir> <id1,id2,...>` / `set-text <dir> <id> <text>` / `set-text-batch <dir> <file.ndjson|->` / `get-text <dir> <id>` / `delete-text <dir> <id>` / `delete-text-batch <dir> <id1,id2,...>`
- `nbase-cli search <dir> <q0,q1,...> [--k <K>] [--ef <EF>] [--exact] [--metric <M>] [--max-distance <D>] [--filter <json>]` / `search-batch <dir> <file.ndjson|-> [--k <K>] [--ef <EF>] [--exact] [--metric <M>] [--threads <T>] [--filter <json>]`
- `nbase-cli search-meta <dir> [--filter <json>] [--limit <N>] [--include-vectors] [--include-meta] [--include-text]`
- `nbase-cli rerank <dir> <diversity|weighted|standard> <id1:dist1,id2:dist2,...> [--k <K>] [--lambda <L>] [--weights <json>]`
- `nbase-cli find-similar <dir> <id> [--k <K>] [--ef <EF>] [--exact] [--filter <json>]`
- `nbase-cli relationships <dir> [--k <K>] [--max-distance <D>] [--filter <json>]`
- `nbase-cli communities <dir> [--k <K>] [--max-distance <D>] [--filter <json>]`
- `nbase-cli distance <metric> <q0,q1,...> <v0,v1,...> [<w0,w1,...> ...]`
- `nbase-cli search-text <dir> <query> [--k <K>] [--filter <json>]` / `search-text-batch <dir> <file.ndjson|-> [--k <K>] [--threads <T>] [--filter <json>]` / `search-hybrid <dir> <q0,q1,...|-> <query_text> [--k <K>] [--alpha <A>] [--rrf-k <R>] [--metric <M>] [--filter <json>]` / `search-hybrid-batch <dir> <file.ndjson|-> [--k <K>] [--alpha <A>] [--rrf-k <R>] [--metric <M>] [--threads <T>] [--filter <json>]`
- `nbase-cli rebuild <dir> [--threads <T>]` / `compact <dir>` / `checkpoint <dir>`
- `nbase-cli import <dir> <file.ndjson|-> [--upsert] [--threads <T>]` / `export <dir> [file.ndjson|-] [--ids <id1,id2,...>] [--filter <json>]`
