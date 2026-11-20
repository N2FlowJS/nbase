# Database Lifecycle, Durability & Caching

## Opening & Creating (`nbase_db_t`)

- **`nbase_create(&cfg, &out_db)`**: Initializes an in-memory vector store, metadata store, optional HNSW graph (`cfg.enable_hnsw`), RW lock, and LRU query cache (`cfg.cache_capacity`).
- **`nbase_save(db, dir_path)`**: Atomically writes `vectors.nbv`, `hnsw.nbi` (when HNSW is enabled), `meta.nbm` (when metadata exists), and `bm25.nbbm` (when text docs exist).
- **`nbase_open(dir_path, &out_db)`**: Loads persisted snapshots into memory and replays `wal.nbw` if present.
- **`nbase_open_mmap(dir_path, &out_db)`**: Maps `vectors.nbv` read-only with zero heap copy for vector payloads (`mmap_backed = true`; write/mutation operations return `NBASE_ERR_UNSUPPORTED`).
- **`nbase_open_wal(dir_path, &out_db)`**: Opens directory, replays any uncheckpointed `wal.nbw` frames, and keeps `wal.nbw` attached so all subsequent mutations are crash-safe.

## Write-Ahead Log (`src/wal.c`)

- Call `nbase_wal_enable(db, dir_path)` on a live DB to write a baseline checkpoint and begin appending binary CRC32-protected `nbw_frame_hdr_t` frames (`ADD`, `UPSERT`, `DELETE`, `SET_META`, `UPDATE_META`, `DELETE_META`, `COMPACT`, `SET_TEXT`, `DELETE_TEXT`).
- Partial/torn tail frames left by power loss or mid-write crashes are automatically detected and truncated cleanly on replay.
- `nbase_wal_checkpoint(db)` atomically persists full state and resets `wal.nbw`.

## Compaction & Index Maintenance

- **`nbase_compact(db)`**: Reclaims tombstone slots (`NBASE_SLOT_TOMBSTONE`) inside `nbase_store_t`, shrinks contiguous arrays, and rebuilds the HNSW graph cleanly.
- **`nbase_rebuild_hnsw(db)`** / **`nbase_rebuild_hnsw_mt(db, num_threads)`**: Drops deleted-node tombstones from the HNSW graph and inserts active nodes in parallel across 256 lock stripes.

## Thread-Safe LRU Query Result Cache

- Configure either at startup (`cfg.cache_capacity`) or at runtime (`nbase_cache_configure(db, max_entries)`). Set `max_entries = 0` to disable and free cached results, or `nbase_cache_clear(db)` to flush entries.
- Cache keys encode `(dim, k, ef_search, mode, metric, query_bytes, serialized_meta_filter)` and bypass caching when `opts.skip_cache = true` or a C function pointer `opts.filter != NULL` is supplied.
- Mutations (`add`, `add_batch`, `upsert`, `delete`, `delete_batch`, `delete_by_filter`, `set_meta`, `update_meta`, `delete_meta`, `compact`, `rebuild_hnsw*`) automatically invalidate the query cache so stale results are never served.
- Strict lock order prevents deadlocks: `pthread_rwlock_*(&db->rwlock)` -> `pthread_mutex_lock(&db->cache_lock)`.

## Memory Accounting

`nbase_stats(db, &st)` computes `st.memory_bytes` across active store buffers, HNSW adjacency lists/metadata, typed metadata tables, BM25 inverted index allocations, and live LRU cache entries.
