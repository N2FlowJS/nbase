# Vector Index & Database Types

NBase provides five storage/index abstractions so you can match exact recall, memory footprint, disk paging, and latency tradeoffs:

| Type | Handle | Persistence | Strength |
|---|---|---|---|
| **Single-Partition DB** | `nbase_db_t` | `nbv`, `nbi`, `nbm`, `nbbm`, `wal.nbw` | Exact + HNSW + BM25 + WAL + LRU cache + `mmap` |
| **Multi-Partition DB** | `nbase_pdb_t` | `manifest.nbp` + `p_<id>/` | Bounds RAM via durable LRU partition eviction |
| **Dynamic Clustered DB** | `nbase_clustered_t` | `.nbcl` (`NBASECL1`) | Fast online clustering & `nprobe` partition routing |
| **IVF-PQ Index** | `nbase_ivfpq_t` | `.nbiv` (`NBASEIV1`) | High compression (`~32x`) + coarse-to-fine search |
| **Multi-Probe LSH** | `nbase_lsh_t` | `.nblh` (`NBASELH1`) | Sub-linear cosine (SimHash) & euclidean hash buckets |

## Multi-Partition DB (`nbase_pdb_t`)

- Automatically shards vectors into partitions of size `cfg.partition_capacity`, keeping at most `cfg.max_active_partitions` loaded in memory under `pdb->mu`.
- Eviction atomically saves dirty partitions (`nbv`/`nbi`/`nbm`/`nbbm`) before releasing memory, and reload pulls evicted partitions transparently for point lookups, updates, metadata queries, batch/filter deletions, text/hybrid search, and graph relationship extraction (`nbase_pdb_extract_relationships`).
- `nbase_pdb_stats(pdb, &st)` exposes `total_vectors`, `total_partitions`, `loaded_partitions`, `eviction_count`, `cache_hits`, `cache_misses`, and `memory_bytes`.

## Dynamic Online Clustered DB (`nbase_clustered_t`)

- Assigns incoming vectors to the nearest cluster centroid within `cfg.distance_threshold` (while below `cfg.target_cluster_size`), incrementally updates cluster centroids on `add`, `upsert`, and `delete`, or creates a new cluster up to `cfg.max_clusters`.
- Call `nbase_clustered_recluster(cl, num_clusters)` to run K-Means++ consolidation, and query top `nprobe` nearest clusters via `nbase_clustered_search`.
