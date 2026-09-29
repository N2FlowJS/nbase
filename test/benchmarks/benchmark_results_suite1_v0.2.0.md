# PartitionedVectorDB Benchmark Results - Suite 1 - v0.2.0

*Run at: 2026-09-29T15:25:59.332Z*

## Configuration
- Total Vectors: 50000
- Vector Dimension: 128
- Partition Capacity: 10000
- Max Active Partitions: 3
- Search K: 10
- Bulk Add Chunk Size: 5000

## Results

| Operation | Total Time (ms) | Average Time (ms) |
|-----------|----------------:|------------------:|
| DB Initialization | 12.42 | 12.42 |
| Bulk Add Batch 1 (5000 vectors) | 3541.73 | 3541.73 |
| Bulk Add Batch 2 (5000 vectors) | 11156.08 | 11156.08 |
| Bulk Add Batch 3 (5000 vectors) | 3477.31 | 3477.31 |
| Bulk Add Batch 4 (5000 vectors) | 11326.63 | 11326.63 |
| Bulk Add Batch 5 (5000 vectors) | 3482.99 | 3482.99 |
| Bulk Add Batch 6 (5000 vectors) | 11207.39 | 11207.39 |
| Bulk Add Batch 7 (5000 vectors) | 3465.18 | 3465.18 |
| Bulk Add Batch 8 (5000 vectors) | 11233.41 | 11233.41 |
| Bulk Add Batch 9 (5000 vectors) | 3468.92 | 3468.92 |
| Bulk Add Batch 10 (5000 vectors) | 11371.10 | 11371.10 |
| Total Bulk Add | 74062.13 | 7406.21 |
| Standard FindNearest | 55.42 | 55.42 |
| Total HNSW Build | 277449.55 | 92483.18 |
| HNSW FindNearest | 20.27 | 20.27 |
| DB Save | 1317.80 | 1317.80 |
| DB Close | 1108.70 | 1108.70 |
| DB Re-Load | 299.30 | 299.30 |
| HNSW FindNearest After Re-Load | 8.19 | 8.19 |

## Search Performance Summary

### Standard vs HNSW Search Comparison

| Search Method | Time (ms) | Speedup Factor |
|---------------|----------:|---------------:|
| Standard Search | 55.42 | 1.00x |
| HNSW Search | 20.27 | 2.73x |
| HNSW Search (After Reload) | 8.19 | 6.77x |

**Note**: HNSW search is faster by a factor of 2.73x.

## Database Stats

- Total partitions: 5
- Loaded partitions: 3
- Total vectors: 50000
- HNSW indices: 0

## Comparison with v0.1.10

Same machine, same configuration (50 000 vectors, dim 128, capacity 10 000,
3 active partitions). v0.1.10 figures are from 2026-06-05.

| Operation | v0.1.10 | v0.2.0 | Change |
| --- | ---: | ---: | ---: |
| DB Initialization | 19.89 | 12.42 | -37.6% |
| Total Bulk Add | 95222.55 | 74062.13 | **-22.2%** |
| Standard FindNearest | 81.31 | 55.42 | **-31.8%** |
| Total HNSW Build | 359095.34 | 277449.55 | **-22.7%** |
| HNSW FindNearest | 25.34 | 20.27 | -20.0% |
| DB Save | 1260.85 | 1317.80 | +4.5% |
| DB Close | 0.34 | 1108.70 | +325988% |
| DB Re-Load | 546.51 | 299.30 | **-45.2%** |
| HNSW FindNearest After Re-Load | 19.23 | 8.19 | **-57.4%** |
| **Total benchmark** | **456.44 s** | **354.87 s** | **-22.3%** |

### DB Close is slower on purpose

`DB Close` went from 0.34 ms to 1108.70 ms, and that is the intended
consequence of the data-loss fix. Previously `close()` set its closing flag
before awaiting the final `save()`, so `save()` bailed out and the database was
discarded without being written. `close()` now flushes first and releases
resources after, which is where the 1.1 s goes. A close that takes no time was
a close that threw your data away.

### The bulk-add alternation is unchanged

The 3.3x alternation between odd and even batches is still present:

| Version | Odd batches (avg) | Even batches (avg) | Ratio |
| --- | ---: | ---: | ---: |
| v0.1.10 | 4411 ms | 14540 ms | 3.30x |
| v0.2.0 | 3487 ms | 11259 ms | 3.23x |

Absolute times improved by about 21% on both halves, so the ratio is
essentially unchanged. The cause is not a v0.1.10 regression and was not
addressed in v0.2.0: see the analysis in the v0.1.10 results file —
`ClusteredVectorDB._assignVectorToCluster` compares against a fixed
`newClusterDistanceThreshold` of 0.5 against a raw Euclidean distance, and this
benchmark's large-magnitude synthetic vectors exceed it for every vector. The
result is one cluster per vector, which makes `clusterSize` inert and the bulk
load quadratic.

## Summary

Total benchmark execution time: 354.87 seconds
