# PartitionedVectorDB Benchmark Results - Suite 1 - v0.1.3

*Run at: 2025-04-10T16:22:05.503Z*

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
| DB Initialization | 20.09 | 20.09 |
| Bulk Add Batch 1 (5000 vectors) | 3087.77 | 3087.77 |
| Bulk Add Batch 2 (5000 vectors) | 11287.98 | 11287.98 |
| Bulk Add Batch 3 (5000 vectors) | 3256.65 | 3256.65 |
| Bulk Add Batch 4 (5000 vectors) | 10781.16 | 10781.16 |
| Bulk Add Batch 5 (5000 vectors) | 3282.28 | 3282.28 |
| Bulk Add Batch 6 (5000 vectors) | 9787.72 | 9787.72 |
| Bulk Add Batch 7 (5000 vectors) | 3106.99 | 3106.99 |
| Bulk Add Batch 8 (5000 vectors) | 9514.98 | 9514.98 |
| Bulk Add Batch 9 (5000 vectors) | 3173.97 | 3173.97 |
| Bulk Add Batch 10 (5000 vectors) | 9795.62 | 9795.62 |
| Total Bulk Add | 67408.81 | 6740.88 |
| Standard FindNearest | 50.49 | 50.49 |
| Total HNSW Build | 253428.89 | 84476.30 |
| HNSW FindNearest | 17.63 | 17.63 |
| DB Save | 1088.92 | 1088.92 |
| DB Close | 0.47 | 0.47 |
| DB Re-Load | 501.52 | 501.52 |
| HNSW FindNearest After Re-Load | 8.62 | 8.62 |

## Search Performance Summary

### Standard vs HNSW Search Comparison

| Search Method | Time (ms) | Speedup Factor |
|---------------|----------:|---------------:|
| Standard Search | 50.49 | 1.00x |
| HNSW Search | 17.63 | 2.86x |
| HNSW Search (After Reload) | 8.62 | 5.86x |

**Note**: HNSW search is faster by a factor of 1.86x.

## Database Stats

- Total partitions: 5
- Loaded partitions: 3
- Total vectors: 50000
- HNSW indices: 0

## Summary

Total benchmark execution time: 322.66 seconds
