# PartitionedVectorDB Benchmark Results - Suite 1 - v0.1.3

*Run at: 2025-04-10T04:42:14.223Z*

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
| DB Initialization | 28.80 | 28.80 |
| Bulk Add Batch 1 (5000 vectors) | 3429.89 | 3429.89 |
| Bulk Add Batch 2 (5000 vectors) | 10409.00 | 10409.00 |
| Bulk Add Batch 3 (5000 vectors) | 3293.84 | 3293.84 |
| Bulk Add Batch 4 (5000 vectors) | 9112.80 | 9112.80 |
| Bulk Add Batch 5 (5000 vectors) | 3144.05 | 3144.05 |
| Bulk Add Batch 6 (5000 vectors) | 9686.83 | 9686.83 |
| Bulk Add Batch 7 (5000 vectors) | 3336.26 | 3336.26 |
| Bulk Add Batch 8 (5000 vectors) | 10335.62 | 10335.62 |
| Bulk Add Batch 9 (5000 vectors) | 3001.84 | 3001.84 |
| Bulk Add Batch 10 (5000 vectors) | 9221.84 | 9221.84 |
| Total Bulk Add | 65288.20 | 6528.82 |
| Standard FindNearest | 27.55 | 27.55 |
| Total HNSW Build | 266230.14 | 88743.38 |
| HNSW FindNearest | 21.03 | 21.03 |
| DB Save | 1112.42 | 1112.42 |
| DB Close | 0.36 | 0.36 |
| DB Re-Load | 568.41 | 568.41 |
| HNSW FindNearest After Re-Load | 6.52 | 6.52 |

## Search Performance Summary

### Standard vs HNSW Search Comparison

| Search Method | Time (ms) | Speedup Factor |
|---------------|----------:|---------------:|
| Standard Search | 27.55 | 1.00x |
| HNSW Search | 21.03 | 1.31x |
| HNSW Search (After Reload) | 6.52 | 4.23x |

**Note**: HNSW search is faster by a factor of 0.31x.

## Database Stats

- Total partitions: 5
- Loaded partitions: 3
- Total vectors: 50000
- HNSW indices: 0

## Summary

Total benchmark execution time: 333.42 seconds
