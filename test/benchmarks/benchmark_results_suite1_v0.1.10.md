# PartitionedVectorDB Benchmark Results - Suite 1 - v0.1.10

*Run at: 2026-06-05T16:11:19.830Z*

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
| DB Initialization | 19.89 | 19.89 |
| Bulk Add Batch 1 (5000 vectors) | 4027.64 | 4027.64 |
| Bulk Add Batch 2 (5000 vectors) | 13692.22 | 13692.22 |
| Bulk Add Batch 3 (5000 vectors) | 4714.89 | 4714.89 |
| Bulk Add Batch 4 (5000 vectors) | 13205.01 | 13205.01 |
| Bulk Add Batch 5 (5000 vectors) | 4388.27 | 4388.27 |
| Bulk Add Batch 6 (5000 vectors) | 15321.62 | 15321.62 |
| Bulk Add Batch 7 (5000 vectors) | 4644.37 | 4644.37 |
| Bulk Add Batch 8 (5000 vectors) | 15544.03 | 15544.03 |
| Bulk Add Batch 9 (5000 vectors) | 4278.13 | 4278.13 |
| Bulk Add Batch 10 (5000 vectors) | 14936.42 | 14936.42 |
| Total Bulk Add | 95222.55 | 9522.25 |
| Standard FindNearest | 81.31 | 81.31 |
| Total HNSW Build | 359095.34 | 119698.45 |
| HNSW FindNearest | 25.34 | 25.34 |
| DB Save | 1260.85 | 1260.85 |
| DB Close | 0.34 | 0.34 |
| DB Re-Load | 546.51 | 546.51 |
| HNSW FindNearest After Re-Load | 19.23 | 19.23 |

## Search Performance Summary

### Standard vs HNSW Search Comparison

| Search Method | Time (ms) | Speedup Factor |
|---------------|----------:|---------------:|
| Standard Search | 81.31 | 1.00x |
| HNSW Search | 25.34 | 3.21x |
| HNSW Search (After Reload) | 19.23 | 4.23x |

**Note**: HNSW search is faster by a factor of 2.21x.

## Database Stats

- Total partitions: 5
- Loaded partitions: 3
- Total vectors: 50000
- HNSW indices: 0

## Summary

Total benchmark execution time: 456.44 seconds
