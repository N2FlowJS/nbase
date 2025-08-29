# PartitionedVectorDB Benchmark Results - Suite 1 - v0.1.10

*Run at: 2025-08-29T13:46:50.436Z*

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
| DB Initialization | 131.29 | 131.29 |
| Bulk Add Batch 1 (5000 vectors) | 6980.68 | 6980.68 |
| Bulk Add Batch 2 (5000 vectors) | 23761.83 | 23761.83 |
| Bulk Add Batch 3 (5000 vectors) | 6793.39 | 6793.39 |
| Bulk Add Batch 4 (5000 vectors) | 20354.65 | 20354.65 |
| Bulk Add Batch 5 (5000 vectors) | 11071.97 | 11071.97 |
| Bulk Add Batch 6 (5000 vectors) | 24820.04 | 24820.04 |
| Bulk Add Batch 7 (5000 vectors) | 7616.75 | 7616.75 |
| Bulk Add Batch 8 (5000 vectors) | 25521.73 | 25521.73 |
| Bulk Add Batch 9 (5000 vectors) | 7129.80 | 7129.80 |
| Bulk Add Batch 10 (5000 vectors) | 23002.02 | 23002.02 |
| Total Bulk Add | 157825.62 | 15782.56 |
| Standard FindNearest | 318.01 | 318.01 |
| Total HNSW Build | 570504.33 | 190168.11 |
| HNSW FindNearest | 120.18 | 120.18 |
| DB Save | 2651.97 | 2651.97 |
| DB Close | 0.55 | 0.55 |
| DB Re-Load | 1110.56 | 1110.56 |
| HNSW FindNearest After Re-Load | 84.31 | 84.31 |

## Search Performance Summary

### Standard vs HNSW Search Comparison

| Search Method | Time (ms) | Speedup Factor |
|---------------|----------:|---------------:|
| Standard Search | 318.01 | 1.00x |
| HNSW Search | 120.18 | 2.65x |
| HNSW Search (After Reload) | 84.31 | 3.77x |

**Note**: HNSW search is faster by a factor of 1.65x.

## Database Stats

- Total partitions: 5
- Loaded partitions: 3
- Total vectors: 50000
- HNSW indices: 0

## Summary

Total benchmark execution time: 733.65 seconds
