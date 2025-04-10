# PartitionedVectorDB Benchmark Results - Suite 2 - v0.1.3

*Run at: 2025-04-10T04:36:35.520Z*

## Configuration
- Vector Size: 128
- Partition Capacity: 10000
- Max Active Partitions: 3
- Vector Count: 5000
- Iterations per test: 100

## Results

| Operation | Average Time (ms) | Total Time (ms) |
|-----------|------------------:|-----------------:|
| addVector | 0.89 | 88.54 |
| bulkAdd | 17.02 | 1701.73 |
| findNearest | 0.43 | 43.41 |
| findNearestHNSW | 0.79 | 78.90 |

## Search Performance Summary

### Standard vs HNSW Search Comparison

| Search Method | Average Time (ms) | Relative Performance |
|---------------|------------------:|---------------------:|
| Standard Search | 0.43 | Fastest (1.00x) |
| HNSW Search | 0.79 | 1.82x slower |

**Summary**: Standard search is approximately 1.82x faster than HNSW search in this benchmark.
