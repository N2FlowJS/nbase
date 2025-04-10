# PartitionedVectorDB Benchmark Results - Suite 2 - v0.1.3

*Run at: 2025-04-10T16:16:37.717Z*

## Configuration
- Vector Size: 128
- Partition Capacity: 10000
- Max Active Partitions: 3
- Vector Count: 5000
- Iterations per test: 100

## Results

| Operation | Average Time (ms) | Total Time (ms) |
|-----------|------------------:|-----------------:|
| addVector | 1.12 | 111.57 |
| bulkAdd | 17.39 | 1738.55 |
| findNearest | 1.37 | 137.34 |
| findNearestHNSW | 0.97 | 97.05 |

## Search Performance Summary

### Standard vs HNSW Search Comparison

| Search Method | Average Time (ms) | Relative Performance |
|---------------|------------------:|---------------------:|
| Standard Search | 1.37 | 1.42x slower |
| HNSW Search | 0.97 | Fastest (1.00x) |

**Summary**: HNSW search is approximately 1.42x faster than Standard search in this benchmark.
