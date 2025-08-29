# PartitionedVectorDB Benchmark Results - Suite 2 - v0.1.9

_Run at: 2025-08-29T13:34:28.513Z_

## Configuration

- Vector Size: 128
- Partition Capacity: 10000
- Max Active Partitions: 3
- Vector Count: 5000
- Iterations per test: 100

## Results

| Operation       | Average Time (ms) | Total Time (ms) |
| --------------- | ----------------: | --------------: |
| addVector       |              1.54 |          153.91 |
| bulkAdd         |             48.97 |         4897.05 |
| findNearest     |              2.14 |          213.96 |
| findNearestHNSW |              1.55 |          154.61 |

## Search Performance Summary

### Standard vs HNSW Search Comparison

| Search Method   | Average Time (ms) | Relative Performance |
| --------------- | ----------------: | -------------------: |
| Standard Search |              2.14 |         1.38x slower |
| HNSW Search     |              1.55 |      Fastest (1.00x) |

**Summary**: HNSW search is approximately 1.38x faster than Standard search in this benchmark.
