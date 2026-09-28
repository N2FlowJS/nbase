# PartitionedVectorDB Benchmark Results - Suite 2 - v0.1.10

_Run at: 2026-06-05T16:04:28.381Z_

## Configuration

- Vector Size: 128
- Partition Capacity: 10000
- Max Active Partitions: 3
- Vector Count: 5000
- Iterations per test: 100

## Results

| Operation       | Average Time (ms) | Total Time (ms) |
| --------------- | ----------------: | --------------: |
| addVector       |              1.04 |          103.90 |
| bulkAdd         |             29.90 |         2989.59 |
| findNearest     |              1.46 |          146.16 |
| findNearestHNSW |              0.99 |           98.72 |

## Search Performance Summary

### Standard vs HNSW Search Comparison

| Search Method   | Average Time (ms) | Relative Performance |
| --------------- | ----------------: | -------------------: |
| Standard Search |              1.46 |         1.48x slower |
| HNSW Search     |              0.99 |      Fastest (1.00x) |

**Summary**: HNSW search is approximately 1.48x faster than Standard search in this benchmark.
