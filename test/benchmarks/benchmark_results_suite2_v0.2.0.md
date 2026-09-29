# PartitionedVectorDB Benchmark Results - Suite 2 - v0.2.0

*Run at: 2026-09-29T15:26:17.697Z*

## Configuration
- Vector Size: 128
- Partition Capacity: 10000
- Max Active Partitions: 3
- Vector Count: 5000
- Iterations per test: 100

## Results

| Operation | Average Time (ms) | Total Time (ms) |
|-----------|------------------:|-----------------:|
| addVector | 0.31 | 30.96 |
| bulkAdd | 4.53 | 452.90 |
| findNearest | 0.49 | 49.19 |
| findNearestHNSW | 0.59 | 59.49 |

## Search Performance Summary

### Standard vs HNSW Search Comparison

| Search Method | Average Time (ms) | Relative Performance |
|---------------|------------------:|---------------------:|
| Standard Search | 0.49 | Fastest (1.00x) |
| HNSW Search | 0.59 | 1.21x slower |

## Comparison with v0.1.10

| Operation | v0.1.10 | v0.2.0 | Change |
| --- | ---: | ---: | ---: |
| addVector | 1.04 | 0.31 | -70.2% |
| bulkAdd | 29.90 | 4.53 | -84.8% |
| findNearest | 1.46 | 0.49 | -66.4% |
| findNearestHNSW | 0.99 | 0.59 | -40.4% |

### HNSW now loses to standard search at this scale

HNSW search was 1.48x faster than standard search in v0.1.10; here it is 1.21x
slower. That is a change in ranking, not a regression in HNSW: absolute HNSW
latency improved by 40% (0.99 ms to 0.59 ms), while standard search improved
by 66% (1.46 ms to 0.49 ms) and overtook it.

At 5 000 vectors with k=10 a linear scan is already cheap enough that the cost
of walking the HNSW graph, including the per-neighbour eligibility check added
in v0.2.0, is not repaid. Suite 1 at 50 000 vectors still shows HNSW ahead
(20.27 ms vs 55.42 ms, 2.73x). The crossover sits somewhere between the two
sizes; this benchmark does not locate it.

**Summary**: Standard search is approximately 1.21x faster than HNSW search in this benchmark.
