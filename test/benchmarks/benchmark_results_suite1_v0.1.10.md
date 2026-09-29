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

**Note**: HNSW search is 3.21x faster than standard search (81.31 ms vs 25.34 ms), and 4.23x after a reload. An earlier revision of this file stated 2.21x, which contradicted the table above it; the table is correct.

### Bulk add timing note

The per-batch times alternate systematically: batches 1/3/5/7/9 average
4411 ms, batches 2/4/6/8/10 average 14540 ms — a 3.3x spread that repeats
across all five pairs. See "clustering threshold" in the notes below.

`Total Bulk Add` (95222.55 ms) is larger than the sum of the ten batch timings
(94752.60 ms) by 470 ms. That difference is the per-batch `getStats()` call and
logging between batches, which fall inside the total but outside the individual
batch timers. It is not a measurement error.

## Database Stats

- Total partitions: 5
- Loaded partitions: 3
- Total vectors: 50000
- HNSW indices: 0

`HNSW indices: 0` is reported by the stats snapshot taken at the end of the
bulk-add phase, before `buildIndexHNSW()` runs. The HNSW timings above are
measured after that build, so the two are not in conflict — but the field is
easy to misread and is worth stating here.

## Summary

Total benchmark execution time: 456.44 seconds

## Analysis: why the bulk-add batches alternate

The 3.3x alternating pattern in the bulk-add table is caused by
`ClusteredVectorDB._assignVectorToCluster` comparing a vector against every
existing centroid, with a fixed absolute cut-off:

    const needsNewCluster =
      clusterMembers.length >= targetClusterSize * newClusterThresholdFactor ||
      minDist > newClusterDistanceThreshold;     // default 0.5

`minDist` is a raw Euclidean distance, so the cut-off only means "0.5" for data
whose scale happens to sit there. This benchmark generates large synthetic
values, so every distance exceeds 0.5 and a new cluster is created for every
vector — `clusterSize: 100` becomes a no-op, and the search is over a centroid
list that grows with the data.

Measured on this machine, 2000-vector chunks against a single partition:

| cumulative vectors | clusters | clusters/vector | ms/vector |
| ---: | ---: | ---: | ---: |
| 2 000 | 2 000 | 1.00 | 0.141 |
| 4 000 | 4 000 | 1.00 | 0.314 |
| 6 000 | 6 000 | 1.00 | 0.545 |
| 8 000 | 8 000 | 1.00 | 0.898 |
| 10 000 | 10 000 | 1.00 | 1.257 |
| 12 000 | 12 000 | 1.00 | 1.802 |

Per-vector cost grows linearly, so the whole bulk load is quadratic.

The same code behaves correctly on normalised embeddings, where Euclidean
distance is small:

| data | clusters (6 000 vectors) | ms/vector by chunk |
| --- | ---: | --- |
| normalised, euclidean | 125 | 0.023 → 0.015 → 0.015 |
| normalised, cosine | 53 | 0.016 → 0.015 → 0.015 |

So this is not a general clustering failure — it is an unnormalised-input
hazard that the fixed 0.5 threshold silently turns into a no-op configuration.
Two things would remove it: comparing in a scale-free metric (cosine) or
deriving the threshold from the observed distance distribution instead of a
constant.

Reproduce with `probe.ts` in this directory's sibling notes; the numbers above
come from `ClusteredVectorDB` directly, not through `PartitionedVectorDB`.
