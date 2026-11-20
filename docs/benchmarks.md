# Benchmarks (`nbase v1.0.0`)

Build and run the standalone C11 Release benchmark binary locally:

```bash
cmake -S . -B /tmp/nbase-release -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/nbase-release --target bench_nbase -j
/tmp/nbase-release/bench_nbase
```

## Measured Reference Results (`libnbase 1.0.0`, `x86_64`, SIMD backend: `avx2+fma`)

### Distance Kernels

| Metric | Dimension | Scalar | SIMD (`avx2+fma`) | Speedup |
|---|---:|---:|---:|---:|
| Euclidean | 128 | 84.11 ns/op | 21.97 ns/op | **3.83x** |
| Cosine | 128 | 110.65 ns/op | 32.29 ns/op | **3.43x** |
| Euclidean | 1536 | 1213.50 ns/op | 277.96 ns/op | **4.37x** |
| Cosine | 1536 | 1240.44 ns/op | 292.46 ns/op | **4.24x** |

### HNSW Build, Search & Multi-Threaded Batch Search (`M=16, efConstruction=200, efSearch=100`)

| Benchmark | Dataset | Latency / Total | Throughput | Quality / Notes |
|---|---|---|---|---|
| HNSW single-thread search | N=2000, dim=128 | build 316.15 ms, query 100.55 us | **9,945 QPS** | recall@10 = **0.995** |
| HNSW single-thread search | N=8000, dim=64 | build 1935.72 ms, query 220.08 us | **4,544 QPS** | recall@10 = **0.975** |
| Batch search (1 thread) | N=8000, dim=64, 200q | 31.59 ms | 6,331 QPS | Work-stealing `pthreads` |
| Batch search (4 threads) | N=8000, dim=64, 200q | 13.08 ms | 15,292 QPS | 2.4x multi-core |
| Batch search (8 threads) | N=8000, dim=64, 200q | 10.30 ms | **19,422 QPS** | 3.1x multi-core |

### Quantization, LSH, BM25, Clustered DB & Query Cache (`dim=64`)

| Subsystem | Configuration | Build / Encode | Search / Scan |
|---|---|---|---|
| **Product Quantization (PQ)** | N=8000, M=8, ks=256 (32x) | train 1170.26 ms, encode 77.48 ms | 64.32 us/query (ADC) |
| **IVF-PQ** | N=8000, nlist=32, M=8 | train 421.64 ms, add 45.36 ms | **28.91 us/query** (`nprobe=4`) |
| **Scalar Quantization (SQ8)** | N=8000 (4x) | train 1.43 ms, encode 2.94 ms | 134.50 us/query |
| **Binary Quantization (BQ)** | N=8000 (32x) | train 0.54 ms, encode 2.57 ms | **41.54 us/query** (Hamming) |
| **Multi-Probe LSH** | N=8000, 8x10b | build 33.72 ms | 231.06 us/query (`multiprobe=2`) |
| **BM25 Full-Text** | 2,000 docs | index 2.28 ms | 235.61 us/query |
| **DB Hybrid RRF** | N=2000 | build 210.78 ms | 231.08 us/query |
| **Dynamic Clustered DB** | N=4000, k=32 | add 5.01 ms, recluster 49.84 ms | **35.77 us/query** (`nprobe=4`) |
| **Query LRU Cache** | N=2000, cap=256 | miss 72.51 us/query | **1.19 us/query** on hit (~61x) |
| **Weighted Reranking** | 20 candidates -> top 10 | — | **2.40 us/call** |
