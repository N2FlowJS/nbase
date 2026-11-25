# NBase (`libnbase`) — Pure C11 / C++17 Vector, Graph & Full-Text Engine (`v1.13.0`)

NBase is a standalone, zero-dependency, embedded vector, metadata, graph, and full-text database written strictly in ISO C11 (with a header-only C++17 RAII wrapper [`include/nbase.hpp`](../include/nbase.hpp)), paired with a POSIX HTTP/1.1 Keep-Alive REST daemon (`nbased`) and streaming NDJSON CLI (`nbase-cli`).

## Key Capabilities

- **SIMD Distance Kernels**: Runtime-dispatched AVX2+FMA (`x86_64`), NEON (`ARM64`), and scalar fallback across 7 metrics (`euclidean`, `squared_euclidean`, `cosine`, `inner_product`, `manhattan`, `chebyshev`, `hamming`).
- **Indexing & Storage**: Contiguous flat vector storage with zero-copy `mmap`, lock-striped parallel HNSW (`Algorithm 4`), Durable LRU Multi-Partition DB (`nbase_pdb_t`), Dynamic Online Clustered Vector Index (`.nbcl`), IVF-PQ (`.nbiv`), Product Quantization (`.nbq`), SIMD SQ8 (`.nbsq`), 1-bit Binary Quantization (`.nbbq`), and Multi-Probe LSH (`.nblh`).
- **Search, Filtering, Caching & Graph**: Typed binary metadata (`int64`, `double`, `bool`, `string`), Okapi BM25 full-text + Hybrid Reciprocal Rank Fusion, MMR & metadata-weighted reranking, thread-safe LRU query result cache, k-NN relationship graphs & Union-Find community extraction.
- **Durability & Server**: CRC32-framed binary formats + Write-Ahead Log (`wal.nbw`), `nbased` HTTP/1.1 Keep-Alive server with Prometheus `/metrics` (P50/P95/P99 latency, `nbase_memory_bytes`, `nbase_cache_*`), CORS, Bearer auth, and rate limiting.

## Quick Build & Test

```bash
cmake -S . -B /tmp/nbase-release -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/nbase-release -j
ctest --test-dir /tmp/nbase-release --output-on-failure
```

## Documentation Guide

- [Installation Guide](installation.md) — Native CMake build, `pkg-config`, Docker & Docker Compose
- [Getting Started](getting-started.md) — C11, C++17, `nbase-cli`, and REST API quickstarts
- [Database Lifecycle & Durability](database.md) — Open modes (`mmap`, `wal`), compaction, and LRU cache
- [Vector Index Types](vector-databases.md) — Single DB, Partitioned DB, Clustered DB, IVF-PQ, LSH
- [Search, Metadata & Reranking](search-techniques.md) — HNSW/exact search, metadata filters, BM25, Hybrid RRF, MMR/weighted rerank, graph analytics
- [Quantization & Binary Formats](advanced-features.md) — PQ, IVF-PQ, SQ8, BQ and `.nb*` file layouts with CRC32
- [HTTP/1.1 Server & CLI](server.md) — `nbased` daemon, all HTTP endpoints, `nbase-cli` commands
- [API Reference](api-reference.md) — Complete C11 (`nbase.h`) & C++17 (`nbase.hpp`) symbol reference
- [Benchmarks](benchmarks.md) — Release benchmark suite and measured `v1.0.0` numbers
- [Migration from Node.js](migration.md) — Upgrading from `@n2flowjs/nbase` `v0.3.x` to `nbase v1.0.0`
- [Contributing](contributing.md) — Sanitizer testing (`ASan+UBSan`, `TSan`) and C11 standards
