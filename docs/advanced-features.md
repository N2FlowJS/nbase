# Quantization & Crash-Safe Binary Formats

## Vector Quantization & Sketching

| Method | Header | Compression | Core Operations |
|---|---|---|---|
| **Scalar Quantization (SQ8)** | `src/sq8.c` | **4x** (`uint8_t` per dim) | `nbase_sq8_train`, `encode`, `decode`, SIMD `nbase_sq8_l2_sq_asymmetric` / `_symmetric` |
| **Product Quantization (PQ)** | `src/pq.c` | **32x** (e.g. `dim=64, M=8`) | K-Means++ codebooks, `nbase_pq_train`, `encode`, `build_lut`, `nbase_pq_adc_distance` |
| **IVF-PQ Index** | `src/ivfpq.c` | **~32x** | Coarse K-Means++ inverted lists + residual PQ + `nprobe` ADC scan |
| **1-bit Binary Quantization (BQ)** | `src/bq.c` | **32x** (`uint64_t` bit-pack) | Mean-threshold binarization + popcount `nbase_bq_hamming` |
| **Multi-Probe LSH** | `src/lsh.c` | Hash tables | Cosine SimHash & Euclidean p-stable buckets with Hamming neighborhood probing |

## Atomic On-Disk Persistence & Checksums

All persisted files use fixed little-endian binary headers (`128` bytes, or `64` bytes for WAL + `16`-byte frame headers) verified at compile time via `_Static_assert` in [`src/nbase_internal.h`](../src/nbase_internal.h), dual IEEE 802.3 CRC32 checksums (`header_crc32` + `payload_crc32`), and atomic write-then-rename (`nbase_write_file_atomic`):

| Artifact | Magic Header | Contents |
|---|---|---|
| `vectors.nbv` | `NBASEVC1` | Contiguous vectors, IDs, slot flags (`mmap`-compatible) |
| `hnsw.nbi` | `NBASEHN1` | Multi-layer HNSW graph, entry point, node states, PRNG state |
| `meta.nbm` | `NBASEMT1` | Typed binary metadata (`int64`, `double`, `bool`, `string`) |
| `wal.nbw` | `NBASEWL1` | Crash-safe append-only operation frames with torn-tail recovery |
| `manifest.nbp` | `NBASEPM1` | Multi-partition registry & global `id -> partition_id` index |
| `.nbq` | `NBASEPQ1` | Product Quantizer sub-codebooks |
| `.nbiv` | `NBASEIV1` | IVF coarse centroids, PQ codebooks, inverted lists |
| `.nbsq` | `NBASESQ1` | Per-dimension min/scale SQ8 parameters |
| `.nbbq` | `NBASEBQ1` | Per-dimension BQ bit-packing thresholds |
| `.nbbm` | `NBASEBM1` | BM25 document map, IDF statistics, posting lists |
| `.nblh` | `NBASELH1` | LSH hyperplanes & bucket tables |
| `.nbcl` | `NBASECL1` | Online Clustered centroids, assignments & vectors |
