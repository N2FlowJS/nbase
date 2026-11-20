# Developer & Contributor Architecture Guide

See the top-level [CONTRIBUTING.md](../CONTRIBUTING.md) for quick commands and PR workflow. This guide details internal module conventions in `nbase v1.0.0`.

## Source Architecture

- [`include/nbase.h`](../include/nbase.h): Strictly C11-compatible public C ABI (`extern "C"` guarded for C++).
- [`include/nbase.hpp`](../include/nbase.hpp): Header-only C++17 RAII layer throwing `nbase::Error` on non-`NBASE_OK` status codes.
- [`src/nbase_internal.h`](../src/nbase_internal.h): Internal binary format structs (`_Static_assert` verified), heap helpers, PRNG (`splitmix64`), and internal cross-module prototypes.
- [`src/distance.c`](../src/distance.c): CPUID / compile-time detection selecting AVX2+FMA, NEON, or scalar distance kernels at runtime.
- [`src/store.c`](../src/store.c): Slot storage + open-addressing hash map + atomic persistence + read-only `mmap`.
- [`src/hnsw.c`](../src/hnsw.c): Lock-striped HNSW with generation-stamped visited sets per thread.
- [`src/nbase.c`](../src/nbase.c): Top-level `nbase_db_t` facade orchestrating RW locks, `cache_lock`, WAL replay, search batching, and compaction.
- [`src/http_server.c`](../src/http_server.c) & [`src/json.c`](../src/json.c): Standalone HTTP/1.1 Keep-Alive REST server with RFC 8259 strict JSON parsing and RFC 9112 lingering close.

## Sanitizers & Thread Safety Checks

Always verify every change under all three build configurations:
1. **AddressSanitizer + UndefinedBehaviorSanitizer** (`-DNBASE_ENABLE_SANITIZERS=ON`): catches buffer bounds, use-after-free, signed overflows, misalignment, and memory leaks.
2. **ThreadSanitizer** (`-DNBASE_ENABLE_TSAN=ON`): catches concurrent RWLock / mutex violations (`rwlock` -> `cache_lock`, `pdb->mu` -> `db->rwlock`) and HTTP worker pool races.
3. **Release (`-O3`)**: verifies timing-sensitive HTTP lingering close and benchmark optimizations.

## Adding Tests

Add test functions in `tests/test_<module>.c`, register them in the module's `test_<module>_run()` function, and ensure all 53+ checks pass cleanly via `ctest --output-on-failure`.
