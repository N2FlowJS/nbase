# Contributing to NBase

Thank you for contributing to **NBase** (`v1.0.0`), written in pure C11 with a header-only C++17 RAII wrapper.

## Repository Layout

- `include/` — Public C11 API ([`include/nbase.h`](include/nbase.h)) and header-only C++17 wrapper ([`include/nbase.hpp`](include/nbase.hpp))
- `src/` — Core engine modules (`distance.c`, `store.c`, `hnsw.c`, `meta.c`, `wal.c`, `bm25.c`, `rerank.c`, `kmeans.c`, `pq.c`, `ivfpq.c`, `sq8.c`, `bq.c`, `lsh.c`, `clustered.c`, `partitioned.c`, `json.c`, `http_server.c`, `nbase.c`)
- `apps/` — CLI (`nbase_cli.c`) and HTTP/1.1 REST daemon (`nbased.c`)
- `tests/` — C11 unit, integration, HTTP, sanitizer, and TSan test suite (`test_main.c`)
- `bench/` — Performance benchmark suite (`bench_nbase.c`)
- `cmake/` — `libnbase.pc.in` pkg-config template

## Development Workflow

1. **Fork and branch from `main`**:
   ```bash
   git clone https://github.com/your-username/nbase.git
   cd nbase
   git checkout -b my-feature
   ```
2. **Build & test across all three build profiles**:
   ```bash
   # Release build + benchmark
   cmake -S . -B /tmp/nbase-release -DCMAKE_BUILD_TYPE=Release
   cmake --build /tmp/nbase-release -j
   ctest --test-dir /tmp/nbase-release --output-on-failure
   /tmp/nbase-release/bench_nbase

   # Debug build with AddressSanitizer + UndefinedBehaviorSanitizer
   cmake -S . -B /tmp/nbase-asan -DCMAKE_BUILD_TYPE=Debug -DNBASE_ENABLE_SANITIZERS=ON
   cmake --build /tmp/nbase-asan -j
   ctest --test-dir /tmp/nbase-asan --output-on-failure

   # Debug build with ThreadSanitizer
   cmake -S . -B /tmp/nbase-tsan -DCMAKE_BUILD_TYPE=Debug -DNBASE_ENABLE_TSAN=ON
   cmake --build /tmp/nbase-tsan -j
   ctest --test-dir /tmp/nbase-tsan --output-on-failure
   ```

## C11 / C++17 Coding Standards

- **Zero compiler warnings**: All C files compile with `-std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror`. All additions to [`include/nbase.hpp`](include/nbase.hpp) must compile cleanly with `g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror`.
- **Binary format invariants**: Do not modify existing 128-byte (or 64-byte WAL) persisted file headers or `header_crc32` byte offsets verified via `_Static_assert` in [`src/nbase_internal.h`](src/nbase_internal.h).
- **Thread safety**: Lock ordering must always follow `db->rwlock` -> `db->cache_lock`, and `pdb->mu` -> per-partition `nbase_*` APIs. Every public change must pass under TSan with zero data races.
- **No runtime dependencies**: Do not introduce external libraries beyond libc, libm, and POSIX `pthreads`.
