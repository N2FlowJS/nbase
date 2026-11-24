# Getting Started with NBase (`v1.9.0`)

Build and install `libnbase` once (see [Installation Guide](installation.md)), or run the compiled binaries directly from your build/install directory.

## 1. Embedded C11 Quickstart

```c
#include "nbase.h"
#include <stdio.h>

int main(void) {
  nbase_config_t cfg = nbase_config_default(4U);
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.cache_capacity = 128U;

  nbase_db_t *db = NULL;
  if (nbase_create(&cfg, &db) != NBASE_OK) {
    return 1;
  }

  float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  float v2[4] = {0.8f, 0.6f, 0.0f, 0.0f};
  nbase_add(db, 1ULL, v1, 4U);
  nbase_add(db, 2ULL, v2, 4U);

  nbase_meta_field_t m1[2] = {
      {"tag", {.type = NBASE_META_STRING, .as = {.str = "sys"}}},
      {"score", {.type = NBASE_META_INT64, .as = {.i64 = 95}}},
  };
  nbase_set_meta(db, 1ULL, m1, 2U);
  nbase_set_text(db, 1ULL, "pure c11 avx2 vector engine");

  nbase_result_t hits[2];
  uint32_t count = 0U;
  if (nbase_search(db, v1, 4U, 2U, NULL, hits, &count) == NBASE_OK) {
    for (uint32_t i = 0U; i < count; i++) {
      printf("id=%llu dist=%.4f\n", (unsigned long long)hits[i].id, hits[i].distance);
    }
  }

  nbase_save(db, "/tmp/nbase_c_example");
  nbase_close(db);
  return 0;
}
```

Compile with:
```bash
cc -std=c11 example.c $(pkg-config --cflags --libs libnbase) -o example
```

## 2. C++17 RAII Quickstart

```cpp
#include "nbase.hpp"
#include <iostream>

int main() {
  auto cfg = nbase_config_default(4U);
  cfg.cache_capacity = 64U;
  auto db = nbase::Database::create(cfg);

  db.add(1U, {1.f, 0.f, 0.f, 0.f});
  db.add(2U, {0.8f, 0.6f, 0.f, 0.f});
  db.set_text(1U, "high performance vector engine");

  auto hits = db.search({1.f, 0.f, 0.f, 0.f}, 2U);
  std::cout << "top id=" << hits[0].id << " dist=" << hits[0].distance << "\n";
  return 0;
}
```

Compile with:
```bash
c++ -std=c++17 example.cpp $(pkg-config --cflags --libs libnbase) -o example_cpp
```

## 3. Command-Line Tool (`nbase-cli`)

```bash
nbase-cli init /tmp/cli_db --dim 3 --metric euclidean
nbase-cli add /tmp/cli_db 1 1,0,0
nbase-cli add /tmp/cli_db 2 0.8,0.6,0
nbase-cli set-text /tmp/cli_db 1 "simd vector storage"
nbase-cli search /tmp/cli_db 1,0,0 --k 2
nbase-cli search-hybrid /tmp/cli_db 1,0,0 "vector" --k 2
nbase-cli delete-batch /tmp/cli_db 2
nbase-cli info /tmp/cli_db
```

## 4. HTTP/1.1 REST Server (`nbased`)

Start the daemon with WAL persistence and query cache:

```bash
nbased --host 127.0.0.1 --port 1307 --data-dir /tmp/srv_db --dim 3 --workers 4 --cache-size 256
```

Send requests with `curl`:

```bash
curl -s http://127.0.0.1:1307/health

curl -s -X POST http://127.0.0.1:1307/api/vectors \
  -H 'Content-Type: application/json' \
  -d '{"vectors":[{"id":1,"vector":[1,0,0],"metadata":{"boost":0.5},"text":"avx2 hnsw engine"},{"id":2,"vector":[0.8,0.6,0],"metadata":{"boost":4.0}}]}'

curl -s -X POST http://127.0.0.1:1307/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":[1,0,0],"k":2,"rerank":true,"rerankingMethod":"weighted","rerankWeights":{"boost":1.0}}'

curl -s http://127.0.0.1:1307/stats
curl -s http://127.0.0.1:1307/metrics
```
