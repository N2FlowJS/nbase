# NBase Installation Guide (Pure C11 / C++17 — v1.17.0)

## Requirements

- **Compiler**: GCC or Clang with C11 support (plus C++17 compiler if consuming `include/nbase.hpp`)
- **Build System**: CMake `>= 3.16`, `make` or `ninja`, POSIX `pthreads`
- **Optional**: `pkg-config` for downstream compilation, Docker / Docker Compose for containerized deployment

## Build & Install from Source

```bash
git clone https://github.com/n2flowjs/nbase.git
cd nbase

# Configure, build, test, and install to /usr/local
cmake -S . -B /tmp/nbase-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/nbase-build -j
ctest --test-dir /tmp/nbase-build --output-on-failure
sudo cmake --install /tmp/nbase-build
sudo ldconfig
```

Installed artifacts:
- Libraries: `libnbase.a`, `libnbase.so.1.17.0` (`libnbase.so.1`, `libnbase.so`)
- Headers: `nbase.h` (C11), `nbase.hpp` (C++17 header-only RAII wrapper)
- Binaries: `nbased` (HTTP/1.1 REST daemon), `nbase-cli` (offline/NDJSON CLI tool)
- Package config: `libnbase.pc`

Compile a downstream C or C++17 binary via `pkg-config`:

```bash
cc  -std=c11   app.c   $(pkg-config --cflags --libs libnbase) -o app
c++ -std=c++17 app.cpp $(pkg-config --cflags --libs libnbase) -o app_cpp
```

## Docker & Docker Compose

```bash
# Build native Debian Bookworm runtime image (~minimal C runtime)
docker build -t nbase:local .
docker run -d --name nbase -p 1307:1307 -v nbase_data:/data nbase:local

# Or start with Docker Compose
cp .env.example .env
docker compose up -d --build
curl -fsS http://127.0.0.1:1307/health
```

## Environment Variables (`nbased`)

| Variable | Fallback | Default | Description |
|---|---|---|---|
| `NBASE_HOST` | `HOST` | `127.0.0.1` (`0.0.0.0` in Docker) | Bind IP address |
| `NBASE_PORT` | `PORT` | `3000` (`1307` in Docker) | HTTP/1.1 listen port |
| `NBASE_DATA_DIR` | `NBASE_DB_PATH` | *(in-memory)* | WAL-backed directory |
| `NBASE_DIM` | `NBASE_VECTOR_SIZE` | `128` | Vector dimension when initializing a new DB |
| `NBASE_METRIC` | — | `euclidean` | Distance metric (`euclidean`, `squared_euclidean`, `cosine`, `inner_product`, `manhattan`, `chebyshev`, `hamming`) |
| `NBASE_WORKERS` | — | `4` | Worker thread count (`1..32`) |
| `NBASE_CACHE_SIZE` | — | `0` (`256` in Compose) | Max LRU query result cache entries |
| `NBASE_API_KEY` | `NBASE_API_KEYS` | *(disabled)* | Optional Bearer API key |
| `NBASE_RATE_LIMIT` | — | `0` | Max `/api/*` requests per second (`0=off`) |
| `NBASE_CORS_ORIGIN` | `NBASE_CORS_ORIGINS` | *(disabled)* | `Access-Control-Allow-Origin` value |
