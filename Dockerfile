FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake ./cmake
COPY include ./include
COPY src ./src
COPY apps ./apps

RUN cmake -S . -B /tmp/build -DCMAKE_BUILD_TYPE=Release \
      -DNBASE_BUILD_TESTS=OFF -DNBASE_BUILD_BENCH=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/build -j && \
    cmake --install /tmp/build && \
    ldconfig

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r nbase && useradd -r -g nbase -m nbase && \
    mkdir -p /data && chown nbase:nbase /data

COPY --from=builder /usr/local/bin/nbased /usr/local/bin/nbased
COPY --from=builder /usr/local/bin/nbase-cli /usr/local/bin/nbase-cli
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /usr/local/include/nbase.h /usr/local/include/nbase.h
COPY --from=builder /usr/local/include/nbase.hpp /usr/local/include/nbase.hpp
RUN ldconfig

USER nbase
WORKDIR /data

ENV NBASE_HOST=0.0.0.0 \
    NBASE_PORT=1307 \
    NBASE_DATA_DIR=/data

EXPOSE 1307
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${NBASE_PORT:-1307}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/nbased"]
