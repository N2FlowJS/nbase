#include "nbase.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop_requested = 0;

static void handle_signal(int sig) {
  (void)sig;
  g_stop_requested = 1;
}

static const char *getenv_nonempty(const char *primary, const char *fallback) {
  const char *v = primary ? getenv(primary) : NULL;
  if (v && v[0] != '\0') {
    return v;
  }
  const char *fb = fallback ? getenv(fallback) : NULL;
  if (fb && fb[0] != '\0') {
    return fb;
  }
  return NULL;
}

static void print_usage(const char *prog) {
  fprintf(
      stderr,
      "Usage: %s [options]\n"
      "  --host <ip>          Bind host (env NBASE_HOST / HOST, default 127.0.0.1)\n"
      "  --port <port>        Listen port (env NBASE_PORT / PORT, default 3000)\n"
      "  --data-dir <dir>     Persistent WAL directory (env NBASE_DATA_DIR / NBASE_DB_PATH)\n"
      "  --dim <dim>          Vector dimension for new DB (env NBASE_DIM / NBASE_VECTOR_SIZE, default 128)\n"
      "  --metric <name>      Distance metric (env NBASE_METRIC, default euclidean)\n"
      "  --workers <n>        Worker threads (env NBASE_WORKERS, default 4)\n"
      "  --cache-size <n>     LRU query cache entries (env NBASE_CACHE_SIZE, default 0=off)\n"
      "  --api-key <key>      Optional Bearer API key (env NBASE_API_KEY / NBASE_API_KEYS)\n"
      "  --rate-limit <rps>   Max /api/* requests per sec (env NBASE_RATE_LIMIT, 0=off)\n"
      "  --cors-origin <orig> CORS Access-Control-Allow-Origin (env NBASE_CORS_ORIGIN / NBASE_CORS_ORIGINS)\n"
      "  --version            Print version and exit\n"
      "  --help               Show this help\n",
      prog);
}

static bool parse_metric(const char *s, nbase_metric_t *out) {
  if (!s) {
    return false;
  }
  if (strcmp(s, "euclidean") == 0) {
    *out = NBASE_METRIC_EUCLIDEAN;
    return true;
  }
  if (strcmp(s, "squared_euclidean") == 0) {
    *out = NBASE_METRIC_SQUARED_EUCLIDEAN;
    return true;
  }
  if (strcmp(s, "cosine") == 0) {
    *out = NBASE_METRIC_COSINE;
    return true;
  }
  if (strcmp(s, "inner_product") == 0 || strcmp(s, "dot") == 0) {
    *out = NBASE_METRIC_INNER_PRODUCT;
    return true;
  }
  if (strcmp(s, "manhattan") == 0) {
    *out = NBASE_METRIC_MANHATTAN;
    return true;
  }
  if (strcmp(s, "chebyshev") == 0) {
    *out = NBASE_METRIC_CHEBYSHEV;
    return true;
  }
  if (strcmp(s, "hamming") == 0) {
    *out = NBASE_METRIC_HAMMING;
    return true;
  }
  return false;
}

static bool parse_u32_arg(const char *s, uint32_t *out) {
  if (!s || s[0] == '\0') {
    return false;
  }
  errno = 0;
  char *endp = NULL;
  unsigned long v = strtoul(s, &endp, 10);
  if (errno != 0 || endp == s || *endp != '\0' || v > UINT32_MAX) {
    return false;
  }
  *out = (uint32_t)v;
  return true;
}

int main(int argc, char **argv) {
  const char *host = getenv_nonempty("NBASE_HOST", "HOST");
  const char *port_env = getenv_nonempty("NBASE_PORT", "PORT");
  const char *data_dir = getenv_nonempty("NBASE_DATA_DIR", "NBASE_DB_PATH");
  const char *dim_env = getenv_nonempty("NBASE_DIM", "NBASE_VECTOR_SIZE");
  const char *metric_env = getenv_nonempty("NBASE_METRIC", NULL);
  const char *workers_env = getenv_nonempty("NBASE_WORKERS", NULL);
  const char *cache_env = getenv_nonempty("NBASE_CACHE_SIZE", NULL);
  const char *api_key = getenv_nonempty("NBASE_API_KEY", "NBASE_API_KEYS");
  const char *rl_env = getenv_nonempty("NBASE_RATE_LIMIT", NULL);
  const char *cors_origin = getenv_nonempty("NBASE_CORS_ORIGIN", "NBASE_CORS_ORIGINS");

  if (!host || host[0] == '\0') {
    host = "127.0.0.1";
  }
  uint16_t port = 3000U;
  if (port_env && port_env[0] != '\0') {
    uint32_t tmp = 0U;
    if (!parse_u32_arg(port_env, &tmp) || tmp > 65535U) {
      fprintf(stderr, "Invalid NBASE_PORT: %s\n", port_env);
      return 1;
    }
    port = (uint16_t)tmp;
  }

  uint32_t dim = 128U;
  if (dim_env && dim_env[0] != '\0') {
    if (!parse_u32_arg(dim_env, &dim) || dim == 0U || dim > NBASE_MAX_DIM) {
      fprintf(stderr, "Invalid NBASE_DIM: %s\n", dim_env);
      return 1;
    }
  }

  nbase_metric_t metric = NBASE_METRIC_EUCLIDEAN;
  if (metric_env && metric_env[0] != '\0') {
    if (!parse_metric(metric_env, &metric)) {
      fprintf(stderr, "Invalid NBASE_METRIC: %s\n", metric_env);
      return 1;
    }
  }

  uint32_t workers = 4U;
  if (workers_env && workers_env[0] != '\0') {
    if (!parse_u32_arg(workers_env, &workers) || workers == 0U || workers > 32U) {
      fprintf(stderr, "Invalid NBASE_WORKERS: %s\n", workers_env);
      return 1;
    }
  }

  uint32_t cache_size = 0U;
  if (cache_env && cache_env[0] != '\0') {
    if (!parse_u32_arg(cache_env, &cache_size)) {
      fprintf(stderr, "Invalid NBASE_CACHE_SIZE: %s\n", cache_env);
      return 1;
    }
  }

  uint32_t rate_limit = 0U;
  if (rl_env && rl_env[0] != '\0') {
    if (!parse_u32_arg(rl_env, &rate_limit)) {
      fprintf(stderr, "Invalid NBASE_RATE_LIMIT: %s\n", rl_env);
      return 1;
    }
  }

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    }
    if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-v") == 0) {
      printf("nbased %s (simd: %s)\n", nbase_version(), nbase_simd_backend());
      return 0;
    }
    if (i + 1 >= argc) {
      fprintf(stderr, "Missing value for %s\n", argv[i]);
      return 1;
    }
    if (strcmp(argv[i], "--host") == 0) {
      host = argv[++i];
    } else if (strcmp(argv[i], "--port") == 0) {
      uint32_t tmp = 0U;
      if (!parse_u32_arg(argv[++i], &tmp) || tmp > 65535U) {
        fprintf(stderr, "Invalid --port\n");
        return 1;
      }
      port = (uint16_t)tmp;
    } else if (strcmp(argv[i], "--data-dir") == 0) {
      data_dir = argv[++i];
    } else if (strcmp(argv[i], "--dim") == 0) {
      if (!parse_u32_arg(argv[++i], &dim) || dim == 0U || dim > NBASE_MAX_DIM) {
        fprintf(stderr, "Invalid --dim\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--metric") == 0) {
      if (!parse_metric(argv[++i], &metric)) {
        fprintf(stderr, "Invalid --metric\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--workers") == 0) {
      if (!parse_u32_arg(argv[++i], &workers) || workers == 0U || workers > 32U) {
        fprintf(stderr, "Invalid --workers\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--cache-size") == 0) {
      if (!parse_u32_arg(argv[++i], &cache_size)) {
        fprintf(stderr, "Invalid --cache-size\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--api-key") == 0) {
      api_key = argv[++i];
    } else if (strcmp(argv[i], "--rate-limit") == 0) {
      if (!parse_u32_arg(argv[++i], &rate_limit)) {
        fprintf(stderr, "Invalid --rate-limit\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--cors-origin") == 0) {
      cors_origin = argv[++i];
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      print_usage(argv[0]);
      return 1;
    }
  }

  (void)signal(SIGPIPE, SIG_IGN);
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = handle_signal;
  sigemptyset(&sa.sa_mask);
  (void)sigaction(SIGINT, &sa, NULL);
  (void)sigaction(SIGTERM, &sa, NULL);

  nbase_db_t *db = NULL;
  bool has_wal = false;
  if (data_dir != NULL && data_dir[0] != '\0') {
    size_t dlen = strlen(data_dir);
    char *vec_path = malloc(dlen + 16U);
    if (!vec_path) {
      fprintf(stderr, "Out of memory\n");
      return 1;
    }
    snprintf(vec_path, dlen + 16U, "%s/vectors.nbv", data_dir);
    struct stat st_vec;
    if (stat(vec_path, &st_vec) == 0) {
      nbase_status_t st = nbase_open_wal(data_dir, &db);
      if (st != NBASE_OK) {
        fprintf(stderr, "Failed to open DB at %s: %s\n", data_dir, nbase_strerror(st));
        free(vec_path);
        return 1;
      }
      has_wal = true;
    } else {
      nbase_config_t cfg = nbase_config_default(dim);
      cfg.metric = metric;
      nbase_status_t st = nbase_create(&cfg, &db);
      if (st == NBASE_OK) {
        st = nbase_wal_enable(db, data_dir);
      }
      if (st != NBASE_OK) {
        fprintf(stderr, "Failed to initialize WAL DB at %s: %s\n", data_dir, nbase_strerror(st));
        nbase_close(db);
        free(vec_path);
        return 1;
      }
      has_wal = true;
    }
    free(vec_path);
  } else {
    nbase_config_t cfg = nbase_config_default(dim);
    cfg.metric = metric;
    nbase_status_t st = nbase_create(&cfg, &db);
    if (st != NBASE_OK) {
      fprintf(stderr, "Failed to create DB: %s\n", nbase_strerror(st));
      return 1;
    }
  }

  nbase_stats_t db_st;
  if (cache_size > 0U && nbase_cache_configure(db, cache_size) != NBASE_OK) {
    fprintf(stderr, "Failed to configure query cache (%u entries)\n", cache_size);
    nbase_close(db);
    return 1;
  }
  if (nbase_stats(db, &db_st) != NBASE_OK) {
    nbase_close(db);
    return 1;
  }

  nbase_http_config_t hcfg = nbase_http_config_default();
  hcfg.bind_host = host;
  hcfg.port = port;
  hcfg.num_workers = workers;
  hcfg.api_key = (api_key && api_key[0] != '\0') ? api_key : NULL;
  hcfg.save_dir = (data_dir && data_dir[0] != '\0') ? data_dir : NULL;
  hcfg.max_requests_per_sec = rate_limit;
  hcfg.cors_origin = (cors_origin && cors_origin[0] != '\0') ? cors_origin : NULL;

  nbase_http_server_t *srv = NULL;
  nbase_status_t st = nbase_http_server_start(db, &hcfg, &srv);
  if (st != NBASE_OK) {
    fprintf(stderr, "Failed to start HTTP server: %s\n", nbase_strerror(st));
    nbase_close(db);
    return 1;
  }

  printf(
      "nbased %s listening on http://%s:%u (workers=%u, dim=%u, active=%llu, simd=%s)\n",
      nbase_version(),
      host,
      (unsigned int)nbase_http_server_port(srv),
      workers,
      db_st.dim,
      (unsigned long long)db_st.active_vectors,
      nbase_simd_backend());
  fflush(stdout);

  while (!g_stop_requested) {
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 100000000L;
    (void)nanosleep(&ts, NULL);
  }

  printf("nbased shutting down...\n");
  fflush(stdout);
  nbase_http_server_stop(srv);
  if (has_wal) {
    nbase_status_t ck = nbase_wal_checkpoint(db);
    if (ck != NBASE_OK) {
      fprintf(stderr, "WAL checkpoint warning: %s\n", nbase_strerror(ck));
    }
  }
  nbase_close(db);
  return 0;
}
