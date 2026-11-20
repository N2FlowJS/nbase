#include "nbase.h"
#include "nbase_json.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NBASE_CLI_IMPORT_BATCH 1024U

static void print_usage(const char *prog) {
  fprintf(
      stderr,
      "Usage: %s <command> [args]\n"
      "Commands:\n"
      "  version\n"
      "  init <dir> --dim <D> [--metric <M>] [--no-hnsw]\n"
      "  info <dir>\n"
      "  add <dir> <id> <v0,v1,...>\n"
      "  get <dir> <id>\n"
      "  delete <dir> <id>\n"
      "  delete-batch <dir> <id1,id2,...>\n"
      "  set-text <dir> <id> <text>\n"
      "  search <dir> <q0,q1,...> [--k <K>] [--ef <EF>] [--exact]\n"
      "  search-text <dir> <query> [--k <K>]\n"
      "  search-hybrid <dir> <q0,q1,...|-> <query_text> [--k <K>] [--alpha <A>] [--rrf-k <R>]\n"
      "  rebuild <dir> [--threads <T>]\n"
      "  compact <dir>\n"
      "  checkpoint <dir>\n"
      "  import <dir> <file.ndjson|-> [--threads <T>]\n"
      "  export <dir> [file.ndjson|-]\n",
      prog);
}

static bool parse_u64(const char *s, uint64_t *out) {
  if (!s || s[0] == '\0') {
    return false;
  }
  for (const char *p = s; *p != '\0'; p++) {
    if (*p < '0' || *p > '9') {
      return false;
    }
  }
  errno = 0;
  char *endp = NULL;
  unsigned long long v = strtoull(s, &endp, 10);
  if (errno != 0 || endp == s || *endp != '\0') {
    return false;
  }
  *out = (uint64_t)v;
  return true;
}

static bool parse_u32(const char *s, uint32_t *out) {
  uint64_t v = 0ULL;
  if (!parse_u64(s, &v) || v > UINT32_MAX) {
    return false;
  }
  *out = (uint32_t)v;
  return true;
}

static bool parse_metric(const char *s, nbase_metric_t *out) {
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

static const char *metric_str(nbase_metric_t m) {
  switch (m) {
    case NBASE_METRIC_EUCLIDEAN:
      return "euclidean";
    case NBASE_METRIC_SQUARED_EUCLIDEAN:
      return "squared_euclidean";
    case NBASE_METRIC_COSINE:
      return "cosine";
    case NBASE_METRIC_INNER_PRODUCT:
      return "inner_product";
    case NBASE_METRIC_MANHATTAN:
      return "manhattan";
    case NBASE_METRIC_CHEBYSHEV:
      return "chebyshev";
    case NBASE_METRIC_HAMMING:
      return "hamming";
    default:
      return "unknown";
  }
}

static bool parse_csv_vec(const char *csv, uint32_t expected_dim, float **out_vec) {
  if (!csv || csv[0] == '\0' || expected_dim == 0U) {
    return false;
  }
  float *vec = malloc((size_t)expected_dim * sizeof(float));
  if (!vec) {
    return false;
  }

  const char *p = csv;
  uint32_t idx = 0U;
  while (*p != '\0') {
    if (idx >= expected_dim) {
      free(vec);
      return false;
    }
    errno = 0;
    char *endp = NULL;
    float v = strtof(p, &endp);
    if (errno == ERANGE || endp == p || !isfinite(v)) {
      free(vec);
      return false;
    }
    vec[idx++] = v;
    if (*endp == ',') {
      p = endp + 1;
      if (*p == '\0') {
        free(vec);
        return false;
      }
    } else if (*endp == '\0') {
      p = endp;
    } else {
      free(vec);
      return false;
    }
  }

  if (idx != expected_dim) {
    free(vec);
    return false;
  }
  *out_vec = vec;
  return true;
}

static bool parse_float_arg(const char *s, float *out) {
  if (!s || s[0] == '\0') {
    return false;
  }
  errno = 0;
  char *endp = NULL;
  float v = strtof(s, &endp);
  if (errno == ERANGE || endp == s || *endp != '\0' || !isfinite(v)) {
    return false;
  }
  *out = v;
  return true;
}

static bool parse_json_u64_val(const nbase_json_t *v, uint64_t *out) {
  if (!v) {
    return false;
  }
  if (v->type == NBASE_JSON_NUMBER && v->is_int && v->i64 >= 0) {
    *out = (uint64_t)v->i64;
    return true;
  }
  if (v->type == NBASE_JSON_STRING && v->str) {
    return parse_u64(v->str, out);
  }
  return false;
}

static bool is_dangerous_meta_key(const char *k) {
  return strcmp(k, "__proto__") == 0 || strcmp(k, "constructor") == 0 || strcmp(k, "prototype") == 0;
}

static nbase_status_t cli_parse_meta_obj(
    const nbase_json_t *obj,
    nbase_meta_field_t **out_fields,
    uint32_t *out_cnt) {
  *out_fields = NULL;
  *out_cnt = 0U;
  if (!obj || obj->type != NBASE_JSON_OBJECT) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (obj->len == 0U) {
    return NBASE_OK;
  }
  nbase_meta_field_t *fields = calloc(obj->len, sizeof(*fields));
  if (!fields) {
    return NBASE_ERR_NOMEM;
  }
  uint32_t n = 0U;
  for (size_t i = 0U; i < obj->len; i++) {
    const char *k = obj->keys[i];
    const nbase_json_t *val = obj->vals[i];
    if (!k || k[0] == '\0' || is_dangerous_meta_key(k)) {
      continue;
    }
    if (!val || val->type == NBASE_JSON_NULL || val->type == NBASE_JSON_ARRAY || val->type == NBASE_JSON_OBJECT) {
      free(fields);
      return NBASE_ERR_INVALID_ARG;
    }
    fields[n].key = k;
    if (val->type == NBASE_JSON_BOOL) {
      fields[n].value.type = NBASE_META_BOOL;
      fields[n].value.as.b = val->b;
    } else if (val->type == NBASE_JSON_STRING) {
      fields[n].value.type = NBASE_META_STRING;
      fields[n].value.as.str = val->str;
    } else if (val->type == NBASE_JSON_NUMBER) {
      if (val->is_int) {
        fields[n].value.type = NBASE_META_INT64;
        fields[n].value.as.i64 = val->i64;
      } else {
        fields[n].value.type = NBASE_META_DOUBLE;
        fields[n].value.as.f64 = val->num;
      }
    }
    n++;
  }
  *out_fields = fields;
  *out_cnt = n;
  return NBASE_OK;
}

static nbase_status_t flush_import_batch(
    nbase_db_t *db,
    uint32_t dim,
    uint32_t threads,
    const uint64_t *ids,
    const float *vecs,
    nbase_json_t **roots,
    uint32_t cnt) {
  if (cnt == 0U) {
    return NBASE_OK;
  }
  nbase_status_t st = nbase_add_batch(db, ids, vecs, cnt, dim, threads);
  for (uint32_t i = 0U; i < cnt && st == NBASE_OK; i++) {
    const nbase_json_t *jm = nbase_json_obj_get(roots[i], "metadata");
    if (jm != NULL && jm->type != NBASE_JSON_NULL) {
      nbase_meta_field_t *fields = NULL;
      uint32_t fcnt = 0U;
      st = cli_parse_meta_obj(jm, &fields, &fcnt);
      if (st == NBASE_OK && fcnt > 0U) {
        st = nbase_set_meta(db, ids[i], fields, fcnt);
      }
      free(fields);
    }
    const nbase_json_t *jt = nbase_json_obj_get(roots[i], "text");
    if (st == NBASE_OK && jt != NULL && jt->type != NBASE_JSON_NULL) {
      if (jt->type != NBASE_JSON_STRING) {
        st = NBASE_ERR_INVALID_ARG;
      } else {
        st = nbase_set_text(db, ids[i], jt->str);
      }
    }
  }
  for (uint32_t i = 0U; i < cnt; i++) {
    nbase_json_free(roots[i]);
    roots[i] = NULL;
  }
  return st;
}

typedef struct cli_meta_ctx {
  nbase_sbuf_t *sb;
  bool first;
} cli_meta_ctx_t;

static bool cli_meta_cb(const char *key, const nbase_meta_value_t *val, void *raw) {
  cli_meta_ctx_t *ctx = (cli_meta_ctx_t *)raw;
  if (ctx->first) {
    nbase_sbuf_append(ctx->sb, ",\"metadata\":{");
    ctx->first = false;
  } else {
    nbase_sbuf_append_n(ctx->sb, ",", 1U);
  }
  nbase_sbuf_append_json_str(ctx->sb, key);
  nbase_sbuf_append_n(ctx->sb, ":", 1U);
  switch (val->type) {
    case NBASE_META_INT64:
      nbase_sbuf_appendf(ctx->sb, "%lld", (long long)val->as.i64);
      break;
    case NBASE_META_DOUBLE:
      nbase_sbuf_appendf(ctx->sb, "%.17g", val->as.f64);
      break;
    case NBASE_META_BOOL:
      nbase_sbuf_append(ctx->sb, val->as.b ? "true" : "false");
      break;
    case NBASE_META_STRING:
      nbase_sbuf_append_json_str(ctx->sb, val->as.str);
      break;
  }
  return !ctx->sb->oom;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }
  const char *cmd = argv[1];

  if (strcmp(cmd, "version") == 0) {
    printf("nbase-cli %s (simd: %s)\n", nbase_version(), nbase_simd_backend());
    return 0;
  }

  if (strcmp(cmd, "init") == 0) {
    if (argc < 3) {
      print_usage(argv[0]);
      return 1;
    }
    const char *dir = argv[2];
    uint32_t dim = 0U;
    nbase_metric_t metric = NBASE_METRIC_EUCLIDEAN;
    bool enable_hnsw = true;

    for (int i = 3; i < argc; i++) {
      if (strcmp(argv[i], "--no-hnsw") == 0) {
        enable_hnsw = false;
      } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &dim) || dim == 0U || dim > NBASE_MAX_DIM) {
          fprintf(stderr, "Invalid --dim\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc) {
        if (!parse_metric(argv[++i], &metric)) {
          fprintf(stderr, "Invalid --metric\n");
          return 1;
        }
      } else {
        fprintf(stderr, "Invalid argument: %s\n", argv[i]);
        return 1;
      }
    }

    if (dim == 0U) {
      fprintf(stderr, "init requires --dim <D>\n");
      return 1;
    }

    nbase_config_t cfg = nbase_config_default(dim);
    cfg.metric = metric;
    cfg.enable_hnsw = enable_hnsw;
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_create(&cfg, &db);
    if (st == NBASE_OK) {
      st = nbase_save(db, dir);
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "init failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("Initialized %s (dim=%u, metric=%s, hnsw=%s)\n", dir, dim, metric_str(metric), enable_hnsw ? "on" : "off");
    return 0;
  }

  if (strcmp(cmd, "info") == 0) {
    if (argc != 3) {
      print_usage(argv[0]);
      return 1;
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    st = nbase_stats(db, &s);
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "stats failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf(
        "{\"dim\":%u,\"metric\":\"%s\",\"active_vectors\":%llu,\"total_slots\":%llu,"
        "\"tombstones\":%llu,\"hnsw_enabled\":%s,\"hnsw_max_level\":%d,\"hnsw_active_nodes\":%llu,"
        "\"text_docs\":%llu,\"memory_bytes\":%llu,"
        "\"cache_capacity\":%u,\"cache_entries\":%u,\"cache_hits\":%llu,\"cache_misses\":%llu}\n",
        s.dim,
        metric_str(s.metric),
        (unsigned long long)s.active_vectors,
        (unsigned long long)s.total_slots,
        (unsigned long long)s.tombstones,
        s.hnsw_enabled ? "true" : "false",
        s.hnsw_max_level,
        (unsigned long long)s.hnsw_active_nodes,
        (unsigned long long)s.text_docs,
        (unsigned long long)s.memory_bytes,
        s.cache_capacity,
        s.cache_entries,
        (unsigned long long)s.cache_hits,
        (unsigned long long)s.cache_misses);
    return 0;
  }

  if (strcmp(cmd, "add") == 0) {
    if (argc != 5) {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t id = 0ULL;
    if (!parse_u64(argv[3], &id)) {
      fprintf(stderr, "Invalid uint64 id: %s\n", argv[3]);
      return 1;
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);
    float *vec = NULL;
    if (!parse_csv_vec(argv[4], s.dim, &vec)) {
      nbase_close(db);
      fprintf(stderr, "Invalid vector CSV (expected %u floats)\n", s.dim);
      return 1;
    }
    st = nbase_add(db, id, vec, s.dim);
    free(vec);
    if (st == NBASE_OK) {
      st = nbase_save(db, argv[2]);
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "add failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("Added vector %llu\n", (unsigned long long)id);
    return 0;
  }

  if (strcmp(cmd, "get") == 0) {
    if (argc != 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t id = 0ULL;
    if (!parse_u64(argv[3], &id)) {
      fprintf(stderr, "Invalid uint64 id: %s\n", argv[3]);
      return 1;
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);
    float *vec = malloc((size_t)s.dim * sizeof(float));
    if (!vec) {
      nbase_close(db);
      return 1;
    }
    st = nbase_get(db, id, vec, s.dim);
    nbase_close(db);
    if (st != NBASE_OK) {
      free(vec);
      fprintf(stderr, "get failed: %s\n", nbase_strerror(st));
      return 1;
    }
    for (uint32_t d = 0U; d < s.dim; d++) {
      printf(d == 0U ? "%.7g" : ",%.7g", (double)vec[d]);
    }
    printf("\n");
    free(vec);
    return 0;
  }

  if (strcmp(cmd, "delete") == 0) {
    if (argc != 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t id = 0ULL;
    if (!parse_u64(argv[3], &id)) {
      fprintf(stderr, "Invalid uint64 id: %s\n", argv[3]);
      return 1;
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    st = nbase_delete(db, id);
    if (st == NBASE_OK) {
      st = nbase_save(db, argv[2]);
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "delete failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("Deleted vector %llu\n", (unsigned long long)id);
    return 0;
  }

  if (strcmp(cmd, "delete-batch") == 0) {
    if (argc != 4 || argv[3][0] == '\0') {
      print_usage(argv[0]);
      return 1;
    }
    const char *csv = argv[3];
    uint32_t cap = 16U;
    uint32_t cnt = 0U;
    uint64_t *ids = malloc((size_t)cap * sizeof(uint64_t));
    if (!ids) {
      return 1;
    }
    const char *p = csv;
    while (*p != '\0') {
      const char *comma = strchr(p, ',');
      size_t tok_len = comma ? (size_t)(comma - p) : strlen(p);
      if (tok_len == 0U || tok_len >= 32U) {
        free(ids);
        fprintf(stderr, "Invalid uint64 id list: %s\n", csv);
        return 1;
      }
      char buf[32];
      memcpy(buf, p, tok_len);
      buf[tok_len] = '\0';
      uint64_t id = 0ULL;
      if (!parse_u64(buf, &id)) {
        free(ids);
        fprintf(stderr, "Invalid uint64 id: %s\n", buf);
        return 1;
      }
      if (cnt == cap) {
        if (cap > UINT32_MAX / 2U) {
          free(ids);
          return 1;
        }
        uint32_t new_cap = cap * 2U;
        uint64_t *grown = realloc(ids, (size_t)new_cap * sizeof(uint64_t));
        if (!grown) {
          free(ids);
          return 1;
        }
        ids = grown;
        cap = new_cap;
      }
      ids[cnt++] = id;
      if (!comma) {
        break;
      }
      p = comma + 1;
      if (*p == '\0') {
        free(ids);
        fprintf(stderr, "Trailing comma in id list: %s\n", csv);
        return 1;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(ids);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    uint32_t deleted = 0U;
    st = nbase_delete_batch(db, ids, cnt, &deleted);
    free(ids);
    if (st == NBASE_OK && deleted > 0U) {
      st = nbase_save(db, argv[2]);
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "delete-batch failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("{\"deleted\":%u}\n", deleted);
    return 0;
  }

  if (strcmp(cmd, "set-text") == 0) {
    if (argc != 5) {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t id = 0ULL;
    if (!parse_u64(argv[3], &id)) {
      fprintf(stderr, "Invalid uint64 id: %s\n", argv[3]);
      return 1;
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st == NBASE_OK) {
      st = nbase_set_text(db, id, argv[4]);
    }
    if (st == NBASE_OK) {
      st = nbase_save(db, argv[2]);
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "set-text failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("Set text for vector %llu\n", (unsigned long long)id);
    return 0;
  }

  if (strcmp(cmd, "search-text") == 0) {
    if (argc < 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t k = 10U;
    for (int i = 4; i < argc; i++) {
      if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &k) || k == 0U || k > 10000U) {
          fprintf(stderr, "Invalid --k\n");
          return 1;
        }
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_scored_id_t *res = malloc((size_t)k * sizeof(nbase_scored_id_t));
    if (!res) {
      nbase_close(db);
      return 1;
    }
    uint32_t got = 0U;
    st = nbase_search_text(db, argv[3], k, NULL, res, &got);
    nbase_close(db);
    if (st != NBASE_OK) {
      free(res);
      fprintf(stderr, "search-text failed: %s\n", nbase_strerror(st));
      return 1;
    }
    for (uint32_t i = 0U; i < got; i++) {
      printf("%llu %.7g\n", (unsigned long long)res[i].id, (double)res[i].score);
    }
    free(res);
    return 0;
  }

  if (strcmp(cmd, "search-hybrid") == 0) {
    if (argc < 5) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t k = 10U;
    float alpha = 0.5f;
    float rrf_k = 60.0f;
    for (int i = 5; i < argc; i++) {
      if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &k) || k == 0U || k > 10000U) {
          fprintf(stderr, "Invalid --k\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
        if (!parse_float_arg(argv[++i], &alpha) || alpha < 0.0f || alpha > 1.0f) {
          fprintf(stderr, "Invalid --alpha\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--rrf-k") == 0 && i + 1 < argc) {
        if (!parse_float_arg(argv[++i], &rrf_k) || rrf_k <= 0.0f) {
          fprintf(stderr, "Invalid --rrf-k\n");
          return 1;
        }
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);

    float *q = NULL;
    uint32_t qdim = 0U;
    if (strcmp(argv[3], "-") != 0) {
      if (!parse_csv_vec(argv[3], s.dim, &q)) {
        nbase_close(db);
        fprintf(stderr, "Invalid query CSV (expected %u floats)\n", s.dim);
        return 1;
      }
      qdim = s.dim;
    }
    const char *qtext = (argv[4][0] != '\0') ? argv[4] : NULL;
    nbase_scored_id_t *res = malloc((size_t)k * sizeof(nbase_scored_id_t));
    if (!res) {
      free(q);
      nbase_close(db);
      return 1;
    }
    uint32_t got = 0U;
    st = nbase_search_hybrid(db, q, qdim, qtext, k, alpha, rrf_k, NULL, res, &got);
    free(q);
    nbase_close(db);
    if (st != NBASE_OK) {
      free(res);
      fprintf(stderr, "search-hybrid failed: %s\n", nbase_strerror(st));
      return 1;
    }
    for (uint32_t i = 0U; i < got; i++) {
      printf("%llu %.7g\n", (unsigned long long)res[i].id, (double)res[i].score);
    }
    free(res);
    return 0;
  }

  if (strcmp(cmd, "search") == 0) {
    if (argc < 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t k = 10U;
    nbase_search_opts_t opts = nbase_search_opts_default();
    for (int i = 4; i < argc; i++) {
      if (strcmp(argv[i], "--exact") == 0) {
        opts.mode = NBASE_SEARCH_EXACT;
      } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &k) || k == 0U || k > 10000U) {
          fprintf(stderr, "Invalid --k\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--ef") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &opts.ef_search) || opts.ef_search == 0U) {
          fprintf(stderr, "Invalid --ef\n");
          return 1;
        }
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);
    float *q = NULL;
    if (!parse_csv_vec(argv[3], s.dim, &q)) {
      nbase_close(db);
      fprintf(stderr, "Invalid query CSV (expected %u floats)\n", s.dim);
      return 1;
    }
    nbase_result_t *res = malloc((size_t)k * sizeof(nbase_result_t));
    if (!res) {
      free(q);
      nbase_close(db);
      return 1;
    }
    uint32_t got = 0U;
    st = nbase_search(db, q, s.dim, k, &opts, res, &got);
    free(q);
    nbase_close(db);
    if (st != NBASE_OK) {
      free(res);
      fprintf(stderr, "search failed: %s\n", nbase_strerror(st));
      return 1;
    }
    for (uint32_t i = 0U; i < got; i++) {
      printf("%llu %.7g\n", (unsigned long long)res[i].id, (double)res[i].distance);
    }
    free(res);
    return 0;
  }

  if (strcmp(cmd, "compact") == 0) {
    if (argc != 3) {
      print_usage(argv[0]);
      return 1;
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st == NBASE_OK) {
      st = nbase_compact(db);
    }
    if (st == NBASE_OK) {
      st = nbase_save(db, argv[2]);
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "compact failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("Compacted %s\n", argv[2]);
    return 0;
  }

  if (strcmp(cmd, "checkpoint") == 0) {
    if (argc != 3) {
      print_usage(argv[0]);
      return 1;
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open_wal(argv[2], &db);
    if (st == NBASE_OK) {
      st = nbase_wal_checkpoint(db);
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "checkpoint failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("Checkpointed WAL in %s\n", argv[2]);
    return 0;
  }

  if (strcmp(cmd, "rebuild") == 0) {
    if (argc < 3) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t threads = 0U;
    for (int i = 3; i < argc; i++) {
      if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &threads) || threads > 64U) {
          fprintf(stderr, "Invalid --threads\n");
          return 1;
        }
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st == NBASE_OK) {
      st = nbase_rebuild_hnsw_mt(db, threads);
    }
    if (st == NBASE_OK) {
      st = nbase_save(db, argv[2]);
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "rebuild failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("Rebuilt HNSW index in %s\n", argv[2]);
    return 0;
  }

  if (strcmp(cmd, "import") == 0) {
    if (argc < 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t threads = 0U;
    for (int i = 4; i < argc; i++) {
      if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &threads) || threads > 64U) {
          fprintf(stderr, "Invalid --threads\n");
          return 1;
        }
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);

    FILE *fp = (strcmp(argv[3], "-") == 0) ? stdin : fopen(argv[3], "r");
    if (!fp) {
      fprintf(stderr, "Failed to open input %s\n", argv[3]);
      nbase_close(db);
      return 1;
    }

    uint64_t *b_ids = malloc(NBASE_CLI_IMPORT_BATCH * sizeof(uint64_t));
    float *b_vecs = malloc((size_t)NBASE_CLI_IMPORT_BATCH * (size_t)s.dim * sizeof(float));
    nbase_json_t **b_roots = calloc(NBASE_CLI_IMPORT_BATCH, sizeof(nbase_json_t *));
    if (!b_ids || !b_vecs || !b_roots) {
      free(b_ids);
      free(b_vecs);
      free(b_roots);
      if (fp != stdin) {
        fclose(fp);
      }
      nbase_close(db);
      return 1;
    }

    char *line = NULL;
    size_t line_cap = 0U;
    ssize_t nread;
    uint32_t b_cnt = 0U;
    uint64_t total_imported = 0ULL;
    bool ok = true;

    while ((nread = getline(&line, &line_cap, fp)) >= 0) {
      size_t len = (size_t)nread;
      while (len > 0U && (line[len - 1U] == '\n' || line[len - 1U] == '\r')) {
        len--;
      }
      line[len] = '\0';
      const char *trim = line;
      while (*trim != '\0' && isspace((unsigned char)*trim)) {
        trim++;
      }
      if (*trim == '\0') {
        continue;
      }

      nbase_json_t *root = NULL;
      if (nbase_json_parse(line, len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
        fprintf(stderr, "Invalid JSON line in NDJSON input\n");
        nbase_json_free(root);
        ok = false;
        break;
      }

      uint64_t id = 0ULL;
      if (!parse_json_u64_val(nbase_json_obj_get(root, "id"), &id)) {
        fprintf(stderr, "NDJSON record missing valid uint64 id\n");
        nbase_json_free(root);
        ok = false;
        break;
      }

      const nbase_json_t *jvec = nbase_json_obj_get(root, "vector");
      if (!jvec || jvec->type != NBASE_JSON_ARRAY || jvec->len != (size_t)s.dim) {
        fprintf(stderr, "NDJSON record vector missing or dimension mismatch\n");
        nbase_json_free(root);
        ok = false;
        break;
      }

      float *dst = b_vecs + (size_t)b_cnt * (size_t)s.dim;
      bool vec_ok = true;
      for (uint32_t d = 0U; d < s.dim; d++) {
        const nbase_json_t *it = jvec->items[d];
        if (!it || it->type != NBASE_JSON_NUMBER || !isfinite(it->num)) {
          vec_ok = false;
          break;
        }
        dst[d] = (float)it->num;
        if (!isfinite(dst[d])) {
          vec_ok = false;
          break;
        }
      }
      if (!vec_ok) {
        fprintf(stderr, "NDJSON record contains non-finite vector value\n");
        nbase_json_free(root);
        ok = false;
        break;
      }

      b_ids[b_cnt] = id;
      b_roots[b_cnt] = root;
      b_cnt++;

      if (b_cnt == NBASE_CLI_IMPORT_BATCH) {
        st = flush_import_batch(db, s.dim, threads, b_ids, b_vecs, b_roots, b_cnt);
        if (st != NBASE_OK) {
          fprintf(stderr, "import batch failed: %s\n", nbase_strerror(st));
          ok = false;
          b_cnt = 0U;
          break;
        }
        total_imported += b_cnt;
        b_cnt = 0U;
      }
    }

    if (ok && b_cnt > 0U) {
      st = flush_import_batch(db, s.dim, threads, b_ids, b_vecs, b_roots, b_cnt);
      if (st != NBASE_OK) {
        fprintf(stderr, "import batch failed: %s\n", nbase_strerror(st));
        ok = false;
      } else {
        total_imported += b_cnt;
      }
      b_cnt = 0U;
    }

    for (uint32_t i = 0U; i < b_cnt; i++) {
      nbase_json_free(b_roots[i]);
    }
    free(line);
    free(b_roots);
    free(b_vecs);
    free(b_ids);
    if (fp != stdin) {
      fclose(fp);
    }

    if (ok) {
      st = nbase_save(db, argv[2]);
      if (st != NBASE_OK) {
        fprintf(stderr, "save failed: %s\n", nbase_strerror(st));
        ok = false;
      }
    }
    nbase_close(db);
    if (!ok) {
      return 1;
    }
    printf("Imported %llu vectors into %s\n", (unsigned long long)total_imported, argv[2]);
    return 0;
  }

  if (strcmp(cmd, "export") == 0) {
    if (argc < 3 || argc > 4) {
      print_usage(argv[0]);
      return 1;
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);

    FILE *out = (argc == 4 && strcmp(argv[3], "-") != 0) ? fopen(argv[3], "w") : stdout;
    if (!out) {
      fprintf(stderr, "Failed to open output %s\n", argv[3]);
      nbase_close(db);
      return 1;
    }

    bool ok = true;
    if (s.active_vectors > 0ULL) {
      uint32_t limit = (s.active_vectors > (uint64_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)s.active_vectors;
      uint64_t *ids = malloc((size_t)limit * sizeof(uint64_t));
      float *vec = malloc((size_t)s.dim * sizeof(float));
      char *txt_buf = NULL;
      size_t txt_cap = 0U;
      nbase_sbuf_t sb;
      nbase_sbuf_init(&sb);

      if (!ids || !vec) {
        ok = false;
      } else {
        uint32_t got = 0U;
        st = nbase_query_meta(db, NULL, limit, ids, &got);
        if (st != NBASE_OK) {
          fprintf(stderr, "query failed: %s\n", nbase_strerror(st));
          ok = false;
        }
        for (uint32_t i = 0U; i < got && ok; i++) {
          st = nbase_get(db, ids[i], vec, s.dim);
          if (st != NBASE_OK) {
            ok = false;
            break;
          }
          sb.len = 0U;
          nbase_sbuf_appendf(&sb, "{\"id\":%llu,\"vector\":[", (unsigned long long)ids[i]);
          for (uint32_t d = 0U; d < s.dim; d++) {
            nbase_sbuf_appendf(&sb, d == 0U ? "%.7g" : ",%.7g", (double)vec[d]);
          }
          nbase_sbuf_append_n(&sb, "]", 1U);

          cli_meta_ctx_t mctx = {&sb, true};
          (void)nbase_foreach_meta(db, ids[i], cli_meta_cb, &mctx);
          if (!mctx.first) {
            nbase_sbuf_append_n(&sb, "}", 1U);
          }

          size_t tlen = 0U;
          if (nbase_get_text(db, ids[i], NULL, 0U, &tlen) == NBASE_OK) {
            if (tlen + 1U > txt_cap) {
              char *nb = realloc(txt_buf, tlen + 1U);
              if (!nb) {
                ok = false;
                break;
              }
              txt_buf = nb;
              txt_cap = tlen + 1U;
            }
            if (nbase_get_text(db, ids[i], txt_buf, txt_cap, &tlen) == NBASE_OK) {
              nbase_sbuf_append(&sb, ",\"text\":");
              nbase_sbuf_append_json_str(&sb, txt_buf);
            }
          }
          nbase_sbuf_append(&sb, "}\n");
          if (sb.oom || !sb.data || fwrite(sb.data, 1U, sb.len, out) != sb.len) {
            ok = false;
            break;
          }
        }
      }
      nbase_sbuf_free(&sb);
      free(txt_buf);
      free(vec);
      free(ids);
    }

    if (out != stdout) {
      fclose(out);
    }
    nbase_close(db);
    return ok ? 0 : 1;
  }

  print_usage(argv[0]);
  return 1;
}
