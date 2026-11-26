#include "nbase.h"
#include "nbase_json.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
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
      "  upsert <dir> <id> <v0,v1,...>\n"
      "  patch <dir> <id> [--vector <v0,v1,...>] [--meta <json>] [--text <str>|--clear-text]\n"
      "  patch-batch <dir> <file.ndjson|->\n"
      "  get <dir> <id>\n"
      "  get-batch <dir> <id1,id2,...>\n"
      "  exists <dir> <id1,id2,...>\n"
      "  list <dir> [--offset <N>] [--limit <M>] [--include-vectors] [--include-meta] [--include-text]\n"
      "  count <dir> [--filter <json>]\n"
      "  delete <dir> <id>\n"
      "  delete-batch <dir> <id1,id2,...>\n"
      "  delete-by-filter <dir> --filter <json>\n"
      "  set-meta <dir> <id> <json-object> [--update]\n"
      "  set-meta-batch <dir> <file.ndjson|-> [--update]\n"
      "  get-meta <dir> <id>\n"
      "  get-meta-batch <dir> <id1,id2,...>\n"
      "  delete-meta <dir> <id>\n"
      "  delete-meta-batch <dir> <id1,id2,...>\n"
      "  set-text <dir> <id> <text>\n"
      "  set-text-batch <dir> <file.ndjson|->\n"
      "  get-text <dir> <id>\n"
      "  get-text-batch <dir> <id1,id2,...>\n"
      "  delete-text <dir> <id>\n"
      "  delete-text-batch <dir> <id1,id2,...>\n"
      "  search <dir> <q0,q1,...> [--k <K>] [--ef <EF>] [--exact] [--metric <M>] [--max-distance <D>] [--filter <json>] [--json] [--include-vectors] [--include-meta] [--include-text]\n"
      "  search-batch <dir> <file.ndjson|-> [--k <K>] [--ef <EF>] [--exact] [--metric <M>] [--threads <T>] [--filter <json>]\n"
      "  search-meta <dir> [--filter <json>] [--limit <N>] [--include-vectors] [--include-meta] [--include-text]\n"
      "  rerank <dir> <diversity|weighted|standard> <id1:dist1,id2:dist2,...> [--k <K>] [--lambda <L>] [--weights <json>]\n"
      "  find-similar <dir> <id> [--k <K>] [--ef <EF>] [--exact] [--metric <M>] [--filter <json>] [--json] [--include-vectors] [--include-meta] [--include-text]\n"
      "  relationships <dir> [--k <K>] [--max-distance <D>] [--filter <json>]\n"
      "  communities <dir> [--k <K>] [--max-distance <D>] [--filter <json>]\n"
      "  distance <metric> <q0,q1,...> <v0,v1,...> [<w0,w1,...> ...]\n"
      "  search-text <dir> <query> [--k <K>] [--filter <json>]\n"
      "  search-text-batch <dir> <file.ndjson|-> [--k <K>] [--threads <T>] [--filter <json>]\n"
      "  search-hybrid <dir> <q0,q1,...|-> <query_text> [--k <K>] [--alpha <A>] [--rrf-k <R>] [--metric <M>] [--filter <json>]\n"
      "  search-hybrid-batch <dir> <file.ndjson|-> [--k <K>] [--alpha <A>] [--rrf-k <R>] [--metric <M>] [--threads <T>] [--filter <json>]\n"
      "  rebuild <dir> [--threads <T>]\n"
      "  compact <dir>\n"
      "  checkpoint <dir>\n"
      "  import <dir> <file.ndjson|-> [--upsert] [--threads <T>]\n"
      "  export <dir> [file.ndjson|-] [--ids <id1,id2,...>] [--filter <json>]\n",
      prog);
}

static char *cli_strdup(const char *s) {
  size_t n = strlen(s) + 1U;
  char *out = malloc(n);
  if (out) {
    memcpy(out, s, n);
  }
  return out;
}

static void cli_print_scored_batch_json(
    uint32_t qcnt,
    uint32_t k,
    const nbase_scored_id_t *res,
    const uint32_t *counts) {
  printf("{\"queryCount\":%u,\"results\":[", qcnt);
  for (uint32_t q = 0U; q < qcnt; q++) {
    printf(q == 0U ? "[" : ",[");
    const nbase_scored_id_t *row = res + (size_t)q * (size_t)k;
    for (uint32_t j = 0U; j < counts[q]; j++) {
      printf(j == 0U ? "{\"id\":%llu,\"score\":%.7g}" : ",{\"id\":%llu,\"score\":%.7g}",
             (unsigned long long)row[j].id,
             (double)row[j].score);
    }
    printf("]");
  }
  printf("]}\n");
}

static int cmp_cli_u64_asc(const void *pa, const void *pb) {
  uint64_t a = *(const uint64_t *)pa;
  uint64_t b = *(const uint64_t *)pb;
  return (a > b) - (a < b);
}

static bool cli_u64_in_sorted(const uint64_t *arr, uint32_t len, uint64_t target) {
  if (!arr || len == 0U) {
    return false;
  }
  return bsearch(&target, arr, len, sizeof(uint64_t), cmp_cli_u64_asc) != NULL;
}

static int cmp_cli_res_asc(const void *pa, const void *pb) {
  const nbase_result_t *a = (const nbase_result_t *)pa;
  const nbase_result_t *b = (const nbase_result_t *)pb;
  if (a->distance < b->distance) {
    return -1;
  }
  if (a->distance > b->distance) {
    return 1;
  }
  return (a->id > b->id) - (a->id < b->id);
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

static bool parse_u64_list_csv(const char *csv, uint64_t **out_ids, uint32_t *out_count) {
  *out_ids = NULL;
  *out_count = 0U;
  if (!csv || csv[0] == '\0') {
    fprintf(stderr, "Empty uint64 id list\n");
    return false;
  }
  uint32_t cap = 16U;
  uint32_t cnt = 0U;
  uint64_t *ids = malloc((size_t)cap * sizeof(uint64_t));
  if (!ids) {
    return false;
  }
  const char *p = csv;
  while (*p != '\0') {
    const char *comma = strchr(p, ',');
    size_t tok_len = comma ? (size_t)(comma - p) : strlen(p);
    if (tok_len == 0U || tok_len >= 32U || cnt >= 100000U) {
      free(ids);
      fprintf(stderr, "Invalid uint64 id list: %s\n", csv);
      return false;
    }
    char buf[32];
    memcpy(buf, p, tok_len);
    buf[tok_len] = '\0';
    uint64_t id = 0ULL;
    if (!parse_u64(buf, &id)) {
      free(ids);
      fprintf(stderr, "Invalid uint64 id: %s\n", buf);
      return false;
    }
    if (cnt == cap) {
      uint32_t new_cap = cap * 2U;
      uint64_t *grown = realloc(ids, (size_t)new_cap * sizeof(uint64_t));
      if (!grown) {
        free(ids);
        return false;
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
      return false;
    }
  }
  *out_ids = ids;
  *out_count = cnt;
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

static bool cli_fill_meta_value(const nbase_json_t *val, nbase_meta_value_t *out) {
  if (!val || val->type == NBASE_JSON_NULL || val->type == NBASE_JSON_ARRAY || val->type == NBASE_JSON_OBJECT) {
    return false;
  }
  if (val->type == NBASE_JSON_BOOL) {
    out->type = NBASE_META_BOOL;
    out->as.b = val->b;
  } else if (val->type == NBASE_JSON_STRING) {
    out->type = NBASE_META_STRING;
    out->as.str = val->str;
  } else if (val->type == NBASE_JSON_NUMBER) {
    if (val->is_int) {
      out->type = NBASE_META_INT64;
      out->as.i64 = val->i64;
    } else {
      out->type = NBASE_META_DOUBLE;
      out->as.f64 = val->num;
    }
  } else {
    return false;
  }
  return true;
}

static bool cli_parse_meta_op(const char *s, nbase_meta_op_t *out) {
  if (strcmp(s, "eq") == 0) {
    *out = NBASE_OP_EQ;
  } else if (strcmp(s, "ne") == 0) {
    *out = NBASE_OP_NE;
  } else if (strcmp(s, "gt") == 0) {
    *out = NBASE_OP_GT;
  } else if (strcmp(s, "gte") == 0) {
    *out = NBASE_OP_GTE;
  } else if (strcmp(s, "lt") == 0) {
    *out = NBASE_OP_LT;
  } else if (strcmp(s, "lte") == 0) {
    *out = NBASE_OP_LTE;
  } else if (strcmp(s, "exists") == 0) {
    *out = NBASE_OP_EXISTS;
  } else if (strcmp(s, "contains") == 0) {
    *out = NBASE_OP_CONTAINS;
  } else {
    return false;
  }
  return true;
}

static nbase_status_t cli_parse_meta_filter_json(
    const nbase_json_t *flt,
    nbase_meta_cond_t **out_conds,
    nbase_meta_filter_t *out_filter) {
  *out_conds = NULL;
  out_filter->conds = NULL;
  out_filter->count = 0U;
  if (!flt || flt->type == NBASE_JSON_NULL) {
    return NBASE_OK;
  }
  if (flt->type == NBASE_JSON_OBJECT) {
    if (flt->len == 0U) {
      return NBASE_OK;
    }
    nbase_meta_cond_t *conds = calloc(flt->len, sizeof(*conds));
    if (!conds) {
      return NBASE_ERR_NOMEM;
    }
    uint32_t n = 0U;
    for (size_t i = 0U; i < flt->len; i++) {
      if (!flt->keys[i] || flt->keys[i][0] == '\0' || is_dangerous_meta_key(flt->keys[i])) {
        continue;
      }
      if (!cli_fill_meta_value(flt->vals[i], &conds[n].value)) {
        free(conds);
        return NBASE_ERR_INVALID_ARG;
      }
      conds[n].key = flt->keys[i];
      conds[n].op = NBASE_OP_EQ;
      n++;
    }
    *out_conds = conds;
    out_filter->conds = conds;
    out_filter->count = n;
    return NBASE_OK;
  }
  if (flt->type == NBASE_JSON_ARRAY) {
    if (flt->len == 0U) {
      return NBASE_OK;
    }
    nbase_meta_cond_t *conds = calloc(flt->len, sizeof(*conds));
    if (!conds) {
      return NBASE_ERR_NOMEM;
    }
    uint32_t n = 0U;
    for (size_t i = 0U; i < flt->len; i++) {
      const nbase_json_t *c = flt->items[i];
      if (!c || c->type != NBASE_JSON_OBJECT) {
        free(conds);
        return NBASE_ERR_INVALID_ARG;
      }
      const nbase_json_t *jk = nbase_json_obj_get(c, "key");
      const nbase_json_t *jop = nbase_json_obj_get(c, "op");
      const nbase_json_t *jval = nbase_json_obj_get(c, "value");
      if (!jk || jk->type != NBASE_JSON_STRING || jk->str[0] == '\0' ||
          !jop || jop->type != NBASE_JSON_STRING) {
        free(conds);
        return NBASE_ERR_INVALID_ARG;
      }
      nbase_meta_op_t op = NBASE_OP_EQ;
      if (!cli_parse_meta_op(jop->str, &op)) {
        free(conds);
        return NBASE_ERR_INVALID_ARG;
      }
      conds[n].key = jk->str;
      conds[n].op = op;
      if (op == NBASE_OP_EXISTS) {
        conds[n].value.type = NBASE_META_BOOL;
        conds[n].value.as.b = (jval && jval->type == NBASE_JSON_BOOL) ? jval->b : true;
      } else if (!cli_fill_meta_value(jval, &conds[n].value)) {
        free(conds);
        return NBASE_ERR_INVALID_ARG;
      }
      n++;
    }
    *out_conds = conds;
    out_filter->conds = conds;
    out_filter->count = n;
    return NBASE_OK;
  }
  return NBASE_ERR_INVALID_ARG;
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
    bool upsert,
    const uint64_t *ids,
    const float *vecs,
    nbase_json_t **roots,
    uint32_t cnt) {
  if (cnt == 0U) {
    return NBASE_OK;
  }
  nbase_status_t st = upsert ? nbase_upsert_batch(db, ids, vecs, cnt, dim, threads)
                             : nbase_add_batch(db, ids, vecs, cnt, dim, threads);
  nbase_meta_batch_item_t *meta_items = calloc(cnt, sizeof(*meta_items));
  nbase_meta_field_t **field_bufs = calloc(cnt, sizeof(*field_bufs));
  uint64_t *text_ids = malloc((size_t)cnt * sizeof(uint64_t));
  const char **text_ptrs = malloc((size_t)cnt * sizeof(const char *));
  if (!meta_items || !field_bufs || !text_ids || !text_ptrs) {
    if (st == NBASE_OK) {
      st = NBASE_ERR_NOMEM;
    }
  }

  uint32_t meta_cnt = 0U;
  uint32_t text_cnt = 0U;
  for (uint32_t i = 0U; i < cnt && st == NBASE_OK; i++) {
    const nbase_json_t *jm = nbase_json_obj_get(roots[i], "metadata");
    if (jm != NULL && jm->type != NBASE_JSON_NULL) {
      nbase_meta_field_t *fields = NULL;
      uint32_t fcnt = 0U;
      st = cli_parse_meta_obj(jm, &fields, &fcnt);
      if (st == NBASE_OK && fcnt > 0U) {
        field_bufs[meta_cnt] = fields;
        meta_items[meta_cnt].id = ids[i];
        meta_items[meta_cnt].fields = fields;
        meta_items[meta_cnt].field_count = fcnt;
        meta_cnt++;
      } else {
        free(fields);
      }
    }
    const nbase_json_t *jt = nbase_json_obj_get(roots[i], "text");
    if (st == NBASE_OK && jt != NULL && jt->type != NBASE_JSON_NULL) {
      if (jt->type != NBASE_JSON_STRING) {
        st = NBASE_ERR_INVALID_ARG;
      } else {
        text_ids[text_cnt] = ids[i];
        text_ptrs[text_cnt] = jt->str;
        text_cnt++;
      }
    }
  }
  if (st == NBASE_OK && meta_cnt > 0U) {
    st = nbase_set_meta_batch(db, meta_items, meta_cnt);
  }
  if (st == NBASE_OK && text_cnt > 0U) {
    st = nbase_set_text_batch(db, text_ids, text_ptrs, text_cnt);
  }

  for (uint32_t i = 0U; i < meta_cnt; i++) {
    free(field_bufs[i]);
  }
  free(meta_items);
  free(field_bufs);
  free(text_ids);
  free((void *)text_ptrs);
  for (uint32_t i = 0U; i < cnt; i++) {
    nbase_json_free(roots[i]);
    roots[i] = NULL;
  }
  return st;
}

static void cli_free_ndjson_id_records(uint64_t *ids, nbase_json_t **roots, uint32_t count) {
  if (roots != NULL) {
    for (uint32_t i = 0U; i < count; i++) {
      nbase_json_free(roots[i]);
    }
    free(roots);
  }
  free(ids);
}

static bool cli_read_ndjson_id_records(
    const char *path,
    uint64_t **out_ids,
    nbase_json_t ***out_roots,
    uint32_t *out_count) {
  *out_ids = NULL;
  *out_roots = NULL;
  *out_count = 0U;

  FILE *fp = (strcmp(path, "-") == 0) ? stdin : fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open input %s\n", path);
    return false;
  }

  uint32_t cap = 64U;
  uint32_t cnt = 0U;
  uint64_t *ids = malloc((size_t)cap * sizeof(uint64_t));
  nbase_json_t **roots = calloc(cap, sizeof(nbase_json_t *));
  char *line = NULL;
  size_t line_cap = 0U;
  bool ok = (ids != NULL && roots != NULL);
  ssize_t nread = 0;

  while (ok && (nread = getline(&line, &line_cap, fp)) >= 0) {
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
    if (cnt >= 100000U) {
      fprintf(stderr, "Batch NDJSON exceeds maximum 100000 records\n");
      ok = false;
      break;
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

    if (cnt == cap) {
      uint32_t new_cap = cap * 2U;
      uint64_t *new_ids = realloc(ids, (size_t)new_cap * sizeof(uint64_t));
      nbase_json_t **new_roots = new_ids ? realloc(roots, (size_t)new_cap * sizeof(nbase_json_t *)) : NULL;
      if (!new_ids || !new_roots) {
        if (new_ids) {
          ids = new_ids;
        }
        if (new_roots) {
          roots = new_roots;
        }
        nbase_json_free(root);
        ok = false;
        break;
      }
      ids = new_ids;
      roots = new_roots;
      memset(roots + cap, 0, (size_t)(new_cap - cap) * sizeof(nbase_json_t *));
      cap = new_cap;
    }
    ids[cnt] = id;
    roots[cnt] = root;
    cnt++;
  }

  free(line);
  if (fp != stdin) {
    fclose(fp);
  }
  if (ok && cnt == 0U) {
    fprintf(stderr, "NDJSON input is empty\n");
    ok = false;
  }
  if (!ok) {
    cli_free_ndjson_id_records(ids, roots, cnt);
    return false;
  }
  *out_ids = ids;
  *out_roots = roots;
  *out_count = cnt;
  return true;
}

typedef struct cli_meta_ctx {
  nbase_sbuf_t *sb;
  bool first;
  const char *prefix;
} cli_meta_ctx_t;

static bool cli_meta_cb(const char *key, const nbase_meta_value_t *val, void *raw) {
  cli_meta_ctx_t *ctx = (cli_meta_ctx_t *)raw;
  if (ctx->first) {
    nbase_sbuf_append(ctx->sb, ctx->prefix);
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

static bool cli_append_enriched_id_array(
    nbase_sbuf_t *sb,
    const nbase_db_t *db,
    uint32_t dim,
    const uint64_t *ids,
    const nbase_result_t *opt_res,
    uint32_t got,
    bool inc_vecs,
    bool inc_meta,
    bool inc_text) {
  uint64_t *id_buf = NULL;
  const uint64_t *lookup_ids = ids;
  float *vecs = NULL;
  bool *vfound = NULL;
  char **texts = NULL;
  bool *tfound = NULL;
  nbase_status_t st = NBASE_OK;

  if (got > 0U && (inc_vecs || inc_text)) {
    if (lookup_ids == NULL && opt_res != NULL) {
      id_buf = malloc((size_t)got * sizeof(uint64_t));
      if (!id_buf) {
        st = NBASE_ERR_NOMEM;
      } else {
        for (uint32_t i = 0U; i < got; i++) {
          id_buf[i] = opt_res[i].id;
        }
        lookup_ids = id_buf;
      }
    }
    if (st == NBASE_OK && inc_vecs) {
      vecs = calloc((size_t)got * (size_t)dim, sizeof(float));
      vfound = calloc(got, sizeof(bool));
      if (!vecs || !vfound) {
        st = NBASE_ERR_NOMEM;
      } else {
        st = nbase_get_batch(db, lookup_ids, got, dim, vecs, vfound);
      }
    }
    if (st == NBASE_OK && inc_text) {
      texts = calloc(got, sizeof(char *));
      tfound = calloc(got, sizeof(bool));
      if (!texts || !tfound) {
        st = NBASE_ERR_NOMEM;
      } else {
        st = nbase_get_text_batch(db, lookup_ids, got, texts, tfound);
      }
    }
  }

  for (uint32_t i = 0U; i < got && st == NBASE_OK && !sb->oom; i++) {
    uint64_t eid = (opt_res != NULL) ? opt_res[i].id : ids[i];
    if (opt_res != NULL) {
      nbase_sbuf_appendf(
          sb,
          i == 0U ? "{\"id\":%llu,\"distance\":%.7g" : ",{\"id\":%llu,\"distance\":%.7g",
          (unsigned long long)eid,
          (double)opt_res[i].distance);
    } else {
      nbase_sbuf_appendf(sb, i == 0U ? "{\"id\":%llu" : ",{\"id\":%llu", (unsigned long long)eid);
    }
    if (inc_vecs && vfound != NULL && vfound[i]) {
      const float *v = vecs + (size_t)i * (size_t)dim;
      nbase_sbuf_append(sb, ",\"vector\":[");
      for (uint32_t d = 0U; d < dim; d++) {
        nbase_sbuf_appendf(sb, d == 0U ? "%.7g" : ",%.7g", (double)v[d]);
      }
      nbase_sbuf_append_n(sb, "]", 1U);
    }
    if (inc_meta) {
      cli_meta_ctx_t mctx = {sb, true, ",\"metadata\":{"};
      nbase_status_t mst = nbase_foreach_meta(db, eid, cli_meta_cb, &mctx);
      if (mst != NBASE_OK) {
        st = mst;
        break;
      }
      if (mctx.first) {
        nbase_sbuf_append(sb, ",\"metadata\":{}");
      } else {
        nbase_sbuf_append_n(sb, "}", 1U);
      }
    }
    if (inc_text) {
      if (tfound != NULL && tfound[i] && texts != NULL && texts[i] != NULL) {
        nbase_sbuf_append(sb, ",\"text\":");
        nbase_sbuf_append_json_str(sb, texts[i]);
      } else {
        nbase_sbuf_append(sb, ",\"text\":null");
      }
    }
    nbase_sbuf_append_n(sb, "}", 1U);
  }

  nbase_free_texts(texts, got);
  free(texts);
  free(tfound);
  free(vecs);
  free(vfound);
  free(id_buf);

  if (st != NBASE_OK || sb->oom) {
    fprintf(stderr, "format entries failed: %s\n", nbase_strerror(st != NBASE_OK ? st : NBASE_ERR_NOMEM));
    return false;
  }
  return true;
}

static bool cli_print_search_results_json(
    const nbase_db_t *db,
    uint32_t dim,
    const nbase_result_t *res,
    uint32_t got,
    bool inc_vecs,
    bool inc_meta,
    bool inc_text) {
  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"count\":%u,\"results\":[", got);
  bool ok = cli_append_enriched_id_array(&sb, db, dim, NULL, res, got, inc_vecs, inc_meta, inc_text);
  if (ok) {
    nbase_sbuf_append(&sb, "]}");
    ok = !sb.oom && (sb.data != NULL);
  }
  if (!ok) {
    nbase_sbuf_free(&sb);
    return false;
  }
  printf("%s\n", sb.data);
  nbase_sbuf_free(&sb);
  return true;
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

  if (strcmp(cmd, "add") == 0 || strcmp(cmd, "upsert") == 0) {
    bool is_upsert = (strcmp(cmd, "upsert") == 0);
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
    st = is_upsert ? nbase_upsert(db, id, vec, s.dim) : nbase_add(db, id, vec, s.dim);
    free(vec);
    if (st == NBASE_OK) {
      st = nbase_save(db, argv[2]);
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "%s failed: %s\n", cmd, nbase_strerror(st));
      return 1;
    }
    printf(
        is_upsert ? "Upserted vector %llu\n" : "Added vector %llu\n",
        (unsigned long long)id);
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

  if (strcmp(cmd, "get-batch") == 0 || strcmp(cmd, "delete-batch") == 0 ||
      strcmp(cmd, "exists") == 0) {
    bool is_get = (strcmp(cmd, "get-batch") == 0);
    bool is_exists = (strcmp(cmd, "exists") == 0);
    if (argc != 4 || argv[3][0] == '\0') {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t *ids = NULL;
    uint32_t cnt = 0U;
    if (!parse_u64_list_csv(argv[3], &ids, &cnt)) {
      return 1;
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(ids);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }

    if (is_exists) {
      uint32_t existing_cnt = 0U;
      printf("{\"count\":%u,\"results\":[", cnt);
      for (uint32_t i = 0U; i < cnt; i++) {
        bool ex = nbase_contains(db, ids[i]);
        if (ex) {
          existing_cnt++;
        }
        printf(
            i == 0U ? "{\"id\":%llu,\"exists\":%s}" : ",{\"id\":%llu,\"exists\":%s}",
            (unsigned long long)ids[i],
            ex ? "true" : "false");
      }
      printf("],\"existingCount\":%u}\n", existing_cnt);
      nbase_close(db);
      free(ids);
      return 0;
    }

    if (is_get) {
      nbase_stats_t s;
      (void)nbase_stats(db, &s);
      float *vecs = malloc((size_t)cnt * (size_t)s.dim * sizeof(float));
      bool *found = malloc((size_t)cnt * sizeof(bool));
      if (!vecs || !found) {
        free(vecs);
        free(found);
        free(ids);
        nbase_close(db);
        return 1;
      }
      st = nbase_get_batch(db, ids, cnt, s.dim, vecs, found);
      nbase_close(db);
      if (st != NBASE_OK) {
        free(vecs);
        free(found);
        free(ids);
        fprintf(stderr, "get-batch failed: %s\n", nbase_strerror(st));
        return 1;
      }
      printf("{\"results\":[");
      for (uint32_t i = 0U; i < cnt; i++) {
        printf(
            i == 0U ? "{\"id\":%llu,\"found\":%s" : ",{\"id\":%llu,\"found\":%s",
            (unsigned long long)ids[i],
            found[i] ? "true" : "false");
        if (found[i]) {
          printf(",\"vector\":[");
          const float *v = vecs + (size_t)i * (size_t)s.dim;
          for (uint32_t d = 0U; d < s.dim; d++) {
            printf(d == 0U ? "%.7g" : ",%.7g", (double)v[d]);
          }
          printf("]");
        }
        printf("}");
      }
      printf("]}\n");
      free(vecs);
      free(found);
      free(ids);
      return 0;
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

  if (strcmp(cmd, "set-meta") == 0) {
    if (argc < 5 || argc > 6) {
      print_usage(argv[0]);
      return 1;
    }
    bool is_update = false;
    if (argc == 6) {
      if (strcmp(argv[5], "--update") == 0) {
        is_update = true;
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[5]);
        return 1;
      }
    }
    uint64_t id = 0ULL;
    if (!parse_u64(argv[3], &id)) {
      fprintf(stderr, "Invalid uint64 id: %s\n", argv[3]);
      return 1;
    }
    nbase_json_t *root = NULL;
    if (nbase_json_parse(argv[4], strlen(argv[4]), &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
      nbase_json_free(root);
      fprintf(stderr, "Invalid JSON metadata object\n");
      return 1;
    }
    nbase_meta_field_t *fields = NULL;
    uint32_t fcnt = 0U;
    nbase_status_t st = cli_parse_meta_obj(root, &fields, &fcnt);
    if (st != NBASE_OK) {
      free(fields);
      nbase_json_free(root);
      fprintf(stderr, "Invalid metadata fields: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_db_t *db = NULL;
    st = nbase_open(argv[2], &db);
    if (st == NBASE_OK) {
      st = is_update ? nbase_update_meta(db, id, fields, fcnt) : nbase_set_meta(db, id, fields, fcnt);
    }
    if (st == NBASE_OK) {
      st = nbase_save(db, argv[2]);
    }
    nbase_close(db);
    free(fields);
    nbase_json_free(root);
    if (st != NBASE_OK) {
      fprintf(stderr, "set-meta failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("{\"id\":%llu,\"%s\":%u}\n", (unsigned long long)id, is_update ? "updated" : "set", fcnt);
    return 0;
  }

  if (strcmp(cmd, "set-meta-batch") == 0) {
    if (argc < 4 || argc > 5) {
      print_usage(argv[0]);
      return 1;
    }
    bool is_update = false;
    if (argc == 5) {
      if (strcmp(argv[4], "--update") != 0) {
        fprintf(stderr, "Invalid option: %s\n", argv[4]);
        return 1;
      }
      is_update = true;
    }

    uint64_t *ids = NULL;
    nbase_json_t **roots = NULL;
    uint32_t cnt = 0U;
    if (!cli_read_ndjson_id_records(argv[3], &ids, &roots, &cnt)) {
      return 1;
    }

    nbase_meta_batch_item_t *items = calloc(cnt, sizeof(*items));
    nbase_meta_field_t **field_bufs = calloc(cnt, sizeof(*field_bufs));
    if (!items || !field_bufs) {
      free(items);
      free(field_bufs);
      cli_free_ndjson_id_records(ids, roots, cnt);
      return 1;
    }

    bool parse_ok = true;
    for (uint32_t i = 0U; i < cnt; i++) {
      const nbase_json_t *jm = nbase_json_obj_get(roots[i], "metadata");
      if (!jm || jm->type != NBASE_JSON_OBJECT) {
        fprintf(stderr, "NDJSON record missing metadata object\n");
        parse_ok = false;
        break;
      }
      nbase_meta_field_t *fields = NULL;
      uint32_t fcnt = 0U;
      nbase_status_t pst = cli_parse_meta_obj(jm, &fields, &fcnt);
      if (pst != NBASE_OK) {
        free(fields);
        fprintf(stderr, "Invalid metadata fields in batch: %s\n", nbase_strerror(pst));
        parse_ok = false;
        break;
      }
      field_bufs[i] = fields;
      items[i].id = ids[i];
      items[i].fields = fields;
      items[i].field_count = fcnt;
    }

    int rc = 1;
    if (parse_ok) {
      nbase_db_t *db = NULL;
      nbase_status_t st = nbase_open(argv[2], &db);
      if (st != NBASE_OK) {
        fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      } else {
        st = is_update ? nbase_update_meta_batch(db, items, cnt)
                       : nbase_set_meta_batch(db, items, cnt);
        if (st == NBASE_OK) {
          st = nbase_save(db, argv[2]);
        }
        if (st != NBASE_OK) {
          fprintf(stderr, "%s failed: %s\n", cmd, nbase_strerror(st));
        } else {
          printf("{\"%s\":%u}\n", is_update ? "updated" : "set", cnt);
          rc = 0;
        }
        nbase_close(db);
      }
    }

    for (uint32_t i = 0U; i < cnt; i++) {
      free(field_bufs[i]);
    }
    free(field_bufs);
    free(items);
    cli_free_ndjson_id_records(ids, roots, cnt);
    return rc;
  }

  if (strcmp(cmd, "get-meta") == 0) {
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
    if (!nbase_contains(db, id)) {
      nbase_close(db);
      fprintf(stderr, "get-meta failed: not found\n");
      return 1;
    }
    nbase_sbuf_t sb;
    nbase_sbuf_init(&sb);
    cli_meta_ctx_t mctx = {&sb, true, "{"};
    st = nbase_foreach_meta(db, id, cli_meta_cb, &mctx);
    nbase_close(db);
    if (st == NBASE_OK) {
      if (mctx.first) {
        nbase_sbuf_append(&sb, "{}");
      } else {
        nbase_sbuf_append_n(&sb, "}", 1U);
      }
    }
    if (st != NBASE_OK || sb.oom || !sb.data) {
      nbase_sbuf_free(&sb);
      fprintf(stderr, "get-meta failed: %s\n", nbase_strerror(st != NBASE_OK ? st : NBASE_ERR_NOMEM));
      return 1;
    }
    printf("%s\n", sb.data);
    nbase_sbuf_free(&sb);
    return 0;
  }

  if (strcmp(cmd, "count") == 0) {
    if (argc < 3) {
      print_usage(argv[0]);
      return 1;
    }
    const char *flt_json = NULL;
    for (int i = 3; i < argc; i++) {
      if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }
    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    uint64_t total = 0ULL;
    if (st == NBASE_OK) {
      st = nbase_count(db, flt.count > 0U ? &flt : NULL, &total);
    }
    nbase_close(db);
    free(conds);
    nbase_json_free(flt_root);
    if (st != NBASE_OK) {
      fprintf(stderr, "count failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("{\"count\":%llu}\n", (unsigned long long)total);
    return 0;
  }

  if (strcmp(cmd, "delete-by-filter") == 0) {
    if (argc < 5) {
      print_usage(argv[0]);
      return 1;
    }
    const char *flt_json = NULL;
    for (int i = 3; i < argc; i++) {
      if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }
    if (!flt_json) {
      fprintf(stderr, "delete-by-filter requires --filter <json>\n");
      return 1;
    }
    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
        cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "Invalid --filter JSON\n");
      return 1;
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    uint32_t deleted = 0U;
    if (st == NBASE_OK) {
      st = nbase_delete_by_filter(db, flt.count > 0U ? &flt : NULL, &deleted);
    }
    if (st == NBASE_OK && deleted > 0U) {
      st = nbase_save(db, argv[2]);
    }
    nbase_close(db);
    free(conds);
    nbase_json_free(flt_root);
    if (st != NBASE_OK) {
      fprintf(stderr, "delete-by-filter failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("{\"deleted\":%u}\n", deleted);
    return 0;
  }

  if (strcmp(cmd, "delete-meta") == 0) {
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
    if (st == NBASE_OK) {
      st = nbase_delete_meta(db, id);
    }
    if (st == NBASE_OK) {
      st = nbase_save(db, argv[2]);
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "delete-meta failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("{\"id\":%llu,\"deleted_meta\":true}\n", (unsigned long long)id);
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

  if (strcmp(cmd, "set-text-batch") == 0) {
    if (argc != 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t *ids = NULL;
    nbase_json_t **roots = NULL;
    uint32_t cnt = 0U;
    if (!cli_read_ndjson_id_records(argv[3], &ids, &roots, &cnt)) {
      return 1;
    }

    const char **texts = calloc(cnt, sizeof(const char *));
    if (!texts) {
      cli_free_ndjson_id_records(ids, roots, cnt);
      return 1;
    }

    bool parse_ok = true;
    for (uint32_t i = 0U; i < cnt; i++) {
      const nbase_json_t *jt = nbase_json_obj_get(roots[i], "text");
      if (!jt || jt->type != NBASE_JSON_STRING) {
        fprintf(stderr, "NDJSON record missing string text\n");
        parse_ok = false;
        break;
      }
      texts[i] = jt->str;
    }

    int rc = 1;
    if (parse_ok) {
      nbase_db_t *db = NULL;
      nbase_status_t st = nbase_open(argv[2], &db);
      if (st != NBASE_OK) {
        fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      } else {
        st = nbase_set_text_batch(db, ids, texts, cnt);
        if (st == NBASE_OK) {
          st = nbase_save(db, argv[2]);
        }
        if (st != NBASE_OK) {
          fprintf(stderr, "set-text-batch failed: %s\n", nbase_strerror(st));
        } else {
          printf("{\"set\":%u}\n", cnt);
          rc = 0;
        }
        nbase_close(db);
      }
    }

    free((void *)texts);
    cli_free_ndjson_id_records(ids, roots, cnt);
    return rc;
  }

  if (strcmp(cmd, "get-text") == 0) {
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
    size_t len = 0U;
    if (st == NBASE_OK) {
      st = nbase_get_text(db, id, NULL, 0U, &len);
    }
    char *buf = NULL;
    if (st == NBASE_OK) {
      buf = malloc(len + 1U);
      if (!buf) {
        st = NBASE_ERR_NOMEM;
      } else {
        st = nbase_get_text(db, id, buf, len + 1U, &len);
      }
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      free(buf);
      fprintf(stderr, "get-text failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("%s\n", buf);
    free(buf);
    return 0;
  }

  if (strcmp(cmd, "delete-text") == 0) {
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
    if (st == NBASE_OK) {
      st = nbase_delete_text(db, id);
    }
    if (st == NBASE_OK) {
      st = nbase_save(db, argv[2]);
    }
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "delete-text failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("{\"id\":%llu,\"deleted_text\":true}\n", (unsigned long long)id);
    return 0;
  }

  if (strcmp(cmd, "delete-meta-batch") == 0 || strcmp(cmd, "delete-text-batch") == 0) {
    bool is_meta = (strcmp(cmd, "delete-meta-batch") == 0);
    if (argc != 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t *ids = NULL;
    uint32_t cnt = 0U;
    if (!parse_u64_list_csv(argv[3], &ids, &cnt)) {
      return 1;
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    uint32_t deleted = 0U;
    if (st == NBASE_OK) {
      st = is_meta ? nbase_delete_meta_batch(db, ids, cnt, &deleted)
                   : nbase_delete_text_batch(db, ids, cnt, &deleted);
    }
    if (st == NBASE_OK) {
      st = nbase_save(db, argv[2]);
    }
    free(ids);
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "%s failed: %s\n", cmd, nbase_strerror(st));
      return 1;
    }
    printf("{\"deleted\":%u}\n", deleted);
    return 0;
  }

  if (strcmp(cmd, "get-meta-batch") == 0 || strcmp(cmd, "get-text-batch") == 0) {
    bool is_meta = (strcmp(cmd, "get-meta-batch") == 0);
    if (argc != 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t *ids = NULL;
    uint32_t cnt = 0U;
    if (!parse_u64_list_csv(argv[3], &ids, &cnt)) {
      return 1;
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(ids);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }

    nbase_sbuf_t sb;
    nbase_sbuf_init(&sb);
    nbase_sbuf_append_n(&sb, "[", 1U);
    char **texts = NULL;
    bool *found = NULL;
    if (!is_meta && cnt > 0U) {
      texts = calloc(cnt, sizeof(char *));
      found = calloc(cnt, sizeof(bool));
      if (!texts || !found) {
        st = NBASE_ERR_NOMEM;
      } else {
        st = nbase_get_text_batch(db, ids, cnt, texts, found);
      }
    }

    for (uint32_t i = 0U; i < cnt && st == NBASE_OK && !sb.oom; i++) {
      if (i > 0U) {
        nbase_sbuf_append_n(&sb, ",", 1U);
      }
      nbase_sbuf_appendf(&sb, "{\"id\":%llu,", (unsigned long long)ids[i]);
      if (is_meta) {
        if (!nbase_contains(db, ids[i])) {
          nbase_sbuf_append(&sb, "\"found\":false,\"metadata\":null}");
        } else {
          nbase_sbuf_append(&sb, "\"found\":true,\"metadata\":");
          cli_meta_ctx_t mctx = {&sb, true, "{"};
          st = nbase_foreach_meta(db, ids[i], cli_meta_cb, &mctx);
          if (st == NBASE_OK) {
            nbase_sbuf_append(&sb, mctx.first ? "{}}" : "}}");
          }
        }
      } else if (found[i] && texts[i] != NULL) {
        nbase_sbuf_append(&sb, "\"found\":true,\"text\":");
        nbase_sbuf_append_json_str(&sb, texts[i]);
        nbase_sbuf_append_n(&sb, "}", 1U);
      } else {
        nbase_sbuf_append(&sb, "\"found\":false,\"text\":null}");
      }
    }

    if (st == NBASE_OK && !sb.oom) {
      nbase_sbuf_append_n(&sb, "]", 1U);
    }
    nbase_free_texts(texts, cnt);
    free(texts);
    free(found);
    free(ids);
    nbase_close(db);
    if (st != NBASE_OK || sb.oom || !sb.data) {
      nbase_sbuf_free(&sb);
      fprintf(stderr, "%s failed: %s\n", cmd, nbase_strerror(st != NBASE_OK ? st : NBASE_ERR_NOMEM));
      return 1;
    }
    printf("%s\n", sb.data);
    nbase_sbuf_free(&sb);
    return 0;
  }

  if (strcmp(cmd, "search-text") == 0) {
    if (argc < 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t k = 10U;
    const char *flt_json = NULL;
    for (int i = 4; i < argc; i++) {
      if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &k) || k == 0U || k > 10000U) {
          fprintf(stderr, "Invalid --k\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }
    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
    }
    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_scored_id_t *res = malloc((size_t)k * sizeof(nbase_scored_id_t));
    if (!res) {
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      return 1;
    }
    uint32_t got = 0U;
    st = nbase_search_text(db, argv[3], k, flt.count > 0U ? &flt : NULL, res, &got);
    nbase_close(db);
    free(conds);
    nbase_json_free(flt_root);
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

  if (strcmp(cmd, "search-text-batch") == 0) {
    if (argc < 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t k = 10U;
    uint32_t num_threads = 0U;
    const char *flt_json = NULL;
    for (int i = 4; i < argc; i++) {
      if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &k) || k == 0U || k > 10000U) {
          fprintf(stderr, "Invalid --k\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &num_threads) || num_threads > 64U) {
          fprintf(stderr, "Invalid --threads\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
    }

    FILE *fp = (strcmp(argv[3], "-") == 0) ? stdin : fopen(argv[3], "r");
    if (!fp) {
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "Failed to open query file %s\n", argv[3]);
      return 1;
    }

    uint32_t cap = 32U;
    uint32_t qcnt = 0U;
    char **queries = malloc((size_t)cap * sizeof(char *));
    char *line = NULL;
    size_t line_cap = 0U;
    ssize_t nread;
    bool ok = (queries != NULL);

    while (ok && (nread = getline(&line, &line_cap, fp)) >= 0) {
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
      nbase_json_t *qroot = NULL;
      if (nbase_json_parse(line, len, &qroot) != NBASE_OK) {
        nbase_json_free(qroot);
        fprintf(stderr, "Invalid JSON query line\n");
        ok = false;
        break;
      }
      const nbase_json_t *jstr = NULL;
      if (qroot->type == NBASE_JSON_STRING) {
        jstr = qroot;
      } else if (qroot->type == NBASE_JSON_OBJECT) {
        jstr = nbase_json_obj_get(qroot, "query");
        if (!jstr) {
          jstr = nbase_json_obj_get(qroot, "text");
        }
      }
      if (!jstr || jstr->type != NBASE_JSON_STRING || qcnt >= 10000U) {
        nbase_json_free(qroot);
        fprintf(stderr, "Each line must be a JSON string or {\"query\"|\"text\":\"...\"}\n");
        ok = false;
        break;
      }
      if (qcnt == cap) {
        uint32_t new_cap = cap * 2U;
        char **grown = realloc(queries, (size_t)new_cap * sizeof(char *));
        if (!grown) {
          nbase_json_free(qroot);
          ok = false;
          break;
        }
        queries = grown;
        cap = new_cap;
      }
      queries[qcnt] = cli_strdup(jstr->str);
      nbase_json_free(qroot);
      if (!queries[qcnt]) {
        ok = false;
        break;
      }
      qcnt++;
    }
    free(line);
    if (fp != stdin) {
      fclose(fp);
    }

    if (!ok) {
      for (uint32_t i = 0U; i < qcnt; i++) {
        free(queries[i]);
      }
      free(queries);
      free(conds);
      nbase_json_free(flt_root);
      return 1;
    }

    if (qcnt == 0U) {
      free(queries);
      free(conds);
      nbase_json_free(flt_root);
      printf("{\"queryCount\":0,\"results\":[]}\n");
      return 0;
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    nbase_scored_id_t *res =
        (st == NBASE_OK) ? malloc((size_t)qcnt * (size_t)k * sizeof(nbase_scored_id_t)) : NULL;
    uint32_t *counts = (st == NBASE_OK) ? calloc((size_t)qcnt, sizeof(uint32_t)) : NULL;
    if (st == NBASE_OK && (!res || !counts)) {
      st = NBASE_ERR_NOMEM;
    }
    if (st == NBASE_OK) {
      st = nbase_search_text_batch(
          db,
          (const char *const *)queries,
          qcnt,
          k,
          num_threads,
          flt.count > 0U ? &flt : NULL,
          res,
          counts);
    }

    for (uint32_t i = 0U; i < qcnt; i++) {
      free(queries[i]);
    }
    free(queries);
    nbase_close(db);
    free(conds);
    nbase_json_free(flt_root);

    if (st != NBASE_OK) {
      free(res);
      free(counts);
      fprintf(stderr, "search-text-batch failed: %s\n", nbase_strerror(st));
      return 1;
    }
    cli_print_scored_batch_json(qcnt, k, res, counts);
    free(res);
    free(counts);
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
    nbase_search_opts_t opts = nbase_search_opts_default();
    const char *flt_json = NULL;
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
      } else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc) {
        if (!parse_metric(argv[++i], &opts.metric)) {
          fprintf(stderr, "Invalid --metric\n");
          return 1;
        }
        opts.use_metric_override = true;
      } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
      if (flt.count > 0U) {
        opts.meta_filter = &flt;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(conds);
      nbase_json_free(flt_root);
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
        free(conds);
        nbase_json_free(flt_root);
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
      free(conds);
      nbase_json_free(flt_root);
      return 1;
    }
    uint32_t got = 0U;
    st = nbase_search_hybrid(db, q, qdim, qtext, k, alpha, rrf_k, &opts, res, &got);
    free(q);
    nbase_close(db);
    free(conds);
    nbase_json_free(flt_root);
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

  if (strcmp(cmd, "search-hybrid-batch") == 0) {
    if (argc < 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t k = 10U;
    float alpha = 0.5f;
    float rrf_k = 60.0f;
    nbase_search_opts_t opts = nbase_search_opts_default();
    const char *flt_json = NULL;
    for (int i = 4; i < argc; i++) {
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
      } else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc) {
        if (!parse_metric(argv[++i], &opts.metric)) {
          fprintf(stderr, "Invalid --metric\n");
          return 1;
        }
        opts.use_metric_override = true;
      } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &opts.num_threads) || opts.num_threads > 64U) {
          fprintf(stderr, "Invalid --threads\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
      if (flt.count > 0U) {
        opts.meta_filter = &flt;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);

    FILE *fp = (strcmp(argv[3], "-") == 0) ? stdin : fopen(argv[3], "r");
    if (!fp) {
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "Failed to open query file %s\n", argv[3]);
      return 1;
    }

    uint32_t cap = 32U;
    uint32_t qcnt = 0U;
    float **qvecs = calloc(cap, sizeof(float *));
    uint32_t *qdims = calloc(cap, sizeof(uint32_t));
    char **qtexts = calloc(cap, sizeof(char *));
    char *line = NULL;
    size_t line_cap = 0U;
    ssize_t nread;
    bool ok = (qvecs != NULL && qdims != NULL && qtexts != NULL);

    while (ok && (nread = getline(&line, &line_cap, fp)) >= 0) {
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
      nbase_json_t *qroot = NULL;
      if (nbase_json_parse(line, len, &qroot) != NBASE_OK || qroot->type != NBASE_JSON_OBJECT ||
          qcnt >= 10000U) {
        nbase_json_free(qroot);
        fprintf(stderr, "Invalid hybrid JSON query line\n");
        ok = false;
        break;
      }
      if (qcnt == cap) {
        uint32_t new_cap = cap * 2U;
        float **gv = realloc(qvecs, (size_t)new_cap * sizeof(float *));
        uint32_t *gd = realloc(qdims, (size_t)new_cap * sizeof(uint32_t));
        char **gt = realloc(qtexts, (size_t)new_cap * sizeof(char *));
        if (!gv || !gd || !gt) {
          free(gv ? gv : qvecs);
          free(gd ? gd : qdims);
          free(gt ? gt : qtexts);
          qvecs = NULL;
          qdims = NULL;
          qtexts = NULL;
          nbase_json_free(qroot);
          ok = false;
          break;
        }
        qvecs = gv;
        qdims = gd;
        qtexts = gt;
        cap = new_cap;
      }
      qvecs[qcnt] = NULL;
      qdims[qcnt] = 0U;
      qtexts[qcnt] = NULL;

      const nbase_json_t *jqv = nbase_json_obj_get(qroot, "query");
      if (jqv != NULL && jqv->type != NBASE_JSON_NULL) {
        if (jqv->type != NBASE_JSON_ARRAY || jqv->len != (size_t)s.dim) {
          fprintf(stderr, "Invalid query vector or dimension mismatch\n");
          nbase_json_free(qroot);
          ok = false;
          break;
        }
        qvecs[qcnt] = malloc((size_t)s.dim * sizeof(float));
        if (!qvecs[qcnt]) {
          nbase_json_free(qroot);
          ok = false;
          break;
        }
        for (uint32_t d = 0U; d < s.dim; d++) {
          const nbase_json_t *it = jqv->items[d];
          if (!it || it->type != NBASE_JSON_NUMBER || !isfinite(it->num) ||
              !isfinite((float)it->num)) {
            fprintf(stderr, "Non-finite value in query vector\n");
            ok = false;
            break;
          }
          qvecs[qcnt][d] = (float)it->num;
        }
        if (!ok) {
          free(qvecs[qcnt]);
          qvecs[qcnt] = NULL;
          nbase_json_free(qroot);
          break;
        }
        qdims[qcnt] = s.dim;
      }
      const nbase_json_t *jtxt = nbase_json_obj_get(qroot, "text");
      if (jtxt != NULL && jtxt->type != NBASE_JSON_NULL) {
        if (jtxt->type != NBASE_JSON_STRING) {
          fprintf(stderr, "text must be a string\n");
          free(qvecs[qcnt]);
          qvecs[qcnt] = NULL;
          nbase_json_free(qroot);
          ok = false;
          break;
        }
        qtexts[qcnt] = cli_strdup(jtxt->str);
        if (!qtexts[qcnt]) {
          free(qvecs[qcnt]);
          qvecs[qcnt] = NULL;
          nbase_json_free(qroot);
          ok = false;
          break;
        }
      }
      nbase_json_free(qroot);
      if (qvecs[qcnt] == NULL && (qtexts[qcnt] == NULL || qtexts[qcnt][0] == '\0')) {
        free(qtexts[qcnt]);
        qtexts[qcnt] = NULL;
        fprintf(stderr, "Each hybrid query requires query vector or non-empty text\n");
        ok = false;
        break;
      }
      qcnt++;
    }
    free(line);
    if (fp != stdin) {
      fclose(fp);
    }

    if (ok && qcnt == 0U) {
      free(qvecs);
      free(qdims);
      free(qtexts);
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      printf("{\"queryCount\":0,\"results\":[]}\n");
      return 0;
    }

    nbase_scored_id_t *res =
        ok ? malloc((size_t)qcnt * (size_t)k * sizeof(nbase_scored_id_t)) : NULL;
    uint32_t *counts = ok ? calloc((size_t)qcnt, sizeof(uint32_t)) : NULL;
    if (ok && (!res || !counts)) {
      st = NBASE_ERR_NOMEM;
    } else if (ok) {
      bool all_have_vec = true;
      bool none_have_vec = true;
      for (uint32_t q = 0U; q < qcnt; q++) {
        if (qvecs[q] == NULL) {
          all_have_vec = false;
        } else {
          none_have_vec = false;
        }
      }
      if (none_have_vec || all_have_vec) {
        float *packed = NULL;
        if (all_have_vec) {
          packed = malloc((size_t)qcnt * (size_t)s.dim * sizeof(float));
          if (!packed) {
            st = NBASE_ERR_NOMEM;
          } else {
            for (uint32_t q = 0U; q < qcnt; q++) {
              memcpy(packed + (size_t)q * (size_t)s.dim, qvecs[q], (size_t)s.dim * sizeof(float));
            }
          }
        }
        if (st == NBASE_OK) {
          st = nbase_search_hybrid_batch(
              db,
              packed,
              s.dim,
              (const char *const *)qtexts,
              qcnt,
              k,
              alpha,
              rrf_k,
              &opts,
              res,
              counts);
        }
        free(packed);
      } else {
        for (uint32_t q = 0U; q < qcnt && st == NBASE_OK; q++) {
          st = nbase_search_hybrid(
              db,
              qvecs[q],
              qdims[q],
              qtexts[q],
              k,
              alpha,
              rrf_k,
              &opts,
              res + (size_t)q * (size_t)k,
              &counts[q]);
        }
      }
    }

    if (qvecs != NULL) {
      for (uint32_t q = 0U; q < qcnt; q++) {
        free(qvecs[q]);
      }
      free(qvecs);
    }
    free(qdims);
    if (qtexts != NULL) {
      for (uint32_t q = 0U; q < qcnt; q++) {
        free(qtexts[q]);
      }
      free(qtexts);
    }
    nbase_close(db);
    free(conds);
    nbase_json_free(flt_root);

    if (!ok || st != NBASE_OK) {
      free(res);
      free(counts);
      if (ok) {
        fprintf(stderr, "search-hybrid-batch failed: %s\n", nbase_strerror(st));
      }
      return 1;
    }
    cli_print_scored_batch_json(qcnt, k, res, counts);
    free(res);
    free(counts);
    return 0;
  }

  if (strcmp(cmd, "list") == 0) {
    if (argc < 3) {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t offset = 0ULL;
    uint32_t limit = 50U;
    bool inc_vecs = false;
    bool inc_meta = false;
    bool inc_text = false;
    for (int i = 3; i < argc; i++) {
      if (strcmp(argv[i], "--offset") == 0 && i + 1 < argc) {
        if (!parse_u64(argv[++i], &offset)) {
          fprintf(stderr, "Invalid --offset\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--limit") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &limit) || limit == 0U || limit > 10000U) {
          fprintf(stderr, "Invalid --limit\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--include-vectors") == 0) {
        inc_vecs = true;
      } else if (strcmp(argv[i], "--include-meta") == 0) {
        inc_meta = true;
      } else if (strcmp(argv[i], "--include-text") == 0) {
        inc_text = true;
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
    uint32_t dim = 0U;
    if (inc_vecs) {
      nbase_stats_t db_st;
      if (nbase_stats(db, &db_st) != NBASE_OK) {
        nbase_close(db);
        fprintf(stderr, "stats failed\n");
        return 1;
      }
      dim = db_st.dim;
    }
    uint64_t *ids = malloc((size_t)limit * sizeof(uint64_t));
    if (!ids) {
      nbase_close(db);
      return 1;
    }
    uint32_t got = 0U;
    uint64_t total = 0ULL;
    st = nbase_list_ids(db, offset, limit, ids, &got, &total);
    if (st != NBASE_OK) {
      free(ids);
      nbase_close(db);
      fprintf(stderr, "list failed: %s\n", nbase_strerror(st));
      return 1;
    }

    nbase_sbuf_t sb;
    nbase_sbuf_init(&sb);
    nbase_sbuf_appendf(
        &sb,
        "{\"total\":%llu,\"offset\":%llu,\"limit\":%u,\"count\":%u,\"ids\":[",
        (unsigned long long)total,
        (unsigned long long)offset,
        limit,
        got);
    for (uint32_t i = 0U; i < got; i++) {
      nbase_sbuf_appendf(&sb, i == 0U ? "%llu" : ",%llu", (unsigned long long)ids[i]);
    }
    nbase_sbuf_append_n(&sb, "]", 1U);

    bool ok = true;
    if (inc_vecs || inc_meta || inc_text) {
      nbase_sbuf_append(&sb, ",\"results\":[");
      ok = cli_append_enriched_id_array(&sb, db, dim, ids, NULL, got, inc_vecs, inc_meta, inc_text);
      if (ok) {
        nbase_sbuf_append_n(&sb, "]", 1U);
      }
    }
    if (ok) {
      nbase_sbuf_append_n(&sb, "}", 1U);
      ok = !sb.oom && (sb.data != NULL);
    }

    free(ids);
    nbase_close(db);
    if (!ok) {
      nbase_sbuf_free(&sb);
      return 1;
    }
    printf("%s\n", sb.data);
    nbase_sbuf_free(&sb);
    return 0;
  }

  if (strcmp(cmd, "patch") == 0) {
    if (argc < 5) {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t id = 0ULL;
    if (!parse_u64(argv[3], &id)) {
      fprintf(stderr, "Invalid id: %s\n", argv[3]);
      return 1;
    }
    const char *vec_csv = NULL;
    const char *meta_json = NULL;
    const char *text_val = NULL;
    bool clear_text = false;
    for (int i = 4; i < argc; i++) {
      if (strcmp(argv[i], "--vector") == 0 && i + 1 < argc) {
        vec_csv = argv[++i];
      } else if (strcmp(argv[i], "--meta") == 0 && i + 1 < argc) {
        meta_json = argv[++i];
      } else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
        text_val = argv[++i];
      } else if (strcmp(argv[i], "--clear-text") == 0) {
        clear_text = true;
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }
    if ((!vec_csv && !meta_json && !text_val && !clear_text) || (text_val && clear_text)) {
      fprintf(stderr, "Specify at least one of --vector, --meta, --text, or --clear-text (not both text flags)\n");
      return 1;
    }

    nbase_json_t *meta_root = NULL;
    nbase_meta_field_t *fields = NULL;
    uint32_t fcnt = 0U;
    if (meta_json != NULL) {
      if (nbase_json_parse(meta_json, strlen(meta_json), &meta_root) != NBASE_OK ||
          meta_root->type != NBASE_JSON_OBJECT ||
          cli_parse_meta_obj(meta_root, &fields, &fcnt) != NBASE_OK) {
        free(fields);
        nbase_json_free(meta_root);
        fprintf(stderr, "Invalid --meta JSON object\n");
        return 1;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(fields);
      nbase_json_free(meta_root);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    if (!nbase_contains(db, id)) {
      nbase_close(db);
      free(fields);
      nbase_json_free(meta_root);
      fprintf(stderr, "vector id %llu not found\n", (unsigned long long)id);
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);

    float *vec = NULL;
    if (vec_csv != NULL && !parse_csv_vec(vec_csv, s.dim, &vec)) {
      nbase_close(db);
      free(fields);
      nbase_json_free(meta_root);
      fprintf(stderr, "Invalid --vector CSV (expected %u floats)\n", s.dim);
      return 1;
    }

    if (vec != NULL) {
      st = nbase_upsert(db, id, vec, s.dim);
    }
    if (st == NBASE_OK && meta_json != NULL && fcnt > 0U) {
      st = nbase_update_meta(db, id, fields, fcnt);
    }
    if (st == NBASE_OK) {
      if (clear_text) {
        st = nbase_delete_text(db, id);
        if (st == NBASE_ERR_NOT_FOUND) {
          st = NBASE_OK;
        }
      } else if (text_val != NULL) {
        st = nbase_set_text(db, id, text_val);
      }
    }
    if (st == NBASE_OK) {
      st = nbase_save(db, argv[2]);
    }
    free(vec);
    free(fields);
    nbase_json_free(meta_root);
    nbase_close(db);
    if (st != NBASE_OK) {
      fprintf(stderr, "patch failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("{\"id\":%llu,\"patched\":true}\n", (unsigned long long)id);
    return 0;
  }

  if (strcmp(cmd, "patch-batch") == 0) {
    if (argc != 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t *ids = NULL;
    nbase_json_t **roots = NULL;
    uint32_t cnt = 0U;
    if (!cli_read_ndjson_id_records(argv[3], &ids, &roots, &cnt)) {
      return 1;
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      cli_free_ndjson_id_records(ids, roots, cnt);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);

    float **vecs = calloc(cnt, sizeof(float *));
    nbase_meta_field_t **field_bufs = calloc(cnt, sizeof(*field_bufs));
    uint32_t *fcnts = calloc(cnt, sizeof(uint32_t));
    const char **texts = calloc(cnt, sizeof(const char *));
    bool *clear_texts = calloc(cnt, sizeof(bool));
    bool parse_ok = (vecs && field_bufs && fcnts && texts && clear_texts);

    for (uint32_t i = 0U; i < cnt && parse_ok; i++) {
      if (!nbase_contains(db, ids[i])) {
        fprintf(stderr, "vector id %llu not found\n", (unsigned long long)ids[i]);
        parse_ok = false;
        break;
      }
      const nbase_json_t *jv = nbase_json_obj_get(roots[i], "vector");
      const nbase_json_t *jm = nbase_json_obj_get(roots[i], "metadata");
      const nbase_json_t *jt = nbase_json_obj_get(roots[i], "text");
      if (jv == NULL && jm == NULL && jt == NULL) {
        fprintf(stderr, "NDJSON patch record requires vector, metadata, or text\n");
        parse_ok = false;
        break;
      }
      if (jv != NULL) {
        if (jv->type != NBASE_JSON_ARRAY || jv->len != (size_t)s.dim) {
          fprintf(stderr, "NDJSON record vector must be array of length %u\n", s.dim);
          parse_ok = false;
          break;
        }
        vecs[i] = malloc((size_t)s.dim * sizeof(float));
        if (!vecs[i]) {
          parse_ok = false;
          break;
        }
        for (uint32_t d = 0U; d < s.dim; d++) {
          const nbase_json_t *it = jv->items[d];
          if (!it || it->type != NBASE_JSON_NUMBER || !isfinite(it->num)) {
            fprintf(stderr, "NDJSON record vector contains non-finite value\n");
            parse_ok = false;
            break;
          }
          vecs[i][d] = (float)it->num;
        }
        if (!parse_ok) {
          break;
        }
      }
      if (jm != NULL) {
        if (jm->type != NBASE_JSON_OBJECT ||
            cli_parse_meta_obj(jm, &field_bufs[i], &fcnts[i]) != NBASE_OK) {
          fprintf(stderr, "Invalid metadata object in NDJSON patch record\n");
          parse_ok = false;
          break;
        }
      }
      if (jt != NULL) {
        if (jt->type == NBASE_JSON_NULL) {
          clear_texts[i] = true;
        } else if (jt->type == NBASE_JSON_STRING) {
          texts[i] = jt->str;
        } else {
          fprintf(stderr, "NDJSON patch record text must be string or null\n");
          parse_ok = false;
          break;
        }
      }
    }

    int rc = 1;
    if (parse_ok) {
      for (uint32_t i = 0U; i < cnt && st == NBASE_OK; i++) {
        if (vecs[i] != NULL) {
          st = nbase_upsert(db, ids[i], vecs[i], s.dim);
        }
        if (st == NBASE_OK && field_bufs[i] != NULL && fcnts[i] > 0U) {
          st = nbase_update_meta(db, ids[i], field_bufs[i], fcnts[i]);
        }
        if (st == NBASE_OK) {
          if (clear_texts[i]) {
            st = nbase_delete_text(db, ids[i]);
            if (st == NBASE_ERR_NOT_FOUND) {
              st = NBASE_OK;
            }
          } else if (texts[i] != NULL) {
            st = nbase_set_text(db, ids[i], texts[i]);
          }
        }
      }
      if (st == NBASE_OK) {
        st = nbase_save(db, argv[2]);
      }
      if (st != NBASE_OK) {
        fprintf(stderr, "patch-batch failed: %s\n", nbase_strerror(st));
      } else {
        printf("{\"patched\":%u}\n", cnt);
        rc = 0;
      }
    }

    if (vecs != NULL) {
      for (uint32_t i = 0U; i < cnt; i++) {
        free(vecs[i]);
      }
    }
    if (field_bufs != NULL) {
      for (uint32_t i = 0U; i < cnt; i++) {
        free(field_bufs[i]);
      }
    }
    free(vecs);
    free(field_bufs);
    free(fcnts);
    free((void *)texts);
    free(clear_texts);
    cli_free_ndjson_id_records(ids, roots, cnt);
    nbase_close(db);
    return rc;
  }

  if (strcmp(cmd, "search-batch") == 0) {
    if (argc < 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t k = 10U;
    nbase_search_opts_t opts = nbase_search_opts_default();
    const char *flt_json = NULL;
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
      } else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc) {
        if (!parse_metric(argv[++i], &opts.metric)) {
          fprintf(stderr, "Invalid --metric\n");
          return 1;
        }
        opts.use_metric_override = true;
      } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &opts.num_threads) || opts.num_threads > 64U) {
          fprintf(stderr, "Invalid --threads\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
      if (flt.count > 0U) {
        opts.meta_filter = &flt;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);

    FILE *fp = (strcmp(argv[3], "-") == 0) ? stdin : fopen(argv[3], "r");
    if (!fp) {
      fprintf(stderr, "Failed to open query file %s\n", argv[3]);
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      return 1;
    }

    uint32_t cap = 32U;
    uint32_t qcnt = 0U;
    float *queries = malloc((size_t)cap * (size_t)s.dim * sizeof(float));
    char *line = NULL;
    size_t line_cap = 0U;
    ssize_t nread;
    bool ok = (queries != NULL);

    while (ok && (nread = getline(&line, &line_cap, fp)) >= 0) {
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
      nbase_json_t *qroot = NULL;
      if (nbase_json_parse(line, len, &qroot) != NBASE_OK) {
        fprintf(stderr, "Invalid JSON query line\n");
        nbase_json_free(qroot);
        ok = false;
        break;
      }
      const nbase_json_t *jarr = (qroot->type == NBASE_JSON_ARRAY) ? qroot : nbase_json_obj_get(qroot, "query");
      if (!jarr || jarr->type != NBASE_JSON_ARRAY || jarr->len != (size_t)s.dim || qcnt >= 10000U) {
        fprintf(stderr, "Invalid query vector or dimension mismatch\n");
        nbase_json_free(qroot);
        ok = false;
        break;
      }
      if (qcnt == cap) {
        uint32_t new_cap = cap * 2U;
        float *grown = realloc(queries, (size_t)new_cap * (size_t)s.dim * sizeof(float));
        if (!grown) {
          nbase_json_free(qroot);
          ok = false;
          break;
        }
        queries = grown;
        cap = new_cap;
      }
      float *dst = queries + (size_t)qcnt * (size_t)s.dim;
      for (uint32_t d = 0U; d < s.dim; d++) {
        const nbase_json_t *it = jarr->items[d];
        if (!it || it->type != NBASE_JSON_NUMBER || !isfinite(it->num) || !isfinite((float)it->num)) {
          ok = false;
          break;
        }
        dst[d] = (float)it->num;
      }
      nbase_json_free(qroot);
      if (ok) {
        qcnt++;
      } else {
        fprintf(stderr, "Non-finite value in query vector\n");
      }
    }
    free(line);
    if (fp != stdin) {
      fclose(fp);
    }

    if (!ok) {
      free(queries);
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      return 1;
    }

    if (qcnt == 0U) {
      free(queries);
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      printf("{\"queryCount\":0,\"results\":[]}\n");
      return 0;
    }

    nbase_result_t *res = malloc((size_t)qcnt * (size_t)k * sizeof(nbase_result_t));
    uint32_t *counts = malloc((size_t)qcnt * sizeof(uint32_t));
    if (!res || !counts) {
      free(res);
      free(counts);
      free(queries);
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      return 1;
    }

    st = nbase_search_batch(db, queries, qcnt, s.dim, k, &opts, res, counts);
    free(queries);
    nbase_close(db);
    free(conds);
    nbase_json_free(flt_root);

    if (st != NBASE_OK) {
      free(res);
      free(counts);
      fprintf(stderr, "search-batch failed: %s\n", nbase_strerror(st));
      return 1;
    }

    printf("{\"queryCount\":%u,\"results\":[", qcnt);
    for (uint32_t q = 0U; q < qcnt; q++) {
      printf(q == 0U ? "[" : ",[");
      const nbase_result_t *row = res + (size_t)q * (size_t)k;
      for (uint32_t j = 0U; j < counts[q]; j++) {
        printf(j == 0U ? "{\"id\":%llu,\"distance\":%.7g}" : ",{\"id\":%llu,\"distance\":%.7g}",
               (unsigned long long)row[j].id,
               (double)row[j].distance);
      }
      printf("]");
    }
    printf("]}\n");
    free(res);
    free(counts);
    return 0;
  }

  if (strcmp(cmd, "search") == 0) {
    if (argc < 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t k = 10U;
    bool use_radius = false;
    bool json_out = false;
    bool inc_vecs = false;
    bool inc_meta = false;
    bool inc_text = false;
    float max_dist = 0.0f;
    nbase_search_opts_t opts = nbase_search_opts_default();
    const char *flt_json = NULL;
    for (int i = 4; i < argc; i++) {
      if (strcmp(argv[i], "--exact") == 0) {
        opts.mode = NBASE_SEARCH_EXACT;
      } else if (strcmp(argv[i], "--json") == 0) {
        json_out = true;
      } else if (strcmp(argv[i], "--include-vectors") == 0) {
        inc_vecs = true;
      } else if (strcmp(argv[i], "--include-meta") == 0) {
        inc_meta = true;
      } else if (strcmp(argv[i], "--include-text") == 0) {
        inc_text = true;
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
      } else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc) {
        if (!parse_metric(argv[++i], &opts.metric)) {
          fprintf(stderr, "Invalid --metric\n");
          return 1;
        }
        opts.use_metric_override = true;
      } else if (strcmp(argv[i], "--max-distance") == 0 && i + 1 < argc) {
        if (!parse_float_arg(argv[++i], &max_dist)) {
          fprintf(stderr, "Invalid --max-distance\n");
          return 1;
        }
        use_radius = true;
      } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
      if (flt.count > 0U) {
        opts.meta_filter = &flt;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);
    float *q = NULL;
    if (!parse_csv_vec(argv[3], s.dim, &q)) {
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "Invalid query CSV (expected %u floats)\n", s.dim);
      return 1;
    }
    nbase_result_t *res = malloc((size_t)k * sizeof(nbase_result_t));
    if (!res) {
      free(q);
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      return 1;
    }
    uint32_t got = 0U;
    st = use_radius
             ? nbase_search_radius(db, q, s.dim, max_dist, k, &opts, res, &got)
             : nbase_search(db, q, s.dim, k, &opts, res, &got);
    free(q);
    free(conds);
    nbase_json_free(flt_root);
    if (st != NBASE_OK) {
      free(res);
      nbase_close(db);
      fprintf(stderr, "search failed: %s\n", nbase_strerror(st));
      return 1;
    }
    int rc = 0;
    if (json_out || inc_vecs || inc_meta || inc_text) {
      if (!cli_print_search_results_json(db, s.dim, res, got, inc_vecs, inc_meta, inc_text)) {
        rc = 1;
      }
    } else {
      for (uint32_t i = 0U; i < got; i++) {
        printf("%llu %.7g\n", (unsigned long long)res[i].id, (double)res[i].distance);
      }
    }
    free(res);
    nbase_close(db);
    return rc;
  }

  if (strcmp(cmd, "find-similar") == 0) {
    if (argc < 4) {
      print_usage(argv[0]);
      return 1;
    }
    uint64_t id = 0ULL;
    if (!parse_u64(argv[3], &id)) {
      fprintf(stderr, "Invalid uint64 id: %s\n", argv[3]);
      return 1;
    }
    uint32_t k = 10U;
    bool json_out = false;
    bool inc_vecs = false;
    bool inc_meta = false;
    bool inc_text = false;
    nbase_search_opts_t opts = nbase_search_opts_default();
    const char *flt_json = NULL;
    for (int i = 4; i < argc; i++) {
      if (strcmp(argv[i], "--exact") == 0) {
        opts.mode = NBASE_SEARCH_EXACT;
      } else if (strcmp(argv[i], "--json") == 0) {
        json_out = true;
      } else if (strcmp(argv[i], "--include-vectors") == 0) {
        inc_vecs = true;
      } else if (strcmp(argv[i], "--include-meta") == 0) {
        inc_meta = true;
      } else if (strcmp(argv[i], "--include-text") == 0) {
        inc_text = true;
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
      } else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc) {
        if (!parse_metric(argv[++i], &opts.metric)) {
          fprintf(stderr, "Invalid --metric\n");
          return 1;
        }
        opts.use_metric_override = true;
      } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
      if (flt.count > 0U) {
        opts.meta_filter = &flt;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);
    nbase_result_t *res = malloc((size_t)k * sizeof(nbase_result_t));
    if (!res) {
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      return 1;
    }
    uint32_t got = 0U;
    st = nbase_find_similar(db, id, k, &opts, res, &got);
    free(conds);
    nbase_json_free(flt_root);
    if (st != NBASE_OK) {
      free(res);
      nbase_close(db);
      fprintf(stderr, "find-similar failed: %s\n", nbase_strerror(st));
      return 1;
    }
    int rc = 0;
    if (json_out || inc_vecs || inc_meta || inc_text) {
      if (!cli_print_search_results_json(db, s.dim, res, got, inc_vecs, inc_meta, inc_text)) {
        rc = 1;
      }
    } else {
      for (uint32_t i = 0U; i < got; i++) {
        printf("%llu %.7g\n", (unsigned long long)res[i].id, (double)res[i].distance);
      }
    }
    free(res);
    nbase_close(db);
    return rc;
  }

  if (strcmp(cmd, "relationships") == 0) {
    if (argc < 3) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t k = 5U;
    float max_dist = FLT_MAX;
    const char *flt_json = NULL;
    for (int i = 3; i < argc; i++) {
      if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &k) || k == 0U || k > 1000U) {
          fprintf(stderr, "Invalid --k\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--max-distance") == 0 && i + 1 < argc) {
        if (!parse_float_arg(argv[++i], &max_dist)) {
          fprintf(stderr, "Invalid --max-distance\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_edge_t *edges = NULL;
    uint64_t edge_cnt = 0ULL;
    st = nbase_extract_relationships_filtered(
        db, k, max_dist, flt.count > 0U ? &flt : NULL, &edges, &edge_cnt);
    nbase_close(db);
    free(conds);
    nbase_json_free(flt_root);
    if (st != NBASE_OK) {
      fprintf(stderr, "relationships failed: %s\n", nbase_strerror(st));
      return 1;
    }
    printf("{\"count\":%llu,\"edges\":[", (unsigned long long)edge_cnt);
    for (uint64_t i = 0ULL; i < edge_cnt; i++) {
      printf(
          i == 0ULL ? "{\"src\":%llu,\"dst\":%llu,\"weight\":%.7g}"
                    : ",{\"src\":%llu,\"dst\":%llu,\"weight\":%.7g}",
          (unsigned long long)edges[i].src_id,
          (unsigned long long)edges[i].dst_id,
          (double)edges[i].weight);
    }
    printf("]}\n");
    nbase_free_edges(edges);
    return 0;
  }

  if (strcmp(cmd, "communities") == 0) {
    if (argc < 3) {
      print_usage(argv[0]);
      return 1;
    }
    uint32_t k = 5U;
    float max_dist = FLT_MAX;
    const char *flt_json = NULL;
    for (int i = 3; i < argc; i++) {
      if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &k) || k == 0U || k > 1000U) {
          fprintf(stderr, "Invalid --k\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--max-distance") == 0 && i + 1 < argc) {
        if (!parse_float_arg(argv[++i], &max_dist)) {
          fprintf(stderr, "Invalid --max-distance\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }

    nbase_stats_t db_st;
    st = nbase_stats(db, &db_st);
    if (st != NBASE_OK) {
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "stats failed: %s\n", nbase_strerror(st));
      return 1;
    }
    if (db_st.active_vectors == 0ULL) {
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      printf("{\"community_count\":0,\"communities\":[]}\n");
      return 0;
    }
    if (db_st.active_vectors > 1000000ULL) {
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "too many nodes for communities command\n");
      return 1;
    }

    uint32_t cap = (uint32_t)db_st.active_vectors + 64U;
    uint64_t *ids = malloc((size_t)cap * sizeof(uint64_t));
    if (!ids) {
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "out of memory\n");
      return 1;
    }
    uint32_t got_ids = 0U;
    st = nbase_query_meta(db, flt.count > 0U ? &flt : NULL, cap, ids, &got_ids);
    if (st != NBASE_OK) {
      free(ids);
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "query failed: %s\n", nbase_strerror(st));
      return 1;
    }
    if (got_ids == 0U) {
      free(ids);
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      printf("{\"community_count\":0,\"communities\":[]}\n");
      return 0;
    }

    nbase_edge_t *edges = NULL;
    uint64_t edge_cnt = 0ULL;
    st = nbase_extract_relationships_filtered(
        db, k, max_dist, flt.count > 0U ? &flt : NULL, &edges, &edge_cnt);
    nbase_close(db);
    free(conds);
    nbase_json_free(flt_root);
    if (st != NBASE_OK) {
      free(ids);
      fprintf(stderr, "relationships failed: %s\n", nbase_strerror(st));
      return 1;
    }

    uint32_t *labels = malloc((size_t)got_ids * sizeof(uint32_t));
    if (!labels) {
      nbase_free_edges(edges);
      free(ids);
      fprintf(stderr, "out of memory\n");
      return 1;
    }
    uint32_t comm_cnt = 0U;
    st = nbase_extract_communities(ids, got_ids, edges, edge_cnt, labels, &comm_cnt);
    nbase_free_edges(edges);
    if (st != NBASE_OK) {
      free(labels);
      free(ids);
      fprintf(stderr, "communities failed: %s\n", nbase_strerror(st));
      return 1;
    }

    printf("{\"community_count\":%u,\"communities\":[", comm_cnt);
    for (uint32_t i = 0U; i < got_ids; i++) {
      printf(
          i == 0U ? "{\"id\":%llu,\"community\":%u}" : ",{\"id\":%llu,\"community\":%u}",
          (unsigned long long)ids[i],
          labels[i]);
    }
    printf("]}\n");
    free(labels);
    free(ids);
    return 0;
  }

  if (strcmp(cmd, "distance") == 0) {
    if (argc < 5) {
      print_usage(argv[0]);
      return 1;
    }
    nbase_metric_t metric = NBASE_METRIC_EUCLIDEAN;
    if (!parse_metric(argv[2], &metric)) {
      fprintf(stderr, "Invalid metric: %s\n", argv[2]);
      return 1;
    }
    uint32_t dim = 1U;
    for (const char *p = argv[3]; *p != '\0'; p++) {
      if (*p == ',') {
        if (dim >= NBASE_MAX_DIM) {
          fprintf(stderr, "Query dimension exceeds maximum\n");
          return 1;
        }
        dim++;
      }
    }
    float *qvec = NULL;
    if (!parse_csv_vec(argv[3], dim, &qvec)) {
      fprintf(stderr, "Invalid query vector\n");
      return 1;
    }

    uint32_t count = (uint32_t)(argc - 4);
    float *targets = malloc((size_t)count * (size_t)dim * sizeof(float));
    float *dists = malloc((size_t)count * sizeof(float));
    if (!targets || !dists) {
      free(targets);
      free(dists);
      free(qvec);
      fprintf(stderr, "out of memory\n");
      return 1;
    }

    for (uint32_t i = 0U; i < count; i++) {
      float *row = NULL;
      if (!parse_csv_vec(argv[4 + (int)i], dim, &row)) {
        free(targets);
        free(dists);
        free(qvec);
        fprintf(stderr, "Invalid target vector at index %u\n", i);
        return 1;
      }
      memcpy(targets + (size_t)i * (size_t)dim, row, (size_t)dim * sizeof(float));
      free(row);
    }

    nbase_status_t st = nbase_distance_batch(metric, qvec, targets, count, dim, dists);
    free(targets);
    free(qvec);
    if (st != NBASE_OK) {
      free(dists);
      fprintf(stderr, "distance failed: %s\n", nbase_strerror(st));
      return 1;
    }

    printf("{\"metric\":\"%s\",\"dimension\":%u,\"count\":%u,\"distances\":[",
           metric_str(metric),
           dim,
           count);
    for (uint32_t i = 0U; i < count; i++) {
      printf(i == 0U ? "%.7g" : ",%.7g", (double)dists[i]);
    }
    printf("]}\n");
    free(dists);
    return 0;
  }

  if (strcmp(cmd, "search-meta") == 0) {
    if (argc < 3) {
      print_usage(argv[0]);
      return 1;
    }
    const char *flt_json = NULL;
    uint32_t limit = 100U;
    bool inc_vecs = false;
    bool inc_meta = false;
    bool inc_text = false;
    for (int i = 3; i < argc; i++) {
      if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else if (strcmp(argv[i], "--limit") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &limit) || limit == 0U || limit > 10000U) {
          fprintf(stderr, "Invalid --limit\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--include-vectors") == 0) {
        inc_vecs = true;
      } else if (strcmp(argv[i], "--include-meta") == 0) {
        inc_meta = true;
      } else if (strcmp(argv[i], "--include-text") == 0) {
        inc_text = true;
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }

    nbase_stats_t db_st;
    st = nbase_stats(db, &db_st);
    uint64_t *ids = malloc((size_t)limit * sizeof(uint64_t));
    if (st != NBASE_OK || !ids) {
      free(ids);
      nbase_close(db);
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "search-meta failed\n");
      return 1;
    }

    uint32_t got = 0U;
    st = nbase_query_meta(db, flt.count > 0U ? &flt : NULL, limit, ids, &got);
    free(conds);
    nbase_json_free(flt_root);
    if (st != NBASE_OK) {
      free(ids);
      nbase_close(db);
      fprintf(stderr, "query failed: %s\n", nbase_strerror(st));
      return 1;
    }

    nbase_sbuf_t sb;
    nbase_sbuf_init(&sb);
    nbase_sbuf_appendf(&sb, "{\"count\":%u,\"results\":[", got);
    bool ok = cli_append_enriched_id_array(
        &sb, db, db_st.dim, ids, NULL, got, inc_vecs, inc_meta, inc_text);
    if (ok) {
      nbase_sbuf_append(&sb, "]}");
      ok = !sb.oom && (sb.data != NULL);
    }
    free(ids);
    nbase_close(db);
    if (!ok) {
      nbase_sbuf_free(&sb);
      return 1;
    }
    printf("%s\n", sb.data);
    nbase_sbuf_free(&sb);
    return 0;
  }

  if (strcmp(cmd, "rerank") == 0) {
    if (argc < 5) {
      print_usage(argv[0]);
      return 1;
    }
    const char *method = argv[3];
    int mcode = 0;
    if (strcmp(method, "diversity") == 0) {
      mcode = 1;
    } else if (strcmp(method, "weighted") == 0) {
      mcode = 2;
    } else if (strcmp(method, "standard") == 0) {
      mcode = 3;
    } else {
      fprintf(stderr, "Invalid rerank method: %s\n", method);
      return 1;
    }

    uint32_t cand_cap = 32U;
    uint32_t cand_cnt = 0U;
    nbase_result_t *cands = malloc((size_t)cand_cap * sizeof(nbase_result_t));
    if (!cands) {
      return 1;
    }
    const char *p = argv[4];
    if (*p == '\0') {
      free(cands);
      fprintf(stderr, "Candidates list cannot be empty\n");
      return 1;
    }
    while (*p != '\0') {
      const char *comma = strchr(p, ',');
      size_t tok_len = comma ? (size_t)(comma - p) : strlen(p);
      if (tok_len == 0U || cand_cnt >= 10000U) {
        free(cands);
        fprintf(stderr, "Invalid candidate token\n");
        return 1;
      }
      const char *colon = memchr(p, ':', tok_len);
      if (!colon || colon == p || (size_t)(colon - p) >= 32U) {
        free(cands);
        fprintf(stderr, "Each candidate must be <id>:<distance>\n");
        return 1;
      }
      char idbuf[32];
      size_t idlen = (size_t)(colon - p);
      memcpy(idbuf, p, idlen);
      idbuf[idlen] = '\0';
      uint64_t cid = 0ULL;
      if (!parse_u64(idbuf, &cid)) {
        free(cands);
        fprintf(stderr, "Invalid candidate id: %s\n", idbuf);
        return 1;
      }
      size_t dlen = tok_len - idlen - 1U;
      if (dlen == 0U || dlen >= 64U) {
        free(cands);
        fprintf(stderr, "Invalid candidate distance\n");
        return 1;
      }
      char dbuf[64];
      memcpy(dbuf, colon + 1, dlen);
      dbuf[dlen] = '\0';
      float cd = 0.0f;
      if (!parse_float_arg(dbuf, &cd)) {
        free(cands);
        fprintf(stderr, "Invalid candidate distance: %s\n", dbuf);
        return 1;
      }
      if (cand_cnt == cand_cap) {
        cand_cap *= 2U;
        nbase_result_t *next = realloc(cands, (size_t)cand_cap * sizeof(nbase_result_t));
        if (!next) {
          free(cands);
          return 1;
        }
        cands = next;
      }
      cands[cand_cnt].id = cid;
      cands[cand_cnt].distance = cd;
      cand_cnt++;
      p = comma ? (comma + 1) : (p + tok_len);
      if (comma && *p == '\0') {
        free(cands);
        fprintf(stderr, "Trailing comma in candidates\n");
        return 1;
      }
    }

    uint32_t k = cand_cnt;
    float lambda = 0.5f;
    const char *weights_json = NULL;
    for (int i = 5; i < argc; i++) {
      if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
        if (!parse_u32(argv[++i], &k) || k == 0U || k > 10000U) {
          free(cands);
          fprintf(stderr, "Invalid --k\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--lambda") == 0 && i + 1 < argc) {
        if (!parse_float_arg(argv[++i], &lambda) || lambda < 0.0f || lambda > 1.0f) {
          free(cands);
          fprintf(stderr, "Invalid --lambda (must be in [0, 1])\n");
          return 1;
        }
      } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
        weights_json = argv[++i];
      } else {
        free(cands);
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    nbase_json_t *w_root = NULL;
    nbase_rerank_weight_t *weights = NULL;
    uint32_t weight_cnt = 0U;
    if (weights_json != NULL) {
      if (nbase_json_parse(weights_json, strlen(weights_json), &w_root) != NBASE_OK ||
          w_root->type != NBASE_JSON_OBJECT || w_root->len > 1024U) {
        nbase_json_free(w_root);
        free(cands);
        fprintf(stderr, "Invalid --weights JSON object\n");
        return 1;
      }
      if (w_root->len > 0U) {
        weights = calloc(w_root->len, sizeof(nbase_rerank_weight_t));
        if (!weights) {
          nbase_json_free(w_root);
          free(cands);
          return 1;
        }
        for (size_t wi = 0U; wi < w_root->len; wi++) {
          const char *wk = w_root->keys[wi];
          const nbase_json_t *wv = w_root->vals[wi];
          if (!wk || wk[0] == '\0' || !wv || wv->type != NBASE_JSON_NUMBER || !isfinite(wv->num)) {
            free(weights);
            nbase_json_free(w_root);
            free(cands);
            fprintf(stderr, "Weights values must be finite numbers\n");
            return 1;
          }
          weights[weight_cnt].key = wk;
          weights[weight_cnt].weight = (float)wv->num;
          weight_cnt++;
        }
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(weights);
      nbase_json_free(w_root);
      free(cands);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }

    nbase_result_t *out_res = malloc((size_t)k * sizeof(nbase_result_t));
    if (!out_res) {
      nbase_close(db);
      free(weights);
      nbase_json_free(w_root);
      free(cands);
      return 1;
    }
    uint32_t out_n = 0U;
    if (mcode == 2) {
      st = nbase_rerank_weighted(db, cands, cand_cnt, weights, weight_cnt, k, out_res, &out_n);
    } else if (mcode == 3) {
      for (uint32_t i = 0U; i < cand_cnt; i++) {
        if (!nbase_contains(db, cands[i].id)) {
          st = NBASE_ERR_NOT_FOUND;
          break;
        }
      }
      if (st == NBASE_OK) {
        qsort(cands, cand_cnt, sizeof(nbase_result_t), cmp_cli_res_asc);
        out_n = (cand_cnt < k) ? cand_cnt : k;
        if (out_n > 0U) {
          memcpy(out_res, cands, (size_t)out_n * sizeof(nbase_result_t));
        }
      }
    } else {
      st = nbase_rerank_mmr(db, cands, cand_cnt, k, lambda, out_res, &out_n);
    }

    nbase_close(db);
    free(weights);
    nbase_json_free(w_root);
    free(cands);
    if (st != NBASE_OK) {
      free(out_res);
      fprintf(stderr, "rerank failed: %s\n", nbase_strerror(st));
      return 1;
    }

    printf("{\"method\":\"%s\",\"count\":%u,\"results\":[", method, out_n);
    for (uint32_t i = 0U; i < out_n; i++) {
      printf(
          i == 0U ? "{\"id\":%llu,\"distance\":%.7g}" : ",{\"id\":%llu,\"distance\":%.7g}",
          (unsigned long long)out_res[i].id,
          (double)out_res[i].distance);
    }
    printf("]}\n");
    free(out_res);
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
    bool upsert = false;
    for (int i = 4; i < argc; i++) {
      if (strcmp(argv[i], "--upsert") == 0) {
        upsert = true;
      } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
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
        st = flush_import_batch(db, s.dim, threads, upsert, b_ids, b_vecs, b_roots, b_cnt);
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
      st = flush_import_batch(db, s.dim, threads, upsert, b_ids, b_vecs, b_roots, b_cnt);
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
    if (argc < 3) {
      print_usage(argv[0]);
      return 1;
    }
    const char *out_path = NULL;
    const char *flt_json = NULL;
    const char *ids_csv = NULL;
    for (int i = 3; i < argc; i++) {
      if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
        flt_json = argv[++i];
      } else if (strcmp(argv[i], "--ids") == 0 && i + 1 < argc) {
        ids_csv = argv[++i];
      } else if ((argv[i][0] != '-' || strcmp(argv[i], "-") == 0) && !out_path) {
        out_path = argv[i];
      } else {
        fprintf(stderr, "Invalid option: %s\n", argv[i]);
        return 1;
      }
    }

    uint64_t *req_ids = NULL;
    uint32_t req_cnt = 0U;
    if (ids_csv != NULL && !parse_u64_list_csv(ids_csv, &req_ids, &req_cnt)) {
      return 1;
    }

    nbase_json_t *flt_root = NULL;
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t flt = {NULL, 0U};
    if (flt_json != NULL) {
      if (nbase_json_parse(flt_json, strlen(flt_json), &flt_root) != NBASE_OK ||
          cli_parse_meta_filter_json(flt_root, &conds, &flt) != NBASE_OK) {
        free(req_ids);
        free(conds);
        nbase_json_free(flt_root);
        fprintf(stderr, "Invalid --filter JSON\n");
        return 1;
      }
    }

    nbase_db_t *db = NULL;
    nbase_status_t st = nbase_open(argv[2], &db);
    if (st != NBASE_OK) {
      free(req_ids);
      free(conds);
      nbase_json_free(flt_root);
      fprintf(stderr, "open failed: %s\n", nbase_strerror(st));
      return 1;
    }
    nbase_stats_t s;
    (void)nbase_stats(db, &s);

    FILE *out = (out_path != NULL && strcmp(out_path, "-") != 0) ? fopen(out_path, "w") : stdout;
    if (!out) {
      fprintf(stderr, "Failed to open output %s\n", out_path ? out_path : "stdout");
      free(req_ids);
      free(conds);
      nbase_json_free(flt_root);
      nbase_close(db);
      return 1;
    }

    bool ok = true;
    if (s.active_vectors > 0ULL) {
      uint32_t limit = (s.active_vectors > (uint64_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)s.active_vectors;
      uint64_t *ids = NULL;
      uint32_t got = 0U;
      float *vec = malloc((size_t)s.dim * sizeof(float));
      char *txt_buf = NULL;
      size_t txt_cap = 0U;
      nbase_sbuf_t sb;
      nbase_sbuf_init(&sb);

      if (!vec) {
        ok = false;
      } else if (req_ids != NULL && flt.count == 0U) {
        ids = req_ids;
        req_ids = NULL;
        for (uint32_t i = 0U; i < req_cnt; i++) {
          if (nbase_contains(db, ids[i])) {
            ids[got++] = ids[i];
          }
        }
        free(conds);
        conds = NULL;
        nbase_json_free(flt_root);
        flt_root = NULL;
      } else {
        ids = malloc((size_t)limit * sizeof(uint64_t));
        if (!ids) {
          ok = false;
        } else {
          st = nbase_query_meta(db, flt.count > 0U ? &flt : NULL, limit, ids, &got);
          free(conds);
          conds = NULL;
          nbase_json_free(flt_root);
          flt_root = NULL;
          if (st != NBASE_OK) {
            fprintf(stderr, "query failed: %s\n", nbase_strerror(st));
            ok = false;
          } else if (req_ids != NULL) {
            if (got > 1U) {
              qsort(ids, got, sizeof(uint64_t), cmp_cli_u64_asc);
            }
            uint32_t filtered_cnt = 0U;
            for (uint32_t i = 0U; i < req_cnt; i++) {
              if (cli_u64_in_sorted(ids, got, req_ids[i])) {
                req_ids[filtered_cnt++] = req_ids[i];
              }
            }
            free(ids);
            ids = req_ids;
            req_ids = NULL;
            got = filtered_cnt;
          }
        }
      }
      if (ok && ids != NULL) {
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

          cli_meta_ctx_t mctx = {&sb, true, ",\"metadata\":{"};
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
    free(req_ids);
    free(conds);
    nbase_json_free(flt_root);
    nbase_close(db);
    return ok ? 0 : 1;
  }

  print_usage(argv[0]);
  return 1;
}
