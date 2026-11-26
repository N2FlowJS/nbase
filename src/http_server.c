#include "nbase_internal.h"
#include "nbase_json.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#define NBASE_HTTP_QUEUE_CAP 256U
#define NBASE_HTTP_MAX_HDR 16384U
#define NBASE_HTTP_MAX_KEEPALIVE_REQS 100U
#define NBASE_HTTP_LAT_WINDOW 256U

struct nbase_http_server {
  nbase_db_t *db;
  int listen_fd;
  int wakeup_pipe[2];
  uint16_t bound_port;
  uint32_t num_workers;
  size_t max_body_bytes;
  uint32_t socket_timeout_ms;
  char *api_key;
  char *save_dir;
  uint32_t max_rps;
  char *cors_origin;

  pthread_mutex_t rl_lock;
  uint64_t rl_sec;
  uint32_t rl_count;

  _Atomic uint64_t req_total;
  _Atomic uint64_t err_total;
  _Atomic uint64_t search_total;
  _Atomic uint64_t search_latency_us_sum;
  _Atomic uint64_t keepalive_reqs;
  _Atomic bool stop_flag;

  pthread_mutex_t lat_lock;
  uint64_t lat_samples[NBASE_HTTP_LAT_WINDOW];
  uint32_t lat_count;
  uint32_t lat_head;

  pthread_mutex_t q_lock;
  pthread_cond_t q_not_empty;
  pthread_cond_t q_not_full;
  int queue[NBASE_HTTP_QUEUE_CAP];
  uint32_t q_head;
  uint32_t q_tail;
  uint32_t q_count;
  bool stop;

  pthread_t accept_thread;
  bool accept_started;
  pthread_t *worker_threads;
  uint32_t workers_started;
};

typedef struct http_conn {
  nbase_http_server_t *srv;
  int fd;
  bool keep_alive;
} http_conn_t;

static char *dup_cstr(const char *s) {
  if (!s) {
    return NULL;
  }
  size_t n = strlen(s);
  char *out = malloc(n + 1U);
  if (!out) {
    return NULL;
  }
  memcpy(out, s, n + 1U);
  return out;
}

static bool send_all_sock(int fd, const char *buf, size_t len) {
  size_t off = 0U;
  while (off < len) {
    ssize_t n = send(fd, buf + off, len - off, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (n == 0) {
      return false;
    }
    off += (size_t)n;
  }
  return true;
}

static const char *http_status_reason(int code) {
  switch (code) {
    case 200:
      return "OK";
    case 201:
      return "Created";
    case 204:
      return "No Content";
    case 400:
      return "Bad Request";
    case 401:
      return "Unauthorized";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 409:
      return "Conflict";
    case 413:
      return "Payload Too Large";
    case 429:
      return "Too Many Requests";
    default:
      return "Internal Server Error";
  }
}

static int http_code_for_status(nbase_status_t st) {
  switch (st) {
    case NBASE_OK:
      return 200;
    case NBASE_ERR_ALREADY_EXISTS:
      return 409;
    case NBASE_ERR_NOT_FOUND:
      return 404;
    case NBASE_ERR_INVALID_ARG:
    case NBASE_ERR_DIM_MISMATCH:
    case NBASE_ERR_INVALID_VECTOR:
    case NBASE_ERR_UNSUPPORTED:
      return 400;
    default:
      return 500;
  }
}

static void send_http_response(
    const http_conn_t *c,
    int code,
    const char *content_type,
    const char *body,
    size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  if (srv != NULL && code >= 400) {
    atomic_fetch_add_explicit(&srv->err_total, 1ULL, memory_order_relaxed);
  }
  const char *conn_hdr = c->keep_alive
                             ? "Connection: keep-alive\r\nKeep-Alive: timeout=5, max=100\r\n"
                             : "Connection: close\r\n";
  char hdr[1280];
  int hlen;
  if (srv != NULL && srv->cors_origin != NULL) {
    hlen = snprintf(
        hdr,
        sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "X-Content-Type-Options: nosniff\r\n"
        "X-Frame-Options: DENY\r\n"
        "Access-Control-Allow-Origin: %s\r\n"
        "Access-Control-Allow-Methods: GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "%s\r\n",
        code,
        http_status_reason(code),
        content_type,
        body_len,
        srv->cors_origin,
        conn_hdr);
  } else {
    hlen = snprintf(
        hdr,
        sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "X-Content-Type-Options: nosniff\r\n"
        "X-Frame-Options: DENY\r\n"
        "%s\r\n",
        code,
        http_status_reason(code),
        content_type,
        body_len,
        conn_hdr);
  }
  if (hlen > 0 && (size_t)hlen < sizeof(hdr)) {
    if (send_all_sock(c->fd, hdr, (size_t)hlen) && body_len > 0U && body != NULL) {
      (void)send_all_sock(c->fd, body, body_len);
    }
  }
}

static void send_json_response(const http_conn_t *c, int code, const char *json_body, size_t body_len) {
  send_http_response(c, code, "application/json", json_body, body_len);
}

static void send_error_json(const http_conn_t *c, int code, const char *msg) {
  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_append(&sb, "{\"success\":false,\"error\":");
  nbase_sbuf_append_json_str(&sb, msg ? msg : "error");
  nbase_sbuf_append(&sb, "}");
  if (!sb.oom && sb.data != NULL) {
    send_json_response(c, code, sb.data, sb.len);
  }
  nbase_sbuf_free(&sb);
}

static void send_sbuf_json(const http_conn_t *c, int code, nbase_sbuf_t *sb) {
  if (sb->oom || !sb->data) {
    send_error_json(c, 500, "out of memory");
  } else {
    send_json_response(c, code, sb->data, sb->len);
  }
}

static bool const_time_bearer_eq(const char *auth_hdr, const char *expected_key) {
  if (!auth_hdr || !expected_key) {
    return false;
  }
  const char *p = auth_hdr;
  while (*p == ' ' || *p == '\t') {
    p++;
  }
  if (strncasecmp(p, "Bearer ", 7U) != 0) {
    return false;
  }
  p += 7U;
  while (*p == ' ' || *p == '\t') {
    p++;
  }
  size_t got_len = strlen(p);
  while (got_len > 0U && (p[got_len - 1U] == ' ' || p[got_len - 1U] == '\t' || p[got_len - 1U] == '\r' || p[got_len - 1U] == '\n')) {
    got_len--;
  }
  size_t exp_len = strlen(expected_key);
  size_t max_len = (got_len > exp_len) ? got_len : exp_len;
  uint32_t diff = (uint32_t)(got_len ^ exp_len);
  for (size_t i = 0U; i < max_len; i++) {
    unsigned char a = (i < got_len) ? (unsigned char)p[i] : 0U;
    unsigned char b = (i < exp_len) ? (unsigned char)expected_key[i] : 0U;
    diff |= (uint32_t)(a ^ b);
  }
  return diff == 0U;
}

static bool parse_u64_str(const char *s, size_t len, uint64_t *out) {
  if (!s || len == 0U || len > 20U) {
    return false;
  }
  uint64_t acc = 0ULL;
  for (size_t i = 0U; i < len; i++) {
    unsigned char ch = (unsigned char)s[i];
    if (ch < '0' || ch > '9') {
      return false;
    }
    uint64_t d = (uint64_t)(ch - '0');
    if (acc > (UINT64_MAX - d) / 10ULL) {
      return false;
    }
    acc = acc * 10ULL + d;
  }
  *out = acc;
  return true;
}

static bool parse_json_u64(const nbase_json_t *v, uint64_t *out) {
  if (!v) {
    return false;
  }
  if (v->type == NBASE_JSON_NUMBER && v->is_int && v->i64 >= 0) {
    *out = (uint64_t)v->i64;
    return true;
  }
  if (v->type == NBASE_JSON_STRING && v->str) {
    return parse_u64_str(v->str, strlen(v->str), out);
  }
  return false;
}

static bool parse_float_array(const nbase_json_t *arr, float **out_vec, uint32_t *out_dim) {
  if (!arr || arr->type != NBASE_JSON_ARRAY || arr->len == 0U || arr->len > NBASE_MAX_DIM) {
    return false;
  }
  uint32_t dim = (uint32_t)arr->len;
  float *vec = malloc((size_t)dim * sizeof(float));
  if (!vec) {
    return false;
  }
  for (uint32_t i = 0U; i < dim; i++) {
    const nbase_json_t *it = arr->items[i];
    if (!it || it->type != NBASE_JSON_NUMBER || !isfinite(it->num)) {
      free(vec);
      return false;
    }
    float f = (float)it->num;
    if (!isfinite(f)) {
      free(vec);
      return false;
    }
    vec[i] = f;
  }
  *out_vec = vec;
  *out_dim = dim;
  return true;
}

static bool is_dangerous_key(const char *k) {
  return strcmp(k, "__proto__") == 0 || strcmp(k, "constructor") == 0 || strcmp(k, "prototype") == 0;
}

static nbase_status_t parse_meta_fields_obj(
    const nbase_json_t *obj,
    nbase_meta_field_t **out_fields,
    uint32_t *out_count) {
  *out_fields = NULL;
  *out_count = 0U;
  if (!obj || obj->type != NBASE_JSON_OBJECT) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (obj->len == 0U) {
    return NBASE_OK;
  }
  if (obj->len > UINT32_MAX) {
    return NBASE_ERR_INVALID_ARG;
  }

  nbase_meta_field_t *fields = calloc(obj->len, sizeof(*fields));
  if (!fields) {
    return NBASE_ERR_NOMEM;
  }

  uint32_t n = 0U;
  for (size_t i = 0U; i < obj->len; i++) {
    const char *k = obj->keys[i];
    const nbase_json_t *val = obj->vals[i];
    if (!k || k[0] == '\0' || is_dangerous_key(k)) {
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
  *out_count = n;
  return NBASE_OK;
}

static bool parse_metric_str(const char *s, nbase_metric_t *out) {
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

static const char *metric_name_str(nbase_metric_t metric) {
  switch (metric) {
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

static bool parse_meta_op_str(const char *s, nbase_meta_op_t *out) {
  if (!s) {
    return false;
  }
  if (strcmp(s, "eq") == 0 || strcmp(s, "EQ") == 0) {
    *out = NBASE_OP_EQ;
    return true;
  }
  if (strcmp(s, "ne") == 0 || strcmp(s, "NE") == 0) {
    *out = NBASE_OP_NE;
    return true;
  }
  if (strcmp(s, "gt") == 0 || strcmp(s, "GT") == 0) {
    *out = NBASE_OP_GT;
    return true;
  }
  if (strcmp(s, "gte") == 0 || strcmp(s, "GTE") == 0) {
    *out = NBASE_OP_GTE;
    return true;
  }
  if (strcmp(s, "lt") == 0 || strcmp(s, "LT") == 0) {
    *out = NBASE_OP_LT;
    return true;
  }
  if (strcmp(s, "lte") == 0 || strcmp(s, "LTE") == 0) {
    *out = NBASE_OP_LTE;
    return true;
  }
  if (strcmp(s, "exists") == 0 || strcmp(s, "EXISTS") == 0) {
    *out = NBASE_OP_EXISTS;
    return true;
  }
  if (strcmp(s, "contains") == 0 || strcmp(s, "CONTAINS") == 0) {
    *out = NBASE_OP_CONTAINS;
    return true;
  }
  return false;
}

static bool fill_meta_value_from_json(const nbase_json_t *v, nbase_meta_value_t *out) {
  if (!v) {
    return false;
  }
  if (v->type == NBASE_JSON_BOOL) {
    out->type = NBASE_META_BOOL;
    out->as.b = v->b;
    return true;
  }
  if (v->type == NBASE_JSON_STRING) {
    out->type = NBASE_META_STRING;
    out->as.str = v->str;
    return true;
  }
  if (v->type == NBASE_JSON_NUMBER) {
    if (v->is_int) {
      out->type = NBASE_META_INT64;
      out->as.i64 = v->i64;
    } else {
      out->type = NBASE_META_DOUBLE;
      out->as.f64 = v->num;
    }
    return true;
  }
  return false;
}

static nbase_status_t parse_meta_filter_json(
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
      if (is_dangerous_key(flt->keys[i])) {
        continue;
      }
      if (!fill_meta_value_from_json(flt->vals[i], &conds[n].value)) {
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
      const nbase_json_t *jk = nbase_json_obj_get(c, "key");
      const nbase_json_t *jop = nbase_json_obj_get(c, "op");
      const nbase_json_t *jval = nbase_json_obj_get(c, "value");
      if (!jk || jk->type != NBASE_JSON_STRING || jk->str[0] == '\0' || !jop || jop->type != NBASE_JSON_STRING) {
        free(conds);
        return NBASE_ERR_INVALID_ARG;
      }
      nbase_meta_op_t op = NBASE_OP_EQ;
      if (!parse_meta_op_str(jop->str, &op)) {
        free(conds);
        return NBASE_ERR_INVALID_ARG;
      }
      conds[n].key = jk->str;
      conds[n].op = op;
      if (op == NBASE_OP_EXISTS) {
        conds[n].value.type = NBASE_META_BOOL;
        conds[n].value.as.b = (jval && jval->type == NBASE_JSON_BOOL) ? jval->b : true;
      } else if (!fill_meta_value_from_json(jval, &conds[n].value)) {
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

static uint64_t monotonic_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)(ts.tv_nsec / 1000L);
}

static bool query_param_get(const char *qs, const char *key, char *out, size_t out_cap) {
  if (!qs || !key || !out || out_cap == 0U) {
    return false;
  }
  size_t klen = strlen(key);
  const char *p = qs;
  while (*p != '\0') {
    const char *amp = strchr(p, '&');
    size_t tok_len = amp ? (size_t)(amp - p) : strlen(p);
    if (tok_len > klen && strncmp(p, key, klen) == 0 && p[klen] == '=') {
      size_t vlen = tok_len - klen - 1U;
      if (vlen >= out_cap) {
        vlen = out_cap - 1U;
      }
      if (vlen > 0U) {
        memcpy(out, p + klen + 1U, vlen);
      }
      out[vlen] = '\0';
      return true;
    }
    if (!amp) {
      break;
    }
    p = amp + 1;
  }
  return false;
}

static bool query_param_bool(const char *qs, const char *key, bool default_val) {
  char buf[16];
  if (!query_param_get(qs, key, buf, sizeof(buf))) {
    return default_val;
  }
  if (strcasecmp(buf, "false") == 0 || strcmp(buf, "0") == 0) {
    return false;
  }
  if (strcasecmp(buf, "true") == 0 || strcmp(buf, "1") == 0) {
    return true;
  }
  return default_val;
}

typedef struct meta_json_ctx {
  nbase_sbuf_t *sb;
  bool first;
} meta_json_ctx_t;

static bool meta_json_cb(const char *key, const nbase_meta_value_t *val, void *raw) {
  meta_json_ctx_t *ctx = (meta_json_ctx_t *)raw;
  if (!ctx->first) {
    nbase_sbuf_append_n(ctx->sb, ",", 1U);
  }
  ctx->first = false;
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

static void append_meta_json(nbase_sbuf_t *sb, const nbase_db_t *db, uint64_t id) {
  nbase_sbuf_append_n(sb, "{", 1U);
  meta_json_ctx_t ctx = {sb, true};
  (void)nbase_foreach_meta(db, id, meta_json_cb, &ctx);
  nbase_sbuf_append_n(sb, "}", 1U);
}

static void append_vector_json(nbase_sbuf_t *sb, const nbase_db_t *db, uint64_t id, uint32_t dim) {
  float stack_buf[128];
  float *vec = (dim <= 128U) ? stack_buf : malloc((size_t)dim * sizeof(float));
  if (!vec) {
    sb->oom = true;
    return;
  }
  if (nbase_get(db, id, vec, dim) != NBASE_OK) {
    if (vec != stack_buf) {
      free(vec);
    }
    nbase_sbuf_append(sb, "null");
    return;
  }
  nbase_sbuf_append_n(sb, "[", 1U);
  for (uint32_t d = 0U; d < dim; d++) {
    nbase_sbuf_appendf(sb, d == 0U ? "%.7g" : ",%.7g", (double)vec[d]);
  }
  nbase_sbuf_append_n(sb, "]", 1U);
  if (vec != stack_buf) {
    free(vec);
  }
}

static void append_text_json(nbase_sbuf_t *sb, const nbase_db_t *db, uint64_t id) {
  size_t tlen = 0U;
  if (nbase_get_text(db, id, NULL, 0U, &tlen) != NBASE_OK) {
    nbase_sbuf_append(sb, "null");
    return;
  }
  char stack_buf[512];
  char *buf = (tlen < sizeof(stack_buf)) ? stack_buf : malloc(tlen + 1U);
  size_t cap = (tlen < sizeof(stack_buf)) ? sizeof(stack_buf) : (tlen + 1U);
  if (!buf) {
    sb->oom = true;
    return;
  }
  if (nbase_get_text(db, id, buf, cap, &tlen) != NBASE_OK) {
    nbase_sbuf_append(sb, "null");
  } else {
    nbase_sbuf_append_json_str(sb, buf);
  }
  if (buf != stack_buf) {
    free(buf);
  }
}

typedef struct http_batch_enrich {
  float *vecs;
  bool *vfound;
  char **texts;
  bool *tfound;
  uint32_t count;
  uint32_t dim;
} http_batch_enrich_t;

static void http_batch_enrich_free(http_batch_enrich_t *e) {
  nbase_free_texts(e->texts, e->count);
  free(e->texts);
  free(e->tfound);
  free(e->vecs);
  free(e->vfound);
  memset(e, 0, sizeof(*e));
}

static bool http_batch_enrich_load(
    const nbase_db_t *db,
    const uint64_t *ids,
    uint32_t count,
    uint32_t dim,
    bool inc_vec,
    bool inc_text,
    http_batch_enrich_t *out) {
  memset(out, 0, sizeof(*out));
  out->count = count;
  out->dim = dim;
  if (count == 0U || (!inc_vec && !inc_text)) {
    return true;
  }
  if (inc_vec) {
    out->vecs = calloc((size_t)count * (size_t)dim, sizeof(float));
    out->vfound = calloc(count, sizeof(bool));
    if (!out->vecs || !out->vfound ||
        nbase_get_batch(db, ids, count, dim, out->vecs, out->vfound) != NBASE_OK) {
      http_batch_enrich_free(out);
      return false;
    }
  }
  if (inc_text) {
    out->texts = calloc(count, sizeof(char *));
    out->tfound = calloc(count, sizeof(bool));
    if (!out->texts || !out->tfound ||
        nbase_get_text_batch(db, ids, count, out->texts, out->tfound) != NBASE_OK) {
      http_batch_enrich_free(out);
      return false;
    }
  }
  return true;
}

static void append_enriched_fields_json(
    nbase_sbuf_t *sb,
    const nbase_db_t *db,
    uint64_t id,
    uint32_t idx,
    bool inc_meta,
    bool inc_vec,
    bool inc_text,
    bool vec_null_on_missing,
    const http_batch_enrich_t *e) {
  if (inc_meta) {
    nbase_sbuf_append(sb, ",\"metadata\":");
    append_meta_json(sb, db, id);
  }
  if (inc_vec) {
    if (e != NULL && e->vfound != NULL && idx < e->count && e->vfound[idx]) {
      const float *v = e->vecs + (size_t)idx * (size_t)e->dim;
      nbase_sbuf_append(sb, ",\"vector\":[");
      for (uint32_t d = 0U; d < e->dim; d++) {
        nbase_sbuf_appendf(sb, d == 0U ? "%.7g" : ",%.7g", (double)v[d]);
      }
      nbase_sbuf_append_n(sb, "]", 1U);
    } else if (vec_null_on_missing) {
      nbase_sbuf_append(sb, ",\"vector\":null");
    }
  }
  if (inc_text) {
    if (e != NULL && e->tfound != NULL && idx < e->count && e->tfound[idx] && e->texts[idx] != NULL) {
      nbase_sbuf_append(sb, ",\"text\":");
      nbase_sbuf_append_json_str(sb, e->texts[idx]);
    } else {
      nbase_sbuf_append(sb, ",\"text\":null");
    }
  }
}

static void handle_health(const http_conn_t *c) {
  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_append(&sb, "{\"status\":\"ok\",\"version\":");
  nbase_sbuf_append_json_str(&sb, nbase_version());
  nbase_sbuf_append(&sb, ",\"simd\":");
  nbase_sbuf_append_json_str(&sb, nbase_simd_backend());
  nbase_sbuf_append(&sb, "}");
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static int cmp_http_u64_asc(const void *pa, const void *pb) {
  uint64_t a = *(const uint64_t *)pa;
  uint64_t b = *(const uint64_t *)pb;
  if (a < b) {
    return -1;
  }
  if (a > b) {
    return 1;
  }
  return 0;
}

static void record_search_latency(nbase_http_server_t *srv, uint64_t qcount, uint64_t delta_us) {
  atomic_fetch_add_explicit(&srv->search_total, qcount, memory_order_relaxed);
  atomic_fetch_add_explicit(&srv->search_latency_us_sum, delta_us, memory_order_relaxed);
  pthread_mutex_lock(&srv->lat_lock);
  srv->lat_samples[srv->lat_head] = delta_us;
  srv->lat_head = (srv->lat_head + 1U) % NBASE_HTTP_LAT_WINDOW;
  if (srv->lat_count < NBASE_HTTP_LAT_WINDOW) {
    srv->lat_count++;
  }
  pthread_mutex_unlock(&srv->lat_lock);
}

static void compute_latency_percentiles(
    nbase_http_server_t *srv,
    uint64_t *p50,
    uint64_t *p95,
    uint64_t *p99) {
  uint64_t tmp[NBASE_HTTP_LAT_WINDOW];
  pthread_mutex_lock(&srv->lat_lock);
  uint32_t cnt = srv->lat_count;
  if (cnt > 0U) {
    memcpy(tmp, srv->lat_samples, (size_t)cnt * sizeof(uint64_t));
  }
  pthread_mutex_unlock(&srv->lat_lock);

  if (cnt == 0U) {
    *p50 = 0ULL;
    *p95 = 0ULL;
    *p99 = 0ULL;
    return;
  }
  qsort(tmp, cnt, sizeof(uint64_t), cmp_http_u64_asc);
  uint32_t i50 = ((cnt * 50U + 99U) / 100U) - 1U;
  uint32_t i95 = ((cnt * 95U + 99U) / 100U) - 1U;
  uint32_t i99 = ((cnt * 99U + 99U) / 100U) - 1U;
  *p50 = tmp[i50];
  *p95 = tmp[i95];
  *p99 = tmp[i99];
}

static void handle_metrics(const http_conn_t *c) {
  nbase_http_server_t *srv = c->srv;
  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    memset(&db_st, 0, sizeof(db_st));
  }
  uint64_t reqs = atomic_load_explicit(&srv->req_total, memory_order_relaxed);
  uint64_t errs = atomic_load_explicit(&srv->err_total, memory_order_relaxed);
  uint64_t searches = atomic_load_explicit(&srv->search_total, memory_order_relaxed);
  uint64_t search_us = atomic_load_explicit(&srv->search_latency_us_sum, memory_order_relaxed);
  uint64_t ka_reqs = atomic_load_explicit(&srv->keepalive_reqs, memory_order_relaxed);
  uint64_t p50 = 0ULL;
  uint64_t p95 = 0ULL;
  uint64_t p99 = 0ULL;
  compute_latency_percentiles(srv, &p50, &p95, &p99);

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(
      &sb,
      "# HELP nbase_requests_total Total HTTP requests handled\n"
      "# TYPE nbase_requests_total counter\n"
      "nbase_requests_total %llu\n"
      "# HELP nbase_errors_total Total HTTP error responses\n"
      "# TYPE nbase_errors_total counter\n"
      "nbase_errors_total %llu\n"
      "# HELP nbase_search_queries_total Total search queries executed\n"
      "# TYPE nbase_search_queries_total counter\n"
      "nbase_search_queries_total %llu\n"
      "# HELP nbase_search_latency_microseconds_sum Cumulative search latency in microseconds\n"
      "# TYPE nbase_search_latency_microseconds_sum counter\n"
      "nbase_search_latency_microseconds_sum %llu\n"
      "# HELP nbase_search_latency_p50_microseconds Rolling P50 search latency in microseconds\n"
      "# TYPE nbase_search_latency_p50_microseconds gauge\n"
      "nbase_search_latency_p50_microseconds %llu\n"
      "# HELP nbase_search_latency_p95_microseconds Rolling P95 search latency in microseconds\n"
      "# TYPE nbase_search_latency_p95_microseconds gauge\n"
      "nbase_search_latency_p95_microseconds %llu\n"
      "# HELP nbase_search_latency_p99_microseconds Rolling P99 search latency in microseconds\n"
      "# TYPE nbase_search_latency_p99_microseconds gauge\n"
      "nbase_search_latency_p99_microseconds %llu\n"
      "# HELP nbase_keepalive_requests_total Requests served over reused keep-alive connections\n"
      "# TYPE nbase_keepalive_requests_total counter\n"
      "nbase_keepalive_requests_total %llu\n"
      "# HELP nbase_vectors_active Active stored vectors\n"
      "# TYPE nbase_vectors_active gauge\n"
      "nbase_vectors_active %llu\n"
      "# HELP nbase_vectors_tombstones Deleted vector tombstones awaiting compaction\n"
      "# TYPE nbase_vectors_tombstones gauge\n"
      "nbase_vectors_tombstones %llu\n"
      "# HELP nbase_hnsw_nodes_active Active HNSW graph nodes\n"
      "# TYPE nbase_hnsw_nodes_active gauge\n"
      "nbase_hnsw_nodes_active %llu\n"
      "# HELP nbase_text_docs_active Active indexed BM25 text documents\n"
      "# TYPE nbase_text_docs_active gauge\n"
      "nbase_text_docs_active %llu\n"
      "# HELP nbase_memory_bytes Estimated in-memory database footprint in bytes\n"
      "# TYPE nbase_memory_bytes gauge\n"
      "nbase_memory_bytes %llu\n"
      "# HELP nbase_cache_entries Number of cached search query entries\n"
      "# TYPE nbase_cache_entries gauge\n"
      "nbase_cache_entries %u\n"
      "# HELP nbase_cache_hits_total Cumulative query cache hits\n"
      "# TYPE nbase_cache_hits_total counter\n"
      "nbase_cache_hits_total %llu\n"
      "# HELP nbase_cache_misses_total Cumulative query cache misses\n"
      "# TYPE nbase_cache_misses_total counter\n"
      "nbase_cache_misses_total %llu\n",
      (unsigned long long)reqs,
      (unsigned long long)errs,
      (unsigned long long)searches,
      (unsigned long long)search_us,
      (unsigned long long)p50,
      (unsigned long long)p95,
      (unsigned long long)p99,
      (unsigned long long)ka_reqs,
      (unsigned long long)db_st.active_vectors,
      (unsigned long long)db_st.tombstones,
      (unsigned long long)db_st.hnsw_active_nodes,
      (unsigned long long)db_st.text_docs,
      (unsigned long long)db_st.memory_bytes,
      db_st.cache_entries,
      (unsigned long long)db_st.cache_hits,
      (unsigned long long)db_st.cache_misses);
  if (sb.oom || !sb.data) {
    send_error_json(c, 500, "out of memory");
  } else {
    send_http_response(c, 200, "text/plain; version=0.0.4; charset=utf-8", sb.data, sb.len);
  }
  nbase_sbuf_free(&sb);
}

static void handle_stats(const http_conn_t *c) {
  nbase_stats_t st_val;
  nbase_status_t st = nbase_stats(c->srv->db, &st_val);
  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  uint64_t p50 = 0ULL;
  uint64_t p95 = 0ULL;
  uint64_t p99 = 0ULL;
  compute_latency_percentiles(c->srv, &p50, &p95, &p99);

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(
      &sb,
      "{\"success\":true,\"dim\":%u,\"metric\":\"%s\","
      "\"active_vectors\":%llu,\"total_slots\":%llu,\"tombstones\":%llu,"
      "\"mmap_backed\":%s,\"hnsw_enabled\":%s,\"hnsw_max_level\":%d,"
      "\"hnsw_active_nodes\":%llu,\"hnsw_deleted_nodes\":%llu,\"text_docs\":%llu,"
      "\"memory_bytes\":%llu,\"cache_capacity\":%u,\"cache_entries\":%u,"
      "\"cache_hits\":%llu,\"cache_misses\":%llu,"
      "\"search_p50_us\":%llu,\"search_p95_us\":%llu,\"search_p99_us\":%llu}",
      st_val.dim,
      metric_name_str(st_val.metric),
      (unsigned long long)st_val.active_vectors,
      (unsigned long long)st_val.total_slots,
      (unsigned long long)st_val.tombstones,
      st_val.mmap_backed ? "true" : "false",
      st_val.hnsw_enabled ? "true" : "false",
      st_val.hnsw_max_level,
      (unsigned long long)st_val.hnsw_active_nodes,
      (unsigned long long)st_val.hnsw_deleted_nodes,
      (unsigned long long)st_val.text_docs,
      (unsigned long long)st_val.memory_bytes,
      st_val.cache_capacity,
      st_val.cache_entries,
      (unsigned long long)st_val.cache_hits,
      (unsigned long long)st_val.cache_misses,
      (unsigned long long)p50,
      (unsigned long long)p95,
      (unsigned long long)p99);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static nbase_status_t apply_json_items_meta_and_text(
    nbase_db_t *db,
    const uint64_t *ids,
    nbase_json_t **items,
    uint32_t count) {
  if (count == 0U) {
    return NBASE_OK;
  }
  nbase_meta_batch_item_t *meta_items = calloc(count, sizeof(*meta_items));
  nbase_meta_field_t **field_bufs = calloc(count, sizeof(*field_bufs));
  uint64_t *text_ids = malloc((size_t)count * sizeof(uint64_t));
  const char **text_ptrs = malloc((size_t)count * sizeof(const char *));
  if (!meta_items || !field_bufs || !text_ids || !text_ptrs) {
    free(meta_items);
    free(field_bufs);
    free(text_ids);
    free((void *)text_ptrs);
    return NBASE_ERR_NOMEM;
  }

  nbase_status_t st = NBASE_OK;
  uint32_t meta_cnt = 0U;
  uint32_t text_cnt = 0U;
  for (uint32_t i = 0U; i < count && st == NBASE_OK; i++) {
    const nbase_json_t *jmeta = nbase_json_obj_get(items[i], "metadata");
    if (jmeta != NULL && jmeta->type != NBASE_JSON_NULL) {
      nbase_meta_field_t *fields = NULL;
      uint32_t fcnt = 0U;
      st = parse_meta_fields_obj(jmeta, &fields, &fcnt);
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
    const nbase_json_t *jtxt = nbase_json_obj_get(items[i], "text");
    if (st == NBASE_OK && jtxt != NULL && jtxt->type == NBASE_JSON_STRING) {
      text_ids[text_cnt] = ids[i];
      text_ptrs[text_cnt] = jtxt->str;
      text_cnt++;
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
  return st;
}

static void handle_write_vectors(
    const http_conn_t *c,
    const char *body,
    size_t body_len,
    bool is_upsert) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jbulk = nbase_json_obj_get(root, "vectors");
  if (jbulk != NULL) {
    if (jbulk->type != NBASE_JSON_ARRAY || jbulk->len == 0U || jbulk->len > 100000U) {
      nbase_json_free(root);
      send_error_json(c, 400, "vectors must be a non-empty array");
      return;
    }

    nbase_stats_t db_st;
    if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
      nbase_json_free(root);
      send_error_json(c, 500, "failed to read db stats");
      return;
    }

    uint32_t threads = 0U;
    const nbase_json_t *jthr = nbase_json_obj_get(root, "threads");
    if (jthr != NULL) {
      if (jthr->type != NBASE_JSON_NUMBER || !jthr->is_int || jthr->i64 < 0 || jthr->i64 > 64) {
        nbase_json_free(root);
        send_error_json(c, 400, "invalid threads parameter");
        return;
      }
      threads = (uint32_t)jthr->i64;
    }

    uint32_t count = (uint32_t)jbulk->len;
    uint32_t dim = db_st.dim;
    uint64_t *ids = malloc((size_t)count * sizeof(uint64_t));
    float *vecs = malloc((size_t)count * (size_t)dim * sizeof(float));
    if (!ids || !vecs) {
      free(ids);
      free(vecs);
      nbase_json_free(root);
      send_error_json(c, 500, "out of memory");
      return;
    }

    for (uint32_t i = 0U; i < count; i++) {
      const nbase_json_t *item = jbulk->items[i];
      if (!item || item->type != NBASE_JSON_OBJECT) {
        free(ids);
        free(vecs);
        nbase_json_free(root);
        send_error_json(c, 400, "each bulk item must be an object");
        return;
      }
      if (!parse_json_u64(nbase_json_obj_get(item, "id"), &ids[i])) {
        free(ids);
        free(vecs);
        nbase_json_free(root);
        send_error_json(c, 400, "each item requires unsigned 64-bit integer id");
        return;
      }
      const nbase_json_t *jtxt = nbase_json_obj_get(item, "text");
      if (jtxt != NULL && jtxt->type != NBASE_JSON_NULL && jtxt->type != NBASE_JSON_STRING) {
        free(ids);
        free(vecs);
        nbase_json_free(root);
        send_error_json(c, 400, "text must be a string");
        return;
      }
      float *tmp_v = NULL;
      uint32_t item_dim = 0U;
      if (!parse_float_array(nbase_json_obj_get(item, "vector"), &tmp_v, &item_dim) || item_dim != dim) {
        free(tmp_v);
        free(ids);
        free(vecs);
        nbase_json_free(root);
        send_error_json(c, 400, "invalid vector or dimension mismatch");
        return;
      }
      memcpy(vecs + (size_t)i * (size_t)dim, tmp_v, (size_t)dim * sizeof(float));
      free(tmp_v);
    }

    nbase_status_t st = is_upsert
                            ? nbase_upsert_batch(srv->db, ids, vecs, count, dim, threads)
                            : nbase_add_batch(srv->db, ids, vecs, count, dim, threads);
    free(vecs);
    if (st == NBASE_OK) {
      st = apply_json_items_meta_and_text(srv->db, ids, jbulk->items, count);
    }
    if (st != NBASE_OK) {
      free(ids);
      nbase_json_free(root);
      send_error_json(c, http_code_for_status(st), nbase_strerror(st));
      return;
    }

    free(ids);
    nbase_json_free(root);
    nbase_sbuf_t sb;
    nbase_sbuf_init(&sb);
    nbase_sbuf_appendf(
        &sb,
        is_upsert ? "{\"success\":true,\"upserted\":%u}" : "{\"success\":true,\"added\":%u}",
        count);
    send_sbuf_json(c, is_upsert ? 200 : 201, &sb);
    nbase_sbuf_free(&sb);
    return;
  }

  uint64_t id = 0ULL;
  if (!parse_json_u64(nbase_json_obj_get(root, "id"), &id)) {
    nbase_json_free(root);
    send_error_json(c, 400, "missing or invalid unsigned 64-bit id");
    return;
  }
  const nbase_json_t *jtxt = nbase_json_obj_get(root, "text");
  if (jtxt != NULL && jtxt->type != NBASE_JSON_NULL && jtxt->type != NBASE_JSON_STRING) {
    nbase_json_free(root);
    send_error_json(c, 400, "text must be a string");
    return;
  }
  float *vec = NULL;
  uint32_t dim = 0U;
  if (!parse_float_array(nbase_json_obj_get(root, "vector"), &vec, &dim)) {
    nbase_json_free(root);
    send_error_json(c, 400, "missing or invalid vector array");
    return;
  }

  nbase_status_t st = is_upsert ? nbase_upsert(srv->db, id, vec, dim) : nbase_add(srv->db, id, vec, dim);
  free(vec);
  if (st != NBASE_OK) {
    nbase_json_free(root);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  const nbase_json_t *jmeta = nbase_json_obj_get(root, "metadata");
  if (jmeta != NULL && jmeta->type != NBASE_JSON_NULL) {
    nbase_meta_field_t *fields = NULL;
    uint32_t fcnt = 0U;
    st = parse_meta_fields_obj(jmeta, &fields, &fcnt);
    if (st == NBASE_OK && fcnt > 0U) {
      st = nbase_set_meta(srv->db, id, fields, fcnt);
    }
    free(fields);
    if (st != NBASE_OK) {
      nbase_json_free(root);
      send_error_json(c, http_code_for_status(st), nbase_strerror(st));
      return;
    }
  }

  if (jtxt != NULL && jtxt->type == NBASE_JSON_STRING) {
    st = nbase_set_text(srv->db, id, jtxt->str);
    if (st != NBASE_OK) {
      nbase_json_free(root);
      send_error_json(c, http_code_for_status(st), nbase_strerror(st));
      return;
    }
  }

  nbase_json_free(root);
  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  if (is_upsert) {
    nbase_sbuf_appendf(&sb, "{\"success\":true,\"id\":%llu,\"upserted\":1}", (unsigned long long)id);
  } else {
    nbase_sbuf_appendf(&sb, "{\"success\":true,\"id\":%llu}", (unsigned long long)id);
  }
  send_sbuf_json(c, is_upsert ? 200 : 201, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_post_vectors(const http_conn_t *c, const char *body, size_t body_len) {
  handle_write_vectors(c, body, body_len, false);
}

static void handle_upsert_vectors(const http_conn_t *c, const char *body, size_t body_len) {
  handle_write_vectors(c, body, body_len, true);
}

static bool parse_include_flags(
    const http_conn_t *c,
    const nbase_json_t *root,
    bool *inc_meta,
    bool *inc_vec,
    bool *inc_text);

static void handle_get_vectors_batch(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jids = nbase_json_obj_get(root, "ids");
  if (!jids || jids->type != NBASE_JSON_ARRAY || jids->len == 0U || jids->len > 100000U) {
    nbase_json_free(root);
    send_error_json(c, 400, "ids must be a non-empty array of at most 100000 uint64 values");
    return;
  }

  bool inc_vec = true;
  bool inc_meta = true;
  bool inc_text = false;
  if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
    nbase_json_free(root);
    return;
  }

  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    nbase_json_free(root);
    send_error_json(c, 500, "failed to read db stats");
    return;
  }

  uint32_t cnt = (uint32_t)jids->len;
  uint32_t dim = db_st.dim;
  uint64_t *ids = malloc((size_t)cnt * sizeof(uint64_t));
  float *vecs = malloc((size_t)cnt * (size_t)dim * sizeof(float));
  bool *found = malloc((size_t)cnt * sizeof(bool));
  if (!ids || !vecs || !found) {
    free(ids);
    free(vecs);
    free(found);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  for (uint32_t i = 0U; i < cnt; i++) {
    if (!parse_json_u64(jids->items[i], &ids[i])) {
      free(ids);
      free(vecs);
      free(found);
      nbase_json_free(root);
      send_error_json(c, 400, "each id must be an unsigned 64-bit integer");
      return;
    }
  }
  nbase_json_free(root);

  nbase_status_t st = nbase_get_batch(srv->db, ids, cnt, dim, vecs, found);
  if (st != NBASE_OK) {
    free(ids);
    free(vecs);
    free(found);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  uint32_t found_cnt = 0U;
  for (uint32_t i = 0U; i < cnt; i++) {
    if (found[i]) {
      found_cnt++;
    }
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(
      &sb,
      "{\"success\":true,\"count\":%u,\"foundCount\":%u,\"results\":[",
      cnt,
      found_cnt);
  for (uint32_t i = 0U; i < cnt; i++) {
    nbase_sbuf_appendf(
        &sb,
        i == 0U ? "{\"id\":%llu,\"found\":%s" : ",{\"id\":%llu,\"found\":%s",
        (unsigned long long)ids[i],
        found[i] ? "true" : "false");
    if (found[i]) {
      if (inc_vec) {
        nbase_sbuf_append(&sb, ",\"vector\":[");
        const float *v = vecs + (size_t)i * (size_t)dim;
        for (uint32_t d = 0U; d < dim; d++) {
          nbase_sbuf_appendf(&sb, d == 0U ? "%.7g" : ",%.7g", (double)v[d]);
        }
        nbase_sbuf_append_n(&sb, "]", 1U);
      }
      if (inc_meta) {
        nbase_sbuf_append(&sb, ",\"metadata\":");
        append_meta_json(&sb, srv->db, ids[i]);
      }
      if (inc_text) {
        nbase_sbuf_append(&sb, ",\"text\":");
        append_text_json(&sb, srv->db, ids[i]);
      }
    }
    nbase_sbuf_append_n(&sb, "}", 1U);
  }
  nbase_sbuf_append(&sb, "]}");

  free(ids);
  free(vecs);
  free(found);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_vectors_exists(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jids = nbase_json_obj_get(root, "ids");
  if (!jids || jids->type != NBASE_JSON_ARRAY || jids->len > 100000U) {
    nbase_json_free(root);
    send_error_json(c, 400, "ids must be an array of at most 100000 uint64 values");
    return;
  }

  uint32_t cnt = (uint32_t)jids->len;
  uint64_t *ids = (cnt > 0U) ? malloc((size_t)cnt * sizeof(uint64_t)) : NULL;
  if (cnt > 0U && !ids) {
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  for (uint32_t i = 0U; i < cnt; i++) {
    if (!parse_json_u64(jids->items[i], &ids[i])) {
      free(ids);
      nbase_json_free(root);
      send_error_json(c, 400, "each id must be an unsigned 64-bit integer");
      return;
    }
  }
  nbase_json_free(root);

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"count\":%u,\"results\":[", cnt);
  uint32_t existing_cnt = 0U;
  for (uint32_t i = 0U; i < cnt; i++) {
    bool ex = nbase_contains(srv->db, ids[i]);
    if (ex) {
      existing_cnt++;
    }
    nbase_sbuf_appendf(
        &sb,
        i == 0U ? "{\"id\":%llu,\"exists\":%s}" : ",{\"id\":%llu,\"exists\":%s}",
        (unsigned long long)ids[i],
        ex ? "true" : "false");
  }
  nbase_sbuf_appendf(&sb, "],\"existingCount\":%u}", existing_cnt);
  free(ids);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static nbase_status_t append_ndjson_batch(
    nbase_sbuf_t *sb,
    nbase_db_t *db,
    const uint64_t *ids,
    uint32_t got,
    uint32_t dim,
    bool inc_vec,
    bool inc_meta,
    bool inc_text,
    float *vecs,
    bool *found) {
  if (got == 0U) {
    return NBASE_OK;
  }
  if (inc_vec) {
    nbase_status_t st = nbase_get_batch(db, ids, got, dim, vecs, found);
    if (st != NBASE_OK) {
      return st;
    }
  }
  for (uint32_t i = 0U; i < got; i++) {
    nbase_sbuf_appendf(sb, "{\"id\":%llu", (unsigned long long)ids[i]);
    if (inc_vec && found[i]) {
      nbase_sbuf_append(sb, ",\"vector\":[");
      const float *v = vecs + (size_t)i * (size_t)dim;
      for (uint32_t d = 0U; d < dim; d++) {
        nbase_sbuf_appendf(sb, d == 0U ? "%.7g" : ",%.7g", (double)v[d]);
      }
      nbase_sbuf_append_n(sb, "]", 1U);
    }
    if (inc_meta) {
      nbase_sbuf_append(sb, ",\"metadata\":");
      append_meta_json(sb, db, ids[i]);
    }
    if (inc_text) {
      size_t tlen = 0U;
      if (nbase_get_text(db, ids[i], NULL, 0U, &tlen) == NBASE_OK) {
        nbase_sbuf_append(sb, ",\"text\":");
        append_text_json(sb, db, ids[i]);
      }
    }
    nbase_sbuf_append(sb, "}\n");
  }
  return sb->oom ? NBASE_ERR_NOMEM : NBASE_OK;
}

static void export_unfiltered_ndjson(
    const http_conn_t *c,
    uint32_t dim,
    bool inc_vec,
    bool inc_meta,
    bool inc_text) {
  nbase_http_server_t *srv = c->srv;
  const uint32_t page_cap = 1024U;
  uint64_t *ids = malloc((size_t)page_cap * sizeof(uint64_t));
  float *vecs = inc_vec ? malloc((size_t)page_cap * (size_t)dim * sizeof(float)) : NULL;
  bool *found = inc_vec ? malloc((size_t)page_cap * sizeof(bool)) : NULL;
  if (!ids || (inc_vec && (!vecs || !found))) {
    free(ids);
    free(vecs);
    free(found);
    send_error_json(c, 500, "out of memory");
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_status_t st = NBASE_OK;
  uint64_t offset = 0ULL;
  uint64_t total = 0ULL;
  while (true) {
    uint32_t got = 0U;
    st = nbase_list_ids(srv->db, offset, page_cap, ids, &got, &total);
    if (st != NBASE_OK || got == 0U) {
      break;
    }
    st = append_ndjson_batch(&sb, srv->db, ids, got, dim, inc_vec, inc_meta, inc_text, vecs, found);
    if (st != NBASE_OK) {
      break;
    }
    offset += (uint64_t)got;
    if (offset >= total) {
      break;
    }
  }

  free(ids);
  free(vecs);
  free(found);
  if (st != NBASE_OK || sb.oom) {
    nbase_sbuf_free(&sb);
    send_error_json(
        c,
        http_code_for_status(st != NBASE_OK ? st : NBASE_ERR_NOMEM),
        nbase_strerror(st != NBASE_OK ? st : NBASE_ERR_NOMEM));
    return;
  }
  send_http_response(c, 200, "application/x-ndjson", sb.data ? sb.data : "", sb.len);
  nbase_sbuf_free(&sb);
}

static bool parse_include_flags(
    const http_conn_t *c,
    const nbase_json_t *root,
    bool *inc_meta,
    bool *inc_vec,
    bool *inc_text) {
  const nbase_json_t *jincm = nbase_json_obj_get(root, "includeMetadata");
  if (jincm != NULL) {
    if (jincm->type != NBASE_JSON_BOOL) {
      send_error_json(c, 400, "includeMetadata must be boolean");
      return false;
    }
    *inc_meta = jincm->b;
  }
  const nbase_json_t *jincv = nbase_json_obj_get(root, "includeVectors");
  if (jincv != NULL) {
    if (jincv->type != NBASE_JSON_BOOL) {
      send_error_json(c, 400, "includeVectors must be boolean");
      return false;
    }
    *inc_vec = jincv->b;
  }
  const nbase_json_t *jinct = nbase_json_obj_get(root, "includeText");
  if (jinct != NULL) {
    if (jinct->type != NBASE_JSON_BOOL) {
      send_error_json(c, 400, "includeText must be boolean");
      return false;
    }
    *inc_text = jinct->b;
  }
  return true;
}

static void handle_export_vectors(const http_conn_t *c, const char *query_str) {
  nbase_http_server_t *srv = c->srv;
  bool inc_vec = query_param_bool(query_str, "includeVectors", true);
  bool inc_meta = query_param_bool(query_str, "includeMetadata", true);
  bool inc_text = query_param_bool(query_str, "includeText", true);

  nbase_stats_t db_st;
  nbase_status_t st = nbase_stats(srv->db, &db_st);
  if (st != NBASE_OK) {
    send_error_json(c, 500, nbase_strerror(st));
    return;
  }
  if (db_st.active_vectors == 0ULL) {
    send_http_response(c, 200, "application/x-ndjson", "", 0U);
    return;
  }
  export_unfiltered_ndjson(c, db_st.dim, inc_vec, inc_meta, inc_text);
}

static int cmp_u64_asc_http(const void *pa, const void *pb) {
  uint64_t a = *(const uint64_t *)pa;
  uint64_t b = *(const uint64_t *)pb;
  return (a > b) - (a < b);
}

static bool u64_in_sorted(const uint64_t *arr, uint32_t len, uint64_t target) {
  if (!arr || len == 0U) {
    return false;
  }
  return bsearch(&target, arr, len, sizeof(uint64_t), cmp_u64_asc_http) != NULL;
}

static void handle_export_vectors_post(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  bool inc_vec = true;
  bool inc_meta = true;
  bool inc_text = true;
  nbase_json_t *root = NULL;
  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt = {NULL, 0U};
  bool has_filter = false;
  bool has_ids = false;
  uint64_t *req_ids = NULL;
  uint32_t req_cnt = 0U;

  if (body_len > 0U && body != NULL) {
    if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
      nbase_json_free(root);
      send_error_json(c, 400, "invalid JSON request body");
      return;
    }
    if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
      nbase_json_free(root);
      return;
    }
    const nbase_json_t *jids = nbase_json_obj_get(root, "ids");
    if (jids != NULL && jids->type != NBASE_JSON_NULL) {
      if (jids->type != NBASE_JSON_ARRAY || jids->len > 100000U) {
        nbase_json_free(root);
        send_error_json(c, 400, "ids must be an array of at most 100000 uint64 values");
        return;
      }
      has_ids = true;
      req_cnt = (uint32_t)jids->len;
      if (req_cnt > 0U) {
        req_ids = malloc((size_t)req_cnt * sizeof(uint64_t));
        if (!req_ids) {
          nbase_json_free(root);
          send_error_json(c, 500, "out of memory");
          return;
        }
        for (uint32_t i = 0U; i < req_cnt; i++) {
          if (!parse_json_u64(jids->items[i], &req_ids[i])) {
            free(req_ids);
            nbase_json_free(root);
            send_error_json(c, 400, "each id must be an unsigned 64-bit integer");
            return;
          }
        }
      }
    }
    const nbase_json_t *jflt = nbase_json_obj_get(root, "filters");
    if (jflt != NULL) {
      if (parse_meta_filter_json(jflt, &conds, &meta_flt) != NBASE_OK) {
        free(conds);
        free(req_ids);
        nbase_json_free(root);
        send_error_json(c, 400, "invalid metadata filters");
        return;
      }
      has_filter = (meta_flt.count > 0U);
    }
  }

  nbase_stats_t db_st;
  nbase_status_t st = nbase_stats(srv->db, &db_st);
  if (st != NBASE_OK) {
    free(conds);
    free(req_ids);
    nbase_json_free(root);
    send_error_json(c, 500, nbase_strerror(st));
    return;
  }
  if (db_st.active_vectors == 0ULL || (has_ids && req_cnt == 0U)) {
    free(conds);
    free(req_ids);
    nbase_json_free(root);
    send_http_response(c, 200, "application/x-ndjson", "", 0U);
    return;
  }
  if (!has_ids && !has_filter) {
    free(conds);
    nbase_json_free(root);
    export_unfiltered_ndjson(c, db_st.dim, inc_vec, inc_meta, inc_text);
    return;
  }

  uint64_t *export_ids = NULL;
  uint32_t export_cnt = 0U;

  if (has_filter) {
    if (db_st.active_vectors > 1000000ULL) {
      free(conds);
      free(req_ids);
      nbase_json_free(root);
      send_error_json(c, 400, "too many vectors for single HTTP export response");
      return;
    }
    uint32_t cap = (uint32_t)db_st.active_vectors + 64U;
    uint64_t *match_ids = malloc((size_t)cap * sizeof(uint64_t));
    if (!match_ids) {
      free(conds);
      free(req_ids);
      nbase_json_free(root);
      send_error_json(c, 500, "out of memory");
      return;
    }
    uint32_t matched = 0U;
    st = nbase_query_meta(srv->db, &meta_flt, cap, match_ids, &matched);
    if (st != NBASE_OK) {
      free(match_ids);
      free(conds);
      free(req_ids);
      nbase_json_free(root);
      send_error_json(c, http_code_for_status(st), nbase_strerror(st));
      return;
    }
    if (!has_ids) {
      export_ids = match_ids;
      export_cnt = matched;
    } else {
      if (matched > 1U) {
        qsort(match_ids, matched, sizeof(uint64_t), cmp_u64_asc_http);
      }
      for (uint32_t i = 0U; i < req_cnt; i++) {
        if (u64_in_sorted(match_ids, matched, req_ids[i])) {
          req_ids[export_cnt++] = req_ids[i];
        }
      }
      free(match_ids);
      export_ids = req_ids;
      req_ids = NULL;
    }
  } else {
    for (uint32_t i = 0U; i < req_cnt; i++) {
      if (nbase_contains(srv->db, req_ids[i])) {
        req_ids[export_cnt++] = req_ids[i];
      }
    }
    export_ids = req_ids;
    req_ids = NULL;
  }

  free(req_ids);
  free(conds);
  nbase_json_free(root);
  if (export_cnt == 0U) {
    free(export_ids);
    send_http_response(c, 200, "application/x-ndjson", "", 0U);
    return;
  }

  const uint32_t page_cap = 1024U;
  uint32_t buf_n = (export_cnt < page_cap) ? export_cnt : page_cap;
  float *vecs = inc_vec ? malloc((size_t)buf_n * (size_t)db_st.dim * sizeof(float)) : NULL;
  bool *found = inc_vec ? malloc((size_t)buf_n * sizeof(bool)) : NULL;
  if (inc_vec && (!vecs || !found)) {
    free(export_ids);
    free(vecs);
    free(found);
    send_error_json(c, 500, "out of memory");
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  for (uint32_t off = 0U; off < export_cnt && st == NBASE_OK; off += page_cap) {
    uint32_t chunk_n = export_cnt - off;
    if (chunk_n > page_cap) {
      chunk_n = page_cap;
    }
    st = append_ndjson_batch(
        &sb, srv->db, export_ids + off, chunk_n, db_st.dim, inc_vec, inc_meta, inc_text, vecs, found);
  }

  free(export_ids);
  free(vecs);
  free(found);
  if (st != NBASE_OK || sb.oom) {
    nbase_sbuf_free(&sb);
    send_error_json(
        c,
        http_code_for_status(st != NBASE_OK ? st : NBASE_ERR_NOMEM),
        nbase_strerror(st != NBASE_OK ? st : NBASE_ERR_NOMEM));
    return;
  }
  send_http_response(c, 200, "application/x-ndjson", sb.data ? sb.data : "", sb.len);
  nbase_sbuf_free(&sb);
}

static void handle_import_vectors(
    const http_conn_t *c,
    const char *query_str,
    const char *body,
    size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  bool is_upsert = query_param_bool(query_str, "upsert", false);
  uint32_t threads = 0U;
  char tbuf[32];
  if (query_param_get(query_str, "threads", tbuf, sizeof(tbuf))) {
    uint64_t tv = 0ULL;
    if (!parse_u64_str(tbuf, strlen(tbuf), &tv) || tv > 64ULL) {
      send_error_json(c, 400, "invalid threads parameter");
      return;
    }
    threads = (uint32_t)tv;
  }

  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    send_error_json(c, 500, "failed to read db stats");
    return;
  }
  if (!body || body_len == 0U) {
    send_json_response(c, 200, "{\"success\":true,\"imported\":0}", 29U);
    return;
  }

  size_t est_lines = 1U;
  for (size_t i = 0U; i < body_len; i++) {
    if (body[i] == '\n') {
      est_lines++;
    }
  }
  size_t cap = (est_lines < 10000U) ? est_lines : 10000U;

  uint64_t *ids = malloc(cap * sizeof(uint64_t));
  float *vecs = malloc(cap * (size_t)db_st.dim * sizeof(float));
  nbase_json_t **roots = calloc(cap, sizeof(nbase_json_t *));
  if (!ids || !vecs || !roots) {
    free(ids);
    free(vecs);
    free(roots);
    send_error_json(c, 500, "out of memory");
    return;
  }

  uint32_t count = 0U;
  size_t pos = 0U;
  while (pos < body_len) {
    size_t line_start = pos;
    while (pos < body_len && body[pos] != '\n') {
      pos++;
    }
    size_t line_end = pos;
    if (pos < body_len && body[pos] == '\n') {
      pos++;
    }
    while (line_start < line_end &&
           (body[line_start] == ' ' || body[line_start] == '\t' || body[line_start] == '\r')) {
      line_start++;
    }
    while (line_end > line_start &&
           (body[line_end - 1U] == ' ' || body[line_end - 1U] == '\t' || body[line_end - 1U] == '\r')) {
      line_end--;
    }
    if (line_start == line_end) {
      continue;
    }
    if (count >= 10000U) {
      for (uint32_t j = 0U; j < count; j++) {
        nbase_json_free(roots[j]);
      }
      free(ids);
      free(vecs);
      free(roots);
      send_error_json(c, 400, "at most 10000 NDJSON records per import request");
      return;
    }

    nbase_json_t *obj = NULL;
    if (nbase_json_parse(body + line_start, line_end - line_start, &obj) != NBASE_OK ||
        obj->type != NBASE_JSON_OBJECT) {
      nbase_json_free(obj);
      for (uint32_t j = 0U; j < count; j++) {
        nbase_json_free(roots[j]);
      }
      free(ids);
      free(vecs);
      free(roots);
      send_error_json(c, 400, "invalid JSON line in NDJSON body");
      return;
    }
    uint64_t id = 0ULL;
    if (!parse_json_u64(nbase_json_obj_get(obj, "id"), &id)) {
      nbase_json_free(obj);
      for (uint32_t j = 0U; j < count; j++) {
        nbase_json_free(roots[j]);
      }
      free(ids);
      free(vecs);
      free(roots);
      send_error_json(c, 400, "each NDJSON item requires uint64 id");
      return;
    }
    const nbase_json_t *jtxt = nbase_json_obj_get(obj, "text");
    if (jtxt != NULL && jtxt->type != NBASE_JSON_NULL && jtxt->type != NBASE_JSON_STRING) {
      nbase_json_free(obj);
      for (uint32_t j = 0U; j < count; j++) {
        nbase_json_free(roots[j]);
      }
      free(ids);
      free(vecs);
      free(roots);
      send_error_json(c, 400, "text must be a string");
      return;
    }
    float *tmp_v = NULL;
    uint32_t item_dim = 0U;
    if (!parse_float_array(nbase_json_obj_get(obj, "vector"), &tmp_v, &item_dim) ||
        item_dim != db_st.dim) {
      free(tmp_v);
      nbase_json_free(obj);
      for (uint32_t j = 0U; j < count; j++) {
        nbase_json_free(roots[j]);
      }
      free(ids);
      free(vecs);
      free(roots);
      send_error_json(c, 400, "invalid vector or dimension mismatch");
      return;
    }
    memcpy(vecs + (size_t)count * (size_t)db_st.dim, tmp_v, (size_t)db_st.dim * sizeof(float));
    free(tmp_v);
    ids[count] = id;
    roots[count] = obj;
    count++;
  }

  nbase_status_t st = NBASE_OK;
  if (count > 0U) {
    st = is_upsert
             ? nbase_upsert_batch(srv->db, ids, vecs, count, db_st.dim, threads)
             : nbase_add_batch(srv->db, ids, vecs, count, db_st.dim, threads);
    if (st == NBASE_OK) {
      st = apply_json_items_meta_and_text(srv->db, ids, roots, count);
    }
  }
  for (uint32_t j = 0U; j < count; j++) {
    nbase_json_free(roots[j]);
  }
  free(roots);
  free(vecs);
  free(ids);

  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }
  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"imported\":%u}", count);
  send_sbuf_json(c, (is_upsert || count == 0U) ? 200 : 201, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_list_vectors(const http_conn_t *c, const char *query_str) {
  nbase_http_server_t *srv = c->srv;
  uint64_t offset = 0ULL;
  uint32_t limit = 50U;
  char obuf[32];
  if (query_param_get(query_str, "offset", obuf, sizeof(obuf))) {
    if (!parse_u64_str(obuf, strlen(obuf), &offset)) {
      send_error_json(c, 400, "offset must be an unsigned 64-bit integer");
      return;
    }
  }
  char lbuf[32];
  if (query_param_get(query_str, "limit", lbuf, sizeof(lbuf))) {
    uint64_t lv = 0ULL;
    if (!parse_u64_str(lbuf, strlen(lbuf), &lv) || lv == 0ULL || lv > 10000ULL) {
      send_error_json(c, 400, "limit must be in 1..10000");
      return;
    }
    limit = (uint32_t)lv;
  }

  bool inc_vec = query_param_bool(query_str, "includeVectors", false);
  bool inc_meta = query_param_bool(query_str, "includeMetadata", false);
  bool inc_text = query_param_bool(query_str, "includeText", false);

  uint64_t *ids = malloc((size_t)limit * sizeof(uint64_t));
  if (!ids) {
    send_error_json(c, 500, "out of memory");
    return;
  }

  uint32_t got = 0U;
  uint64_t total = 0ULL;
  nbase_status_t st = nbase_list_ids(srv->db, offset, limit, ids, &got, &total);
  if (st != NBASE_OK) {
    free(ids);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  uint32_t dim = 0U;
  if (inc_vec && got > 0U) {
    nbase_stats_t db_st;
    st = nbase_stats(srv->db, &db_st);
    if (st != NBASE_OK) {
      free(ids);
      send_error_json(c, 500, nbase_strerror(st));
      return;
    }
    dim = db_st.dim;
  }

  http_batch_enrich_t enrich;
  if (!http_batch_enrich_load(srv->db, ids, got, dim, inc_vec, inc_text, &enrich)) {
    free(ids);
    send_error_json(c, 500, "out of memory");
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(
      &sb,
      "{\"success\":true,\"total\":%llu,\"offset\":%llu,\"limit\":%u,\"count\":%u,\"ids\":[",
      (unsigned long long)total,
      (unsigned long long)offset,
      limit,
      got);
  for (uint32_t i = 0U; i < got; i++) {
    nbase_sbuf_appendf(&sb, i == 0U ? "%llu" : ",%llu", (unsigned long long)ids[i]);
  }
  nbase_sbuf_append(&sb, "],\"results\":[");
  for (uint32_t i = 0U; i < got; i++) {
    nbase_sbuf_appendf(&sb, i == 0U ? "{\"id\":%llu" : ",{\"id\":%llu", (unsigned long long)ids[i]);
    append_enriched_fields_json(&sb, srv->db, ids[i], i, inc_meta, inc_vec, inc_text, false, &enrich);
    nbase_sbuf_append_n(&sb, "}", 1U);
  }
  nbase_sbuf_append(&sb, "]}");

  http_batch_enrich_free(&enrich);
  free(ids);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_put_vector_id(const http_conn_t *c, uint64_t id, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jtxt = nbase_json_obj_get(root, "text");
  if (jtxt != NULL && jtxt->type != NBASE_JSON_NULL && jtxt->type != NBASE_JSON_STRING) {
    nbase_json_free(root);
    send_error_json(c, 400, "text must be a string");
    return;
  }

  float *vec = NULL;
  uint32_t dim = 0U;
  if (!parse_float_array(nbase_json_obj_get(root, "vector"), &vec, &dim)) {
    nbase_json_free(root);
    send_error_json(c, 400, "missing or invalid vector array");
    return;
  }

  nbase_status_t st = nbase_upsert(srv->db, id, vec, dim);
  free(vec);
  if (st == NBASE_OK) {
    const nbase_json_t *jmeta = nbase_json_obj_get(root, "metadata");
    if (jmeta != NULL && jmeta->type != NBASE_JSON_NULL) {
      nbase_meta_field_t *fields = NULL;
      uint32_t fcnt = 0U;
      st = parse_meta_fields_obj(jmeta, &fields, &fcnt);
      if (st == NBASE_OK && fcnt > 0U) {
        st = nbase_set_meta(srv->db, id, fields, fcnt);
      }
      free(fields);
    }
  }
  if (st == NBASE_OK && jtxt != NULL && jtxt->type == NBASE_JSON_STRING) {
    st = nbase_set_text(srv->db, id, jtxt->str);
  }
  nbase_json_free(root);

  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"id\":%llu}", (unsigned long long)id);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

typedef struct http_patch_item {
  uint64_t id;
  float *vec;
  uint32_t dim;
  nbase_meta_field_t *fields;
  uint32_t fcnt;
  const char *text;
  bool clear_text;
} http_patch_item_t;

static void http_patch_item_free(http_patch_item_t *it) {
  free(it->vec);
  free(it->fields);
  it->vec = NULL;
  it->fields = NULL;
  it->fcnt = 0U;
  it->text = NULL;
  it->clear_text = false;
}

static bool parse_and_validate_patch_item(
    const nbase_http_server_t *srv,
    uint32_t expected_dim,
    const nbase_json_t *obj,
    uint64_t id,
    http_patch_item_t *out,
    int *out_http_code,
    const char **out_msg) {
  memset(out, 0, sizeof(*out));
  out->id = id;
  const nbase_json_t *jvec = nbase_json_obj_get(obj, "vector");
  const nbase_json_t *jmeta = nbase_json_obj_get(obj, "metadata");
  const nbase_json_t *jtxt = nbase_json_obj_get(obj, "text");

  bool has_vec = (jvec != NULL && jvec->type != NBASE_JSON_NULL);
  bool has_meta = (jmeta != NULL && jmeta->type != NBASE_JSON_NULL);
  bool has_text = (jtxt != NULL);

  if (!has_vec && !has_meta && !has_text) {
    *out_http_code = 400;
    *out_msg = "at least one of vector, metadata, or text is required";
    return false;
  }
  if (jtxt != NULL && jtxt->type != NBASE_JSON_NULL && jtxt->type != NBASE_JSON_STRING) {
    *out_http_code = 400;
    *out_msg = "text must be a string or null";
    return false;
  }
  if (has_vec) {
    if (!parse_float_array(jvec, &out->vec, &out->dim)) {
      *out_http_code = 400;
      *out_msg = "invalid vector array";
      return false;
    }
    if (out->dim != expected_dim) {
      free(out->vec);
      out->vec = NULL;
      *out_http_code = http_code_for_status(NBASE_ERR_DIM_MISMATCH);
      *out_msg = nbase_strerror(NBASE_ERR_DIM_MISMATCH);
      return false;
    }
  }
  if (has_meta) {
    nbase_status_t mst = parse_meta_fields_obj(jmeta, &out->fields, &out->fcnt);
    if (mst != NBASE_OK) {
      http_patch_item_free(out);
      *out_http_code = http_code_for_status(mst);
      *out_msg = nbase_strerror(mst);
      return false;
    }
  }
  if (has_text) {
    if (jtxt->type == NBASE_JSON_STRING) {
      out->text = jtxt->str;
    } else {
      out->clear_text = true;
    }
  }
  if (!nbase_contains(srv->db, id)) {
    http_patch_item_free(out);
    *out_http_code = 404;
    *out_msg = nbase_strerror(NBASE_ERR_NOT_FOUND);
    return false;
  }
  return true;
}

static nbase_status_t apply_patch_item(nbase_db_t *db, const http_patch_item_t *it) {
  nbase_status_t st = NBASE_OK;
  if (it->vec != NULL) {
    st = nbase_upsert(db, it->id, it->vec, it->dim);
  }
  if (st == NBASE_OK && it->fields != NULL && it->fcnt > 0U) {
    st = nbase_update_meta(db, it->id, it->fields, it->fcnt);
  }
  if (st == NBASE_OK) {
    if (it->clear_text) {
      st = nbase_delete_text(db, it->id);
      if (st == NBASE_ERR_NOT_FOUND) {
        st = NBASE_OK;
      }
    } else if (it->text != NULL) {
      st = nbase_set_text(db, it->id, it->text);
    }
  }
  return st;
}

static void handle_patch_vector_id(const http_conn_t *c, uint64_t id, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  nbase_stats_t db_st;
  (void)nbase_stats(srv->db, &db_st);
  http_patch_item_t item;
  int err_code = 400;
  const char *err_msg = "invalid request";
  if (!parse_and_validate_patch_item(srv, db_st.dim, root, id, &item, &err_code, &err_msg)) {
    nbase_json_free(root);
    send_error_json(c, err_code, err_msg);
    return;
  }

  nbase_status_t st = apply_patch_item(srv->db, &item);
  http_patch_item_free(&item);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"id\":%llu}", (unsigned long long)id);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_patch_vectors_batch(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jitems = nbase_json_obj_get(root, "items");
  if (!jitems || jitems->type != NBASE_JSON_ARRAY || jitems->len == 0U || jitems->len > 100000U) {
    nbase_json_free(root);
    send_error_json(c, 400, "items must be a non-empty array of at most 100000 objects");
    return;
  }

  uint32_t cnt = (uint32_t)jitems->len;
  http_patch_item_t *items = calloc(cnt, sizeof(*items));
  if (!items) {
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  nbase_stats_t db_st;
  (void)nbase_stats(srv->db, &db_st);
  int err_code = 0;
  const char *err_msg = NULL;

  for (uint32_t i = 0U; i < cnt; i++) {
    const nbase_json_t *it = jitems->items[i];
    if (!it || it->type != NBASE_JSON_OBJECT) {
      err_code = 400;
      err_msg = "each item must be a JSON object";
      break;
    }
    uint64_t id = 0ULL;
    if (!parse_json_u64(nbase_json_obj_get(it, "id"), &id)) {
      err_code = 400;
      err_msg = "each item requires unsigned 64-bit integer id";
      break;
    }
    if (!parse_and_validate_patch_item(srv, db_st.dim, it, id, &items[i], &err_code, &err_msg)) {
      break;
    }
  }

  if (err_code != 0) {
    for (uint32_t i = 0U; i < cnt; i++) {
      http_patch_item_free(&items[i]);
    }
    free(items);
    nbase_json_free(root);
    send_error_json(c, err_code, err_msg);
    return;
  }

  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < cnt && st == NBASE_OK; i++) {
    st = apply_patch_item(srv->db, &items[i]);
  }
  for (uint32_t i = 0U; i < cnt; i++) {
    http_patch_item_free(&items[i]);
  }
  free(items);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"patched\":%u}", cnt);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_put_vector_text(const http_conn_t *c, uint64_t id, const char *body, size_t body_len) {
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }
  const nbase_json_t *jtxt = nbase_json_obj_get(root, "text");
  if (!jtxt || jtxt->type != NBASE_JSON_STRING) {
    nbase_json_free(root);
    send_error_json(c, 400, "text string field is required");
    return;
  }
  nbase_status_t st = nbase_set_text(c->srv->db, id, jtxt->str);
  nbase_json_free(root);
  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }
  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"id\":%llu}", (unsigned long long)id);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_get_vector_id(const http_conn_t *c, uint64_t id, const char *query_str) {
  nbase_http_server_t *srv = c->srv;
  if (!nbase_contains(srv->db, id)) {
    send_error_json(c, 404, nbase_strerror(NBASE_ERR_NOT_FOUND));
    return;
  }
  bool inc_vec = query_param_bool(query_str, "includeVector", true);
  bool inc_meta = query_param_bool(query_str, "includeMetadata", true);
  bool inc_text = query_param_bool(query_str, "includeText", false);

  nbase_stats_t db_st;
  nbase_status_t st = nbase_stats(srv->db, &db_st);
  if (st != NBASE_OK) {
    send_error_json(c, 500, nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"id\":%llu,\"dimension\":%u", (unsigned long long)id, db_st.dim);
  if (inc_vec) {
    nbase_sbuf_append(&sb, ",\"vector\":");
    append_vector_json(&sb, srv->db, id, db_st.dim);
  }
  if (inc_meta) {
    nbase_sbuf_append(&sb, ",\"metadata\":");
    append_meta_json(&sb, srv->db, id);
  }
  if (inc_text) {
    nbase_sbuf_append(&sb, ",\"text\":");
    append_text_json(&sb, srv->db, id);
  }
  nbase_sbuf_append(&sb, "}");
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void append_result_entries_json(
    nbase_sbuf_t *sb,
    nbase_db_t *db,
    const nbase_result_t *res,
    uint32_t count,
    uint32_t dim,
    bool inc_meta,
    bool inc_vec,
    bool inc_text);

static void send_similar_results(
    const http_conn_t *c,
    const nbase_result_t *res,
    uint32_t got,
    uint32_t dim,
    bool inc_meta,
    bool inc_vec,
    bool inc_text) {
  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"count\":%u,\"results\":[", got);
  append_result_entries_json(&sb, c->srv->db, res, got, dim, inc_meta, inc_vec, inc_text);
  nbase_sbuf_append(&sb, "]}");
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_get_similar(const http_conn_t *c, uint64_t id, const char *query_str) {
  nbase_http_server_t *srv = c->srv;
  uint32_t k = 10U;
  char kbuf[32];
  if (query_param_get(query_str, "k", kbuf, sizeof(kbuf))) {
    uint64_t kv = 0ULL;
    if (!parse_u64_str(kbuf, strlen(kbuf), &kv) || kv == 0ULL || kv > 10000ULL) {
      send_error_json(c, 400, "k must be in 1..10000");
      return;
    }
    k = (uint32_t)kv;
  }
  bool inc_meta = query_param_bool(query_str, "includeMetadata", false);
  bool inc_vec = query_param_bool(query_str, "includeVectors", false);
  bool inc_text = query_param_bool(query_str, "includeText", false);

  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    send_error_json(c, 500, "failed to read db stats");
    return;
  }

  nbase_result_t *res = malloc((size_t)k * sizeof(nbase_result_t));
  if (!res) {
    send_error_json(c, 500, "out of memory");
    return;
  }

  uint64_t t0 = monotonic_us();
  uint32_t got = 0U;
  nbase_status_t st = nbase_find_similar(srv->db, id, k, NULL, res, &got);
  uint64_t t1 = monotonic_us();
  record_search_latency(srv, 1ULL, t1 - t0);

  if (st != NBASE_OK) {
    free(res);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  send_similar_results(c, res, got, db_st.dim, inc_meta, inc_vec, inc_text);
  free(res);
}

static bool parse_search_common_opts(
    const nbase_json_t *root,
    nbase_search_opts_t *opts,
    nbase_meta_cond_t **out_conds,
    nbase_meta_filter_t *out_filter);

static void handle_post_similar(
    const http_conn_t *c,
    uint64_t id,
    const char *body,
    size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  uint32_t k = 10U;
  bool inc_meta = false;
  bool inc_vec = false;
  bool inc_text = false;
  nbase_search_opts_t opts = nbase_search_opts_default();
  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt = {NULL, 0U};

  if (body_len > 0U && body != NULL) {
    if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
      nbase_json_free(root);
      send_error_json(c, 400, "invalid JSON request body");
      return;
    }

    const nbase_json_t *jk = nbase_json_obj_get(root, "k");
    if (jk != NULL) {
      if (jk->type != NBASE_JSON_NUMBER || !jk->is_int || jk->i64 <= 0 || jk->i64 > 10000) {
        nbase_json_free(root);
        send_error_json(c, 400, "k must be in 1..10000");
        return;
      }
      k = (uint32_t)jk->i64;
    }

    if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
      nbase_json_free(root);
      return;
    }

    if (!parse_search_common_opts(root, &opts, &conds, &meta_flt)) {
      free(conds);
      nbase_json_free(root);
      send_error_json(c, 400, "invalid search options");
      return;
    }
  }

  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "failed to read db stats");
    return;
  }

  nbase_result_t *res = malloc((size_t)k * sizeof(nbase_result_t));
  if (!res) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  uint64_t t0 = monotonic_us();
  uint32_t got = 0U;
  nbase_status_t st = nbase_find_similar(srv->db, id, k, &opts, res, &got);
  uint64_t t1 = monotonic_us();
  record_search_latency(srv, 1ULL, t1 - t0);

  free(conds);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    free(res);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  send_similar_results(c, res, got, db_st.dim, inc_meta, inc_vec, inc_text);
  free(res);
}

static void handle_search_similar_batch(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jids = nbase_json_obj_get(root, "ids");
  if (!jids || jids->type != NBASE_JSON_ARRAY || jids->len == 0U || jids->len > 1000U) {
    nbase_json_free(root);
    send_error_json(c, 400, "ids must be a non-empty array of at most 1000 ids");
    return;
  }

  uint32_t k = 10U;
  const nbase_json_t *jk = nbase_json_obj_get(root, "k");
  if (jk != NULL) {
    if (jk->type != NBASE_JSON_NUMBER || !jk->is_int || jk->i64 <= 0 || jk->i64 > 10000) {
      nbase_json_free(root);
      send_error_json(c, 400, "k must be in 1..10000");
      return;
    }
    k = (uint32_t)jk->i64;
  }

  bool inc_meta = false;
  bool inc_vec = false;
  bool inc_text = false;
  if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
    nbase_json_free(root);
    return;
  }

  nbase_search_opts_t opts = nbase_search_opts_default();
  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt = {NULL, 0U};
  if (!parse_search_common_opts(root, &opts, &conds, &meta_flt)) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 400, "invalid search options");
    return;
  }

  uint32_t cnt = (uint32_t)jids->len;
  uint64_t *ids = malloc((size_t)cnt * sizeof(uint64_t));
  if (!ids) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }
  for (uint32_t i = 0U; i < cnt; i++) {
    if (!parse_json_u64(jids->items[i], &ids[i])) {
      free(ids);
      free(conds);
      nbase_json_free(root);
      send_error_json(c, 400, "ids array elements must be unsigned 64-bit integers");
      return;
    }
  }

  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    free(ids);
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "failed to read db stats");
    return;
  }

  for (uint32_t i = 0U; i < cnt; i++) {
    if (!nbase_contains(srv->db, ids[i])) {
      free(ids);
      free(conds);
      nbase_json_free(root);
      send_error_json(c, 404, nbase_strerror(NBASE_ERR_NOT_FOUND));
      return;
    }
  }

  nbase_result_t *res_buf = malloc((size_t)cnt * (size_t)k * sizeof(nbase_result_t));
  uint32_t *res_cnts = calloc(cnt, sizeof(uint32_t));
  if (!res_buf || !res_cnts) {
    free(res_buf);
    free(res_cnts);
    free(ids);
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  uint64_t t0 = monotonic_us();
  nbase_status_t st = NBASE_OK;
  for (uint32_t q = 0U; q < cnt && st == NBASE_OK; q++) {
    st = nbase_find_similar(
        srv->db, ids[q], k, &opts, res_buf + (size_t)q * (size_t)k, &res_cnts[q]);
  }
  uint64_t t1 = monotonic_us();
  record_search_latency(srv, cnt, t1 - t0);

  free(ids);
  free(conds);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    free(res_buf);
    free(res_cnts);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"count\":%u,\"results\":[", cnt);
  for (uint32_t q = 0U; q < cnt && !sb.oom; q++) {
    nbase_sbuf_append_n(&sb, q == 0U ? "[" : ",[", q == 0U ? 1U : 2U);
    append_result_entries_json(
        &sb,
        srv->db,
        res_buf + (size_t)q * (size_t)k,
        res_cnts[q],
        db_st.dim,
        inc_meta,
        inc_vec,
        inc_text);
    nbase_sbuf_append_n(&sb, "]", 1U);
  }
  nbase_sbuf_append(&sb, "]}");
  free(res_buf);
  free(res_cnts);

  if (sb.oom) {
    nbase_sbuf_free(&sb);
    send_error_json(c, 500, "out of memory");
    return;
  }
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_id_metadata_write(
    const http_conn_t *c,
    uint64_t id,
    const char *body,
    size_t body_len,
    bool is_update) {
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *src = nbase_json_obj_get(root, "metadata");
  if (!src) {
    src = root;
  }
  nbase_meta_field_t *fields = NULL;
  uint32_t fcnt = 0U;
  nbase_status_t st = parse_meta_fields_obj(src, &fields, &fcnt);
  if (st == NBASE_OK) {
    if (!is_update) {
      st = nbase_set_meta(c->srv->db, id, fields, fcnt);
    } else if (fcnt > 0U) {
      st = nbase_update_meta(c->srv->db, id, fields, fcnt);
    } else if (!nbase_contains(c->srv->db, id)) {
      st = NBASE_ERR_NOT_FOUND;
    }
  }
  free(fields);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"id\":%llu}", (unsigned long long)id);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_batch_metadata(
    const http_conn_t *c,
    const char *body,
    size_t body_len,
    bool default_update) {
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  bool is_update = default_update;
  const nbase_json_t *jmode = nbase_json_obj_get(root, "mode");
  if (jmode != NULL) {
    if (jmode->type != NBASE_JSON_STRING) {
      nbase_json_free(root);
      send_error_json(c, 400, "mode must be 'set' or 'update'");
      return;
    }
    if (strcmp(jmode->str, "set") == 0) {
      is_update = false;
    } else if (strcmp(jmode->str, "update") == 0) {
      is_update = true;
    } else {
      nbase_json_free(root);
      send_error_json(c, 400, "mode must be 'set' or 'update'");
      return;
    }
  }

  const nbase_json_t *jitems = nbase_json_obj_get(root, "items");
  if (!jitems || jitems->type != NBASE_JSON_ARRAY || jitems->len == 0U || jitems->len > 100000U) {
    nbase_json_free(root);
    send_error_json(c, 400, "items must be a non-empty array of at most 100000 objects");
    return;
  }

  uint32_t cnt = (uint32_t)jitems->len;
  nbase_meta_batch_item_t *items = calloc(cnt, sizeof(*items));
  nbase_meta_field_t **field_bufs = calloc(cnt, sizeof(*field_bufs));
  if (!items || !field_bufs) {
    free(items);
    free(field_bufs);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < cnt; i++) {
    const nbase_json_t *it = jitems->items[i];
    if (!it || it->type != NBASE_JSON_OBJECT) {
      st = NBASE_ERR_INVALID_ARG;
      break;
    }
    uint64_t id = 0ULL;
    const nbase_json_t *jmeta = nbase_json_obj_get(it, "metadata");
    if (!parse_json_u64(nbase_json_obj_get(it, "id"), &id) || !jmeta || jmeta->type != NBASE_JSON_OBJECT) {
      st = NBASE_ERR_INVALID_ARG;
      break;
    }
    nbase_meta_field_t *fields = NULL;
    uint32_t fcnt = 0U;
    st = parse_meta_fields_obj(jmeta, &fields, &fcnt);
    if (st != NBASE_OK) {
      free(fields);
      break;
    }
    field_bufs[i] = fields;
    items[i].id = id;
    items[i].fields = fields;
    items[i].field_count = fcnt;
  }

  if (st == NBASE_OK) {
    st = is_update
             ? nbase_update_meta_batch(c->srv->db, items, cnt)
             : nbase_set_meta_batch(c->srv->db, items, cnt);
  }

  for (uint32_t i = 0U; i < cnt; i++) {
    free(field_bufs[i]);
  }
  free(field_bufs);
  free(items);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"updated\":%u}", cnt);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_batch_text(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jitems = nbase_json_obj_get(root, "items");
  if (!jitems || jitems->type != NBASE_JSON_ARRAY || jitems->len == 0U || jitems->len > 100000U) {
    nbase_json_free(root);
    send_error_json(c, 400, "items must be a non-empty array of at most 100000 objects");
    return;
  }

  uint32_t cnt = (uint32_t)jitems->len;
  uint64_t *ids = malloc((size_t)cnt * sizeof(uint64_t));
  const char **texts = malloc((size_t)cnt * sizeof(const char *));
  if (!ids || !texts) {
    free(ids);
    free((void *)texts);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < cnt; i++) {
    const nbase_json_t *it = jitems->items[i];
    if (!it || it->type != NBASE_JSON_OBJECT) {
      st = NBASE_ERR_INVALID_ARG;
      break;
    }
    const nbase_json_t *jtxt = nbase_json_obj_get(it, "text");
    if (!parse_json_u64(nbase_json_obj_get(it, "id"), &ids[i]) ||
        !jtxt || jtxt->type != NBASE_JSON_STRING) {
      st = NBASE_ERR_INVALID_ARG;
      break;
    }
    texts[i] = jtxt->str;
  }

  if (st == NBASE_OK) {
    st = nbase_set_text_batch(c->srv->db, ids, texts, cnt);
  }
  free(ids);
  free((void *)texts);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"updated\":%u}", cnt);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_count_vectors(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_json_t *root = NULL;
  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt = {NULL, 0U};
  bool has_filter = false;

  if (body != NULL && body_len > 0U) {
    if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
      nbase_json_free(root);
      send_error_json(c, 400, "invalid JSON request body");
      return;
    }
    const nbase_json_t *jflt = nbase_json_obj_get(root, "filters");
    if (jflt != NULL) {
      if (parse_meta_filter_json(jflt, &conds, &meta_flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(root);
        send_error_json(c, 400, "invalid metadata filters");
        return;
      }
      has_filter = (meta_flt.count > 0U);
    }
  }

  uint64_t cnt = 0ULL;
  nbase_status_t st = nbase_count(c->srv->db, has_filter ? &meta_flt : NULL, &cnt);
  free(conds);
  nbase_json_free(root);
  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"count\":%llu}", (unsigned long long)cnt);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static bool parse_search_common_opts(
    const nbase_json_t *root,
    nbase_search_opts_t *opts,
    nbase_meta_cond_t **out_conds,
    nbase_meta_filter_t *out_filter) {
  *opts = nbase_search_opts_default();
  *out_conds = NULL;
  out_filter->conds = NULL;
  out_filter->count = 0U;

  const nbase_json_t *jef = nbase_json_obj_get(root, "efSearch");
  if (jef != NULL) {
    if (jef->type != NBASE_JSON_NUMBER || !jef->is_int || jef->i64 <= 0 || jef->i64 > 100000) {
      return false;
    }
    opts->ef_search = (uint32_t)jef->i64;
  }

  const nbase_json_t *jmetric = nbase_json_obj_get(root, "metric");
  if (jmetric != NULL) {
    if (jmetric->type != NBASE_JSON_STRING || !parse_metric_str(jmetric->str, &opts->metric)) {
      return false;
    }
    opts->use_metric_override = true;
  }

  const nbase_json_t *jmode = nbase_json_obj_get(root, "mode");
  if (jmode != NULL) {
    if (jmode->type != NBASE_JSON_STRING) {
      return false;
    }
    if (strcmp(jmode->str, "auto") == 0) {
      opts->mode = NBASE_SEARCH_AUTO;
    } else if (strcmp(jmode->str, "exact") == 0) {
      opts->mode = NBASE_SEARCH_EXACT;
    } else if (strcmp(jmode->str, "hnsw") == 0) {
      opts->mode = NBASE_SEARCH_HNSW;
    } else {
      return false;
    }
  }

  const nbase_json_t *jskip = nbase_json_obj_get(root, "skipCache");
  if (jskip != NULL) {
    if (jskip->type != NBASE_JSON_BOOL) {
      return false;
    }
    opts->skip_cache = jskip->b;
  }

  const nbase_json_t *jflt = nbase_json_obj_get(root, "filters");
  if (jflt != NULL) {
    if (parse_meta_filter_json(jflt, out_conds, out_filter) != NBASE_OK) {
      return false;
    }
    if (out_filter->count > 0U) {
      opts->meta_filter = out_filter;
    }
  }
  return true;
}

static bool parse_rerank_weights_obj(
    const nbase_json_t *jrw,
    nbase_rerank_weight_t **out_weights,
    uint32_t *out_count,
    const char **out_err,
    int *out_http_code) {
  *out_weights = NULL;
  *out_count = 0U;
  if (jrw->type != NBASE_JSON_OBJECT || jrw->len > 1024U) {
    *out_err = "rerankWeights must be a JSON object";
    *out_http_code = 400;
    return false;
  }
  if (jrw->len == 0U) {
    return true;
  }
  nbase_rerank_weight_t *w = calloc(jrw->len, sizeof(nbase_rerank_weight_t));
  if (!w) {
    *out_err = "out of memory";
    *out_http_code = 500;
    return false;
  }
  for (size_t wi = 0U; wi < jrw->len; wi++) {
    const char *wk = jrw->keys[wi];
    const nbase_json_t *wv = jrw->vals[wi];
    if (!wk || wk[0] == '\0' || !wv || wv->type != NBASE_JSON_NUMBER || !isfinite(wv->num)) {
      free(w);
      *out_err = "rerankWeights values must be finite numbers";
      *out_http_code = 400;
      return false;
    }
    w[wi].key = wk;
    w[wi].weight = (float)wv->num;
  }
  *out_weights = w;
  *out_count = (uint32_t)jrw->len;
  return true;
}

static void append_result_entries_json(
    nbase_sbuf_t *sb,
    nbase_db_t *db,
    const nbase_result_t *res,
    uint32_t count,
    uint32_t dim,
    bool inc_meta,
    bool inc_vec,
    bool inc_text) {
  http_batch_enrich_t enrich;
  memset(&enrich, 0, sizeof(enrich));
  if (count > 0U && (inc_vec || inc_text)) {
    uint64_t stack_ids[128];
    uint64_t *ids = (count <= 128U) ? stack_ids : malloc((size_t)count * sizeof(uint64_t));
    if (!ids) {
      sb->oom = true;
      return;
    }
    for (uint32_t i = 0U; i < count; i++) {
      ids[i] = res[i].id;
    }
    bool ok = http_batch_enrich_load(db, ids, count, dim, inc_vec, inc_text, &enrich);
    if (ids != stack_ids) {
      free(ids);
    }
    if (!ok) {
      sb->oom = true;
      return;
    }
  }
  for (uint32_t i = 0U; i < count; i++) {
    nbase_sbuf_appendf(
        sb,
        i == 0U ? "{\"id\":%llu,\"distance\":%.7g" : ",{\"id\":%llu,\"distance\":%.7g",
        (unsigned long long)res[i].id,
        (double)res[i].distance);
    append_enriched_fields_json(sb, db, res[i].id, i, inc_meta, inc_vec, inc_text, true, &enrich);
    nbase_sbuf_append_n(sb, "}", 1U);
  }
  http_batch_enrich_free(&enrich);
}

static void handle_search(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  float *query = NULL;
  uint32_t dim = 0U;
  if (!parse_float_array(nbase_json_obj_get(root, "query"), &query, &dim)) {
    nbase_json_free(root);
    send_error_json(c, 400, "missing or invalid query vector");
    return;
  }

  uint32_t k = 10U;
  const nbase_json_t *jk = nbase_json_obj_get(root, "k");
  if (jk != NULL) {
    if (jk->type != NBASE_JSON_NUMBER || !jk->is_int || jk->i64 <= 0 || jk->i64 > 10000) {
      free(query);
      nbase_json_free(root);
      send_error_json(c, 400, "k must be in 1..10000");
      return;
    }
    k = (uint32_t)jk->i64;
  }

  bool do_rerank = false;
  float mmr_lambda = 0.5f;
  int rerank_method = 0; /* 0=unspecified(diversity), 1=diversity, 2=weighted, 3=standard */
  nbase_rerank_weight_t *weights = NULL;
  uint32_t weight_cnt = 0U;

  const nbase_json_t *jrr = nbase_json_obj_get(root, "rerank");
  if (jrr != NULL) {
    if (jrr->type != NBASE_JSON_BOOL) {
      free(query);
      nbase_json_free(root);
      send_error_json(c, 400, "rerank must be boolean");
      return;
    }
    do_rerank = jrr->b;
  }
  const nbase_json_t *jrm = nbase_json_obj_get(root, "rerankingMethod");
  if (jrm == NULL) {
    jrm = nbase_json_obj_get(root, "rerankMethod");
  }
  if (jrm != NULL) {
    if (jrm->type != NBASE_JSON_STRING) {
      free(query);
      nbase_json_free(root);
      send_error_json(c, 400, "rerankingMethod must be a string");
      return;
    }
    if (strcmp(jrm->str, "diversity") == 0) {
      rerank_method = 1;
    } else if (strcmp(jrm->str, "weighted") == 0) {
      rerank_method = 2;
    } else if (strcmp(jrm->str, "standard") == 0) {
      rerank_method = 3;
    } else {
      free(query);
      nbase_json_free(root);
      send_error_json(c, 400, "rerankingMethod must be standard, diversity, or weighted");
      return;
    }
    if (jrr == NULL) {
      do_rerank = true;
    }
  }
  const nbase_json_t *jrw = nbase_json_obj_get(root, "rerankWeights");
  if (jrw != NULL) {
    const char *w_err = NULL;
    int w_code = 400;
    if (!parse_rerank_weights_obj(jrw, &weights, &weight_cnt, &w_err, &w_code)) {
      free(query);
      nbase_json_free(root);
      send_error_json(c, w_code, w_err);
      return;
    }
    if (jrr == NULL) {
      do_rerank = true;
    }
  }
  const nbase_json_t *jlam = nbase_json_obj_get(root, "rerankLambda");
  if (jlam != NULL) {
    if (jlam->type != NBASE_JSON_NUMBER || !isfinite(jlam->num) || jlam->num < 0.0 || jlam->num > 1.0) {
      free(weights);
      free(query);
      nbase_json_free(root);
      send_error_json(c, 400, "rerankLambda must be in [0, 1]");
      return;
    }
    mmr_lambda = (float)jlam->num;
  }

  bool inc_meta = false;
  bool inc_vec = false;
  bool inc_text = false;
  if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
    free(weights);
    free(query);
    nbase_json_free(root);
    return;
  }

  nbase_search_opts_t opts;
  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt;
  if (!parse_search_common_opts(root, &opts, &conds, &meta_flt)) {
    free(conds);
    free(weights);
    free(query);
    nbase_json_free(root);
    send_error_json(c, 400, "invalid search options");
    return;
  }

  bool use_radius = false;
  float max_distance = 0.0f;
  const nbase_json_t *jmd = nbase_json_obj_get(root, "maxDistance");
  if (jmd != NULL) {
    if (jmd->type != NBASE_JSON_NUMBER || !isfinite(jmd->num)) {
      free(conds);
      free(weights);
      free(query);
      nbase_json_free(root);
      send_error_json(c, 400, "maxDistance must be a finite number");
      return;
    }
    use_radius = true;
    max_distance = (float)jmd->num;
  }

  uint32_t fetch_k = do_rerank ? ((k <= 2500U) ? k * 4U : 10000U) : k;
  nbase_result_t *cands = malloc((size_t)fetch_k * sizeof(nbase_result_t));
  nbase_result_t *final_res = do_rerank ? malloc((size_t)k * sizeof(nbase_result_t)) : NULL;
  if (!cands || (do_rerank && !final_res)) {
    free(cands);
    free(final_res);
    free(conds);
    free(weights);
    free(query);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  uint64_t t0 = monotonic_us();
  uint32_t got = 0U;
  nbase_status_t st = use_radius
                          ? nbase_search_radius(
                                srv->db, query, dim, max_distance, fetch_k, &opts, cands, &got)
                          : nbase_search(srv->db, query, dim, fetch_k, &opts, cands, &got);
  free(conds);
  free(query);

  const nbase_result_t *out_ptr = cands;
  uint32_t out_n = got;
  if (st == NBASE_OK && do_rerank) {
    if (rerank_method == 2 || (rerank_method == 0 && weight_cnt > 0U)) {
      st = nbase_rerank_weighted(srv->db, cands, got, weights, weight_cnt, k, final_res, &out_n);
    } else if (rerank_method == 3) {
      out_n = (got < k) ? got : k;
      if (out_n > 0U) {
        memcpy(final_res, cands, (size_t)out_n * sizeof(nbase_result_t));
      }
    } else {
      st = nbase_rerank_mmr(srv->db, cands, got, k, mmr_lambda, final_res, &out_n);
    }
    out_ptr = final_res;
  }
  uint64_t t1 = monotonic_us();
  record_search_latency(srv, 1ULL, t1 - t0);
  free(weights);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    free(cands);
    free(final_res);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"count\":%u,\"results\":[", out_n);
  append_result_entries_json(&sb, srv->db, out_ptr, out_n, dim, inc_meta, inc_vec, inc_text);
  nbase_sbuf_append(&sb, "]}");
  free(cands);
  free(final_res);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_search_radius(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  float *query = NULL;
  uint32_t dim = 0U;
  if (!parse_float_array(nbase_json_obj_get(root, "query"), &query, &dim)) {
    nbase_json_free(root);
    send_error_json(c, 400, "missing or invalid query vector");
    return;
  }

  const nbase_json_t *jmd = nbase_json_obj_get(root, "maxDistance");
  if (!jmd || jmd->type != NBASE_JSON_NUMBER || !isfinite(jmd->num)) {
    free(query);
    nbase_json_free(root);
    send_error_json(c, 400, "maxDistance must be a finite number");
    return;
  }
  float max_distance = (float)jmd->num;
  if (!isfinite(max_distance)) {
    free(query);
    nbase_json_free(root);
    send_error_json(c, 400, "maxDistance must be a finite float32");
    return;
  }

  uint32_t max_results = 100U;
  const nbase_json_t *jlim = nbase_json_obj_get(root, "maxResults");
  if (jlim == NULL) {
    jlim = nbase_json_obj_get(root, "limit");
  }
  if (jlim != NULL) {
    if (jlim->type != NBASE_JSON_NUMBER || !jlim->is_int || jlim->i64 <= 0 || jlim->i64 > 10000) {
      free(query);
      nbase_json_free(root);
      send_error_json(c, 400, "maxResults must be in 1..10000");
      return;
    }
    max_results = (uint32_t)jlim->i64;
  }

  bool inc_meta = false;
  bool inc_vec = false;
  bool inc_text = false;
  if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
    free(query);
    nbase_json_free(root);
    return;
  }

  nbase_search_opts_t opts;
  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt;
  if (!parse_search_common_opts(root, &opts, &conds, &meta_flt)) {
    free(conds);
    free(query);
    nbase_json_free(root);
    send_error_json(c, 400, "invalid search options");
    return;
  }

  nbase_result_t *res = malloc((size_t)max_results * sizeof(nbase_result_t));
  if (!res) {
    free(conds);
    free(query);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  uint64_t t0 = monotonic_us();
  uint32_t got = 0U;
  nbase_status_t st = nbase_search_radius(
      srv->db, query, dim, max_distance, max_results, &opts, res, &got);
  uint64_t t1 = monotonic_us();
  record_search_latency(srv, 1ULL, t1 - t0);

  free(conds);
  free(query);
  nbase_json_free(root);
  if (st != NBASE_OK) {
    free(res);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"count\":%u,\"results\":[", got);
  append_result_entries_json(&sb, srv->db, res, got, dim, inc_meta, inc_vec, inc_text);
  nbase_sbuf_append(&sb, "]}");
  free(res);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static bool parse_threads_field(const nbase_json_t *root, uint32_t *out_threads) {
  const nbase_json_t *jthr = nbase_json_obj_get(root, "threads");
  if (jthr == NULL) {
    return true;
  }
  if (jthr->type != NBASE_JSON_NUMBER || !jthr->is_int || jthr->i64 < 0 || jthr->i64 > 64) {
    return false;
  }
  *out_threads = (uint32_t)jthr->i64;
  return true;
}

static void handle_search_batch(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jqs = nbase_json_obj_get(root, "queries");
  if (!jqs || jqs->type != NBASE_JSON_ARRAY || jqs->len == 0U || jqs->len > 10000U) {
    nbase_json_free(root);
    send_error_json(c, 400, "queries must be a non-empty array");
    return;
  }

  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    nbase_json_free(root);
    send_error_json(c, 500, "failed to read db stats");
    return;
  }

  uint32_t k = 10U;
  const nbase_json_t *jk = nbase_json_obj_get(root, "k");
  if (jk != NULL) {
    if (jk->type != NBASE_JSON_NUMBER || !jk->is_int || jk->i64 <= 0 || jk->i64 > 1000) {
      nbase_json_free(root);
      send_error_json(c, 400, "k must be in 1..1000");
      return;
    }
    k = (uint32_t)jk->i64;
  }

  bool inc_meta = false;
  bool inc_vec = false;
  bool inc_text = false;
  if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
    nbase_json_free(root);
    return;
  }

  nbase_search_opts_t opts;
  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt;
  if (!parse_search_common_opts(root, &opts, &conds, &meta_flt)) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 400, "invalid search options");
    return;
  }
  if (!parse_threads_field(root, &opts.num_threads)) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 400, "invalid threads parameter");
    return;
  }

  uint32_t qcnt = (uint32_t)jqs->len;
  uint32_t dim = db_st.dim;
  float *queries = malloc((size_t)qcnt * (size_t)dim * sizeof(float));
  nbase_result_t *results = malloc((size_t)qcnt * (size_t)k * sizeof(nbase_result_t));
  uint32_t *counts = malloc((size_t)qcnt * sizeof(uint32_t));
  if (!queries || !results || !counts) {
    free(queries);
    free(results);
    free(counts);
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  for (uint32_t qi = 0U; qi < qcnt; qi++) {
    float *qtmp = NULL;
    uint32_t qdim = 0U;
    if (!parse_float_array(jqs->items[qi], &qtmp, &qdim) || qdim != dim) {
      free(qtmp);
      free(queries);
      free(results);
      free(counts);
      free(conds);
      nbase_json_free(root);
      send_error_json(c, 400, "invalid query vector or dimension mismatch");
      return;
    }
    memcpy(queries + (size_t)qi * (size_t)dim, qtmp, (size_t)dim * sizeof(float));
    free(qtmp);
  }

  uint64_t t0 = monotonic_us();
  nbase_status_t st = nbase_search_batch(srv->db, queries, qcnt, dim, k, &opts, results, counts);
  uint64_t t1 = monotonic_us();
  record_search_latency(srv, qcnt, t1 - t0);

  free(queries);
  free(conds);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    free(results);
    free(counts);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_append(&sb, "{\"success\":true,\"results\":[");
  for (uint32_t qi = 0U; qi < qcnt; qi++) {
    nbase_sbuf_append(&sb, qi == 0U ? "[" : ",[");
    uint32_t got = counts[qi];
    const nbase_result_t *row = results + (size_t)qi * (size_t)k;
    append_result_entries_json(&sb, srv->db, row, got, dim, inc_meta, inc_vec, inc_text);
    nbase_sbuf_append(&sb, "]");
  }
  nbase_sbuf_append(&sb, "]}");
  free(results);
  free(counts);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_search_metadata(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  uint32_t limit = 100U;
  const nbase_json_t *jlim = nbase_json_obj_get(root, "limit");
  if (jlim != NULL) {
    if (jlim->type != NBASE_JSON_NUMBER || !jlim->is_int || jlim->i64 <= 0 || jlim->i64 > 10000) {
      nbase_json_free(root);
      send_error_json(c, 400, "limit must be in 1..10000");
      return;
    }
    limit = (uint32_t)jlim->i64;
  }

  bool inc_meta = true;
  bool inc_vec = false;
  bool inc_text = false;
  if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
    nbase_json_free(root);
    return;
  }

  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt;
  const nbase_json_t *jflt = nbase_json_obj_get(root, "filters");
  if (parse_meta_filter_json(jflt, &conds, &meta_flt) != NBASE_OK) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 400, "invalid metadata filters");
    return;
  }

  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "failed to read db stats");
    return;
  }

  uint64_t *ids = malloc((size_t)limit * sizeof(uint64_t));
  if (!ids) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  uint32_t got = 0U;
  nbase_status_t st = nbase_query_meta(srv->db, meta_flt.count > 0U ? &meta_flt : NULL, limit, ids, &got);
  free(conds);
  nbase_json_free(root);
  if (st != NBASE_OK) {
    free(ids);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  http_batch_enrich_t enrich;
  if (!http_batch_enrich_load(srv->db, ids, got, db_st.dim, inc_vec, inc_text, &enrich)) {
    free(ids);
    send_error_json(c, 500, "out of memory");
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"count\":%u,\"results\":[", got);
  for (uint32_t i = 0U; i < got; i++) {
    nbase_sbuf_appendf(&sb, i == 0U ? "{\"id\":%llu" : ",{\"id\":%llu", (unsigned long long)ids[i]);
    append_enriched_fields_json(&sb, srv->db, ids[i], i, inc_meta, inc_vec, inc_text, true, &enrich);
    nbase_sbuf_append_n(&sb, "}", 1U);
  }
  nbase_sbuf_append(&sb, "]}");
  http_batch_enrich_free(&enrich);
  free(ids);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void append_scored_entries_json(
    nbase_sbuf_t *sb,
    nbase_db_t *db,
    const nbase_scored_id_t *res,
    uint32_t count,
    uint32_t dim,
    bool inc_meta,
    bool inc_vec,
    bool inc_text) {
  http_batch_enrich_t enrich;
  memset(&enrich, 0, sizeof(enrich));
  if (count > 0U && (inc_vec || inc_text)) {
    uint64_t stack_ids[128];
    uint64_t *ids = (count <= 128U) ? stack_ids : malloc((size_t)count * sizeof(uint64_t));
    if (!ids) {
      sb->oom = true;
      return;
    }
    for (uint32_t i = 0U; i < count; i++) {
      ids[i] = res[i].id;
    }
    bool ok = http_batch_enrich_load(db, ids, count, dim, inc_vec, inc_text, &enrich);
    if (ids != stack_ids) {
      free(ids);
    }
    if (!ok) {
      sb->oom = true;
      return;
    }
  }
  for (uint32_t i = 0U; i < count; i++) {
    nbase_sbuf_appendf(
        sb,
        i == 0U ? "{\"id\":%llu,\"score\":%.7g" : ",{\"id\":%llu,\"score\":%.7g",
        (unsigned long long)res[i].id,
        (double)res[i].score);
    append_enriched_fields_json(sb, db, res[i].id, i, inc_meta, inc_vec, inc_text, true, &enrich);
    nbase_sbuf_append_n(sb, "}", 1U);
  }
  http_batch_enrich_free(&enrich);
}

static void handle_search_text(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jq = nbase_json_obj_get(root, "query");
  if (!jq || jq->type != NBASE_JSON_STRING) {
    nbase_json_free(root);
    send_error_json(c, 400, "query string field is required");
    return;
  }

  uint32_t k = 10U;
  const nbase_json_t *jk = nbase_json_obj_get(root, "k");
  if (jk != NULL) {
    if (jk->type != NBASE_JSON_NUMBER || !jk->is_int || jk->i64 <= 0 || jk->i64 > 10000) {
      nbase_json_free(root);
      send_error_json(c, 400, "k must be in 1..10000");
      return;
    }
    k = (uint32_t)jk->i64;
  }

  bool inc_meta = false;
  bool inc_vec = false;
  bool inc_text = false;
  if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
    nbase_json_free(root);
    return;
  }

  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt;
  if (parse_meta_filter_json(nbase_json_obj_get(root, "filters"), &conds, &meta_flt) != NBASE_OK) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 400, "invalid metadata filters");
    return;
  }

  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "failed to read db stats");
    return;
  }

  nbase_scored_id_t *res = malloc((size_t)k * sizeof(nbase_scored_id_t));
  if (!res) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  uint64_t t0 = monotonic_us();
  uint32_t got = 0U;
  nbase_status_t st = nbase_search_text(
      srv->db, jq->str, k, meta_flt.count > 0U ? &meta_flt : NULL, res, &got);
  uint64_t t1 = monotonic_us();
  record_search_latency(srv, 1ULL, t1 - t0);

  free(conds);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    free(res);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"count\":%u,\"results\":[", got);
  append_scored_entries_json(&sb, srv->db, res, got, db_st.dim, inc_meta, inc_vec, inc_text);
  nbase_sbuf_append(&sb, "]}");
  free(res);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_search_text_batch(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jqarr = nbase_json_obj_get(root, "queries");
  if (!jqarr || jqarr->type != NBASE_JSON_ARRAY || jqarr->len == 0U || jqarr->len > 1000U) {
    nbase_json_free(root);
    send_error_json(c, 400, "queries array of 1..1000 strings is required");
    return;
  }
  uint32_t qcnt = (uint32_t)jqarr->len;
  for (uint32_t i = 0U; i < qcnt; i++) {
    if (!jqarr->items[i] || jqarr->items[i]->type != NBASE_JSON_STRING) {
      nbase_json_free(root);
      send_error_json(c, 400, "each query must be a string");
      return;
    }
  }

  uint32_t k = 10U;
  const nbase_json_t *jk = nbase_json_obj_get(root, "k");
  if (jk != NULL) {
    if (jk->type != NBASE_JSON_NUMBER || !jk->is_int || jk->i64 <= 0 || jk->i64 > 10000) {
      nbase_json_free(root);
      send_error_json(c, 400, "k must be in 1..10000");
      return;
    }
    k = (uint32_t)jk->i64;
  }

  bool inc_meta = false;
  bool inc_vec = false;
  bool inc_text = false;
  if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
    nbase_json_free(root);
    return;
  }

  uint32_t num_threads = 0U;
  if (!parse_threads_field(root, &num_threads)) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid threads parameter");
    return;
  }

  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt;
  if (parse_meta_filter_json(nbase_json_obj_get(root, "filters"), &conds, &meta_flt) != NBASE_OK) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 400, "invalid metadata filters");
    return;
  }

  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "failed to read db stats");
    return;
  }

  const char **qptrs = malloc((size_t)qcnt * sizeof(const char *));
  nbase_scored_id_t *res = malloc((size_t)qcnt * (size_t)k * sizeof(nbase_scored_id_t));
  uint32_t *counts = calloc((size_t)qcnt, sizeof(uint32_t));
  if (!qptrs || !res || !counts) {
    free((void *)qptrs);
    free(res);
    free(counts);
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }
  for (uint32_t q = 0U; q < qcnt; q++) {
    qptrs[q] = jqarr->items[q]->str;
  }

  uint64_t t0 = monotonic_us();
  nbase_status_t st = nbase_search_text_batch(
      srv->db,
      qptrs,
      qcnt,
      k,
      num_threads,
      meta_flt.count > 0U ? &meta_flt : NULL,
      res,
      counts);
  uint64_t t1 = monotonic_us();
  record_search_latency(srv, (uint64_t)qcnt, t1 - t0);

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  if (st == NBASE_OK) {
    nbase_sbuf_appendf(&sb, "{\"success\":true,\"queryCount\":%u,\"results\":[", qcnt);
    for (uint32_t q = 0U; q < qcnt; q++) {
      nbase_sbuf_append(&sb, q == 0U ? "[" : ",[");
      append_scored_entries_json(
          &sb, srv->db, res + (size_t)q * (size_t)k, counts[q], db_st.dim, inc_meta, inc_vec, inc_text);
      nbase_sbuf_append_n(&sb, "]", 1U);
    }
    nbase_sbuf_append(&sb, "]}");
  }
  free((void *)qptrs);
  free(res);
  free(counts);
  free(conds);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    nbase_sbuf_free(&sb);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_search_hybrid(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  float *qvec = NULL;
  uint32_t qdim = 0U;
  const nbase_json_t *jqvec = nbase_json_obj_get(root, "query");
  if (jqvec != NULL && jqvec->type != NBASE_JSON_NULL) {
    if (!parse_float_array(jqvec, &qvec, &qdim)) {
      nbase_json_free(root);
      send_error_json(c, 400, "invalid query vector");
      return;
    }
  }

  const char *qtext = NULL;
  const nbase_json_t *jtxt = nbase_json_obj_get(root, "text");
  if (jtxt != NULL && jtxt->type != NBASE_JSON_NULL) {
    if (jtxt->type != NBASE_JSON_STRING) {
      free(qvec);
      nbase_json_free(root);
      send_error_json(c, 400, "text must be a string");
      return;
    }
    qtext = jtxt->str;
  }

  if (qvec == NULL && (qtext == NULL || qtext[0] == '\0')) {
    free(qvec);
    nbase_json_free(root);
    send_error_json(c, 400, "query vector or text is required");
    return;
  }

  uint32_t k = 10U;
  const nbase_json_t *jk = nbase_json_obj_get(root, "k");
  if (jk != NULL) {
    if (jk->type != NBASE_JSON_NUMBER || !jk->is_int || jk->i64 <= 0 || jk->i64 > 10000) {
      free(qvec);
      nbase_json_free(root);
      send_error_json(c, 400, "k must be in 1..10000");
      return;
    }
    k = (uint32_t)jk->i64;
  }

  float alpha = 0.5f;
  const nbase_json_t *jalpha = nbase_json_obj_get(root, "alpha");
  if (jalpha != NULL) {
    if (jalpha->type != NBASE_JSON_NUMBER || !isfinite(jalpha->num) || jalpha->num < 0.0 || jalpha->num > 1.0) {
      free(qvec);
      nbase_json_free(root);
      send_error_json(c, 400, "alpha must be in [0, 1]");
      return;
    }
    alpha = (float)jalpha->num;
  }

  float rrf_k = 60.0f;
  const nbase_json_t *jrrf = nbase_json_obj_get(root, "rrfK");
  if (jrrf != NULL) {
    if (jrrf->type != NBASE_JSON_NUMBER || !isfinite(jrrf->num) || jrrf->num <= 0.0) {
      free(qvec);
      nbase_json_free(root);
      send_error_json(c, 400, "rrfK must be > 0");
      return;
    }
    rrf_k = (float)jrrf->num;
  }

  bool inc_meta = false;
  bool inc_vec = false;
  bool inc_text = false;
  if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
    free(qvec);
    nbase_json_free(root);
    return;
  }

  nbase_search_opts_t opts;
  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt;
  if (!parse_search_common_opts(root, &opts, &conds, &meta_flt)) {
    free(conds);
    free(qvec);
    nbase_json_free(root);
    send_error_json(c, 400, "invalid search options");
    return;
  }

  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    free(conds);
    free(qvec);
    nbase_json_free(root);
    send_error_json(c, 500, "failed to read db stats");
    return;
  }

  nbase_scored_id_t *res = malloc((size_t)k * sizeof(nbase_scored_id_t));
  if (!res) {
    free(conds);
    free(qvec);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  uint64_t t0 = monotonic_us();
  uint32_t got = 0U;
  nbase_status_t st = nbase_search_hybrid(
      srv->db, qvec, qdim, qtext, k, alpha, rrf_k, &opts, res, &got);
  uint64_t t1 = monotonic_us();
  record_search_latency(srv, 1ULL, t1 - t0);

  free(conds);
  free(qvec);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    free(res);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"count\":%u,\"results\":[", got);
  append_scored_entries_json(&sb, srv->db, res, got, db_st.dim, inc_meta, inc_vec, inc_text);
  nbase_sbuf_append(&sb, "]}");
  free(res);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_search_hybrid_batch(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jqarr = nbase_json_obj_get(root, "queries");
  if (!jqarr || jqarr->type != NBASE_JSON_ARRAY || jqarr->len == 0U || jqarr->len > 1000U) {
    nbase_json_free(root);
    send_error_json(c, 400, "queries array of 1..1000 objects is required");
    return;
  }

  uint32_t k = 10U;
  const nbase_json_t *jk = nbase_json_obj_get(root, "k");
  if (jk != NULL) {
    if (jk->type != NBASE_JSON_NUMBER || !jk->is_int || jk->i64 <= 0 || jk->i64 > 10000) {
      nbase_json_free(root);
      send_error_json(c, 400, "k must be in 1..10000");
      return;
    }
    k = (uint32_t)jk->i64;
  }

  float alpha = 0.5f;
  const nbase_json_t *jalpha = nbase_json_obj_get(root, "alpha");
  if (jalpha != NULL) {
    if (jalpha->type != NBASE_JSON_NUMBER || !isfinite(jalpha->num) || jalpha->num < 0.0 || jalpha->num > 1.0) {
      nbase_json_free(root);
      send_error_json(c, 400, "alpha must be in [0, 1]");
      return;
    }
    alpha = (float)jalpha->num;
  }

  float rrf_k = 60.0f;
  const nbase_json_t *jrrf = nbase_json_obj_get(root, "rrfK");
  if (jrrf != NULL) {
    if (jrrf->type != NBASE_JSON_NUMBER || !isfinite(jrrf->num) || jrrf->num <= 0.0) {
      nbase_json_free(root);
      send_error_json(c, 400, "rrfK must be > 0");
      return;
    }
    rrf_k = (float)jrrf->num;
  }

  bool inc_meta = false;
  bool inc_vec = false;
  bool inc_text = false;
  if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
    nbase_json_free(root);
    return;
  }

  nbase_search_opts_t opts;
  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt;
  if (!parse_search_common_opts(root, &opts, &conds, &meta_flt)) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 400, "invalid search options");
    return;
  }
  if (!parse_threads_field(root, &opts.num_threads)) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 400, "invalid threads parameter");
    return;
  }

  uint32_t qcnt = (uint32_t)jqarr->len;
  float **qvecs = calloc(qcnt, sizeof(float *));
  uint32_t *qdims = calloc(qcnt, sizeof(uint32_t));
  const char **qtexts = calloc(qcnt, sizeof(const char *));
  nbase_scored_id_t *res = malloc((size_t)k * sizeof(nbase_scored_id_t));
  if (!qvecs || !qdims || !qtexts || !res) {
    free(qvecs);
    free(qdims);
    free(qtexts);
    free(res);
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  const char *item_err = NULL;
  for (uint32_t q = 0U; q < qcnt; q++) {
    const nbase_json_t *item = jqarr->items[q];
    if (!item || item->type != NBASE_JSON_OBJECT) {
      item_err = "each query entry must be a JSON object";
      break;
    }
    const nbase_json_t *iqv = nbase_json_obj_get(item, "query");
    if (iqv != NULL && iqv->type != NBASE_JSON_NULL) {
      if (!parse_float_array(iqv, &qvecs[q], &qdims[q])) {
        item_err = "invalid query vector";
        break;
      }
    }
    const nbase_json_t *itxt = nbase_json_obj_get(item, "text");
    if (itxt != NULL && itxt->type != NBASE_JSON_NULL) {
      if (itxt->type != NBASE_JSON_STRING) {
        item_err = "text must be a string";
        break;
      }
      qtexts[q] = itxt->str;
    }
    if (qvecs[q] == NULL && (qtexts[q] == NULL || qtexts[q][0] == '\0')) {
      item_err = "each query entry requires vector or non-empty text";
      break;
    }
  }
  if (item_err != NULL) {
    for (uint32_t q = 0U; q < qcnt; q++) {
      free(qvecs[q]);
    }
    free(qvecs);
    free(qdims);
    free((void *)qtexts);
    free(res);
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 400, item_err);
    return;
  }

  nbase_stats_t db_st;
  if (nbase_stats(srv->db, &db_st) != NBASE_OK) {
    for (uint32_t q = 0U; q < qcnt; q++) {
      free(qvecs[q]);
    }
    free(qvecs);
    free(qdims);
    free((void *)qtexts);
    free(res);
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "failed to read db stats");
    return;
  }

  bool all_have_vec = true;
  bool none_have_vec = true;
  bool all_dims_match = true;
  for (uint32_t q = 0U; q < qcnt; q++) {
    if (qvecs[q] == NULL) {
      all_have_vec = false;
    } else {
      none_have_vec = false;
      if (qdims[q] != db_st.dim) {
        all_dims_match = false;
      }
    }
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"queryCount\":%u,\"results\":[", qcnt);

  nbase_status_t st = NBASE_OK;
  if (none_have_vec || (all_have_vec && all_dims_match)) {
    float *packed_vecs = NULL;
    if (all_have_vec) {
      packed_vecs = malloc((size_t)qcnt * (size_t)db_st.dim * sizeof(float));
      if (!packed_vecs) {
        st = NBASE_ERR_NOMEM;
      } else {
        for (uint32_t q = 0U; q < qcnt; q++) {
          memcpy(packed_vecs + (size_t)q * (size_t)db_st.dim, qvecs[q], (size_t)db_st.dim * sizeof(float));
        }
      }
    }
    nbase_scored_id_t *res_all =
        (st == NBASE_OK) ? malloc((size_t)qcnt * (size_t)k * sizeof(nbase_scored_id_t)) : NULL;
    uint32_t *counts = (st == NBASE_OK) ? calloc((size_t)qcnt, sizeof(uint32_t)) : NULL;
    if (st == NBASE_OK && (!res_all || !counts)) {
      st = NBASE_ERR_NOMEM;
    }
    if (st == NBASE_OK) {
      uint64_t t0 = monotonic_us();
      st = nbase_search_hybrid_batch(
          srv->db, packed_vecs, db_st.dim, qtexts, qcnt, k, alpha, rrf_k, &opts, res_all, counts);
      uint64_t t1 = monotonic_us();
      record_search_latency(srv, (uint64_t)qcnt, t1 - t0);
      if (st == NBASE_OK) {
        for (uint32_t q = 0U; q < qcnt; q++) {
          nbase_sbuf_append(&sb, q == 0U ? "[" : ",[");
          append_scored_entries_json(
              &sb, srv->db, res_all + (size_t)q * (size_t)k, counts[q], db_st.dim, inc_meta, inc_vec, inc_text);
          nbase_sbuf_append_n(&sb, "]", 1U);
        }
      }
    }
    free(packed_vecs);
    free(res_all);
    free(counts);
  } else {
    uint64_t t0 = monotonic_us();
    for (uint32_t q = 0U; q < qcnt; q++) {
      uint32_t got = 0U;
      st = nbase_search_hybrid(
          srv->db, qvecs[q], qdims[q], qtexts[q], k, alpha, rrf_k, &opts, res, &got);
      if (st != NBASE_OK) {
        break;
      }
      nbase_sbuf_append(&sb, q == 0U ? "[" : ",[");
      append_scored_entries_json(&sb, srv->db, res, got, db_st.dim, inc_meta, inc_vec, inc_text);
      nbase_sbuf_append_n(&sb, "]", 1U);
    }
    uint64_t t1 = monotonic_us();
    record_search_latency(srv, (uint64_t)qcnt, t1 - t0);
  }

  nbase_sbuf_append(&sb, "]}");
  for (uint32_t q = 0U; q < qcnt; q++) {
    free(qvecs[q]);
  }
  free(qvecs);
  free(qdims);
  free((void *)qtexts);
  free(res);
  free(conds);
  nbase_json_free(root);

  if (st != NBASE_OK) {
    nbase_sbuf_free(&sb);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_search_relationships(const http_conn_t *c, const char *body, size_t body_len) {
  uint32_t k_neighbors = 5U;
  float max_distance = 1.0f;
  nbase_json_t *root = NULL;
  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt = {NULL, 0U};
  bool has_filter = false;

  if (body_len > 0U && body != NULL) {
    if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
      nbase_json_free(root);
      send_error_json(c, 400, "invalid JSON request body");
      return;
    }
    const nbase_json_t *jkn = nbase_json_obj_get(root, "kNeighbors");
    if (jkn != NULL) {
      if (jkn->type != NBASE_JSON_NUMBER || !jkn->is_int || jkn->i64 <= 0 || jkn->i64 > 1000) {
        nbase_json_free(root);
        send_error_json(c, 400, "kNeighbors must be in 1..1000");
        return;
      }
      k_neighbors = (uint32_t)jkn->i64;
    }
    const nbase_json_t *jmd = nbase_json_obj_get(root, "maxDistance");
    if (jmd != NULL) {
      if (jmd->type != NBASE_JSON_NUMBER || !isfinite(jmd->num)) {
        nbase_json_free(root);
        send_error_json(c, 400, "maxDistance must be a finite number");
        return;
      }
      max_distance = (float)jmd->num;
    }
    const nbase_json_t *jflt = nbase_json_obj_get(root, "filters");
    if (jflt != NULL) {
      if (parse_meta_filter_json(jflt, &conds, &meta_flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(root);
        send_error_json(c, 400, "invalid metadata filters");
        return;
      }
      has_filter = (meta_flt.count > 0U);
    }
  }

  nbase_edge_t *edges = NULL;
  uint64_t edge_cnt = 0ULL;
  nbase_status_t st = nbase_extract_relationships_filtered(
      c->srv->db, k_neighbors, max_distance, has_filter ? &meta_flt : NULL, &edges, &edge_cnt);
  free(conds);
  nbase_json_free(root);
  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"count\":%llu,\"edges\":[", (unsigned long long)edge_cnt);
  for (uint64_t i = 0ULL; i < edge_cnt; i++) {
    nbase_sbuf_appendf(
        &sb,
        i == 0ULL ? "{\"src\":%llu,\"dst\":%llu,\"weight\":%.7g}"
                  : ",{\"src\":%llu,\"dst\":%llu,\"weight\":%.7g}",
        (unsigned long long)edges[i].src_id,
        (unsigned long long)edges[i].dst_id,
        (double)edges[i].weight);
  }
  nbase_sbuf_append(&sb, "]}");
  nbase_free_edges(edges);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_search_communities(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  uint32_t k_neighbors = 5U;
  float max_distance = 1.0f;
  nbase_json_t *root = NULL;
  nbase_meta_cond_t *conds = NULL;
  nbase_meta_filter_t meta_flt = {NULL, 0U};
  bool has_filter = false;

  if (body_len > 0U && body != NULL) {
    if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
      nbase_json_free(root);
      send_error_json(c, 400, "invalid JSON request body");
      return;
    }
    const nbase_json_t *jkn = nbase_json_obj_get(root, "kNeighbors");
    if (jkn != NULL) {
      if (jkn->type != NBASE_JSON_NUMBER || !jkn->is_int || jkn->i64 <= 0 || jkn->i64 > 1000) {
        nbase_json_free(root);
        send_error_json(c, 400, "kNeighbors must be in 1..1000");
        return;
      }
      k_neighbors = (uint32_t)jkn->i64;
    }
    const nbase_json_t *jmd = nbase_json_obj_get(root, "maxDistance");
    if (jmd != NULL) {
      if (jmd->type != NBASE_JSON_NUMBER || !isfinite(jmd->num)) {
        nbase_json_free(root);
        send_error_json(c, 400, "maxDistance must be a finite number");
        return;
      }
      max_distance = (float)jmd->num;
    }
    const nbase_json_t *jflt = nbase_json_obj_get(root, "filters");
    if (jflt != NULL) {
      if (parse_meta_filter_json(jflt, &conds, &meta_flt) != NBASE_OK) {
        free(conds);
        nbase_json_free(root);
        send_error_json(c, 400, "invalid metadata filters");
        return;
      }
      has_filter = (meta_flt.count > 0U);
    }
  }

  nbase_stats_t db_st;
  nbase_status_t st = nbase_stats(srv->db, &db_st);
  if (st != NBASE_OK) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, nbase_strerror(st));
    return;
  }
  if (db_st.active_vectors == 0ULL) {
    free(conds);
    nbase_json_free(root);
    send_json_response(c, 200, "{\"success\":true,\"community_count\":0,\"communities\":[]}", 52U);
    return;
  }
  if (db_st.active_vectors > 1000000ULL) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 400, "too many nodes for single HTTP community response");
    return;
  }

  uint32_t cap = (uint32_t)db_st.active_vectors + 64U;
  uint64_t *ids = malloc((size_t)cap * sizeof(uint64_t));
  if (!ids) {
    free(conds);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }
  uint32_t got_ids = 0U;
  st = nbase_query_meta(srv->db, has_filter ? &meta_flt : NULL, cap, ids, &got_ids);
  if (st != NBASE_OK) {
    free(conds);
    nbase_json_free(root);
    free(ids);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }
  if (got_ids == 0U) {
    free(conds);
    nbase_json_free(root);
    free(ids);
    send_json_response(c, 200, "{\"success\":true,\"community_count\":0,\"communities\":[]}", 52U);
    return;
  }

  nbase_edge_t *edges = NULL;
  uint64_t edge_cnt = 0ULL;
  st = nbase_extract_relationships_filtered(
      srv->db, k_neighbors, max_distance, has_filter ? &meta_flt : NULL, &edges, &edge_cnt);
  free(conds);
  nbase_json_free(root);
  if (st != NBASE_OK) {
    free(ids);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  uint32_t *labels = malloc((size_t)(got_ids > 0U ? got_ids : 1U) * sizeof(uint32_t));
  if (!labels) {
    nbase_free_edges(edges);
    free(ids);
    send_error_json(c, 500, "out of memory");
    return;
  }
  uint32_t comm_cnt = 0U;
  st = nbase_extract_communities(ids, got_ids, edges, edge_cnt, labels, &comm_cnt);
  nbase_free_edges(edges);
  if (st != NBASE_OK) {
    free(labels);
    free(ids);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"community_count\":%u,\"communities\":[", comm_cnt);
  for (uint32_t i = 0U; i < got_ids; i++) {
    nbase_sbuf_appendf(
        &sb,
        i == 0U ? "{\"id\":%llu,\"community\":%u}" : ",{\"id\":%llu,\"community\":%u}",
        (unsigned long long)ids[i],
        labels[i]);
  }
  nbase_sbuf_append(&sb, "]}");
  free(labels);
  free(ids);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static bool rate_limit_allow(nbase_http_server_t *srv) {
  if (srv->max_rps == 0U) {
    return true;
  }
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  uint64_t now_sec = (uint64_t)ts.tv_sec;
  pthread_mutex_lock(&srv->rl_lock);
  if (now_sec != srv->rl_sec) {
    srv->rl_sec = now_sec;
    srv->rl_count = 0U;
  }
  bool ok = (srv->rl_count < srv->max_rps);
  if (ok) {
    srv->rl_count++;
  }
  pthread_mutex_unlock(&srv->rl_lock);
  return ok;
}

static bool parse_json_u64_id_array(
    const http_conn_t *c,
    const nbase_json_t *jids,
    uint64_t **out_ids,
    uint32_t *out_cnt) {
  *out_ids = NULL;
  *out_cnt = 0U;
  if (!jids || jids->type != NBASE_JSON_ARRAY || jids->len > 100000U) {
    send_error_json(c, 400, "ids must be an array of at most 100000 uint64 values");
    return false;
  }
  uint32_t cnt = (uint32_t)jids->len;
  uint64_t *ids = (cnt > 0U) ? malloc((size_t)cnt * sizeof(uint64_t)) : NULL;
  if (cnt > 0U && !ids) {
    send_error_json(c, 500, "out of memory");
    return false;
  }
  for (uint32_t i = 0U; i < cnt; i++) {
    if (!parse_json_u64(jids->items[i], &ids[i])) {
      free(ids);
      send_error_json(c, 400, "each id must be an unsigned 64-bit integer");
      return false;
    }
  }
  *out_ids = ids;
  *out_cnt = cnt;
  return true;
}

static void handle_delete_vectors(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jids = nbase_json_obj_get(root, "ids");
  const nbase_json_t *jflt = nbase_json_obj_get(root, "filters");
  uint32_t deleted = 0U;
  nbase_status_t st = NBASE_OK;

  if (jids != NULL && jids->type != NBASE_JSON_NULL) {
    uint64_t *ids = NULL;
    uint32_t cnt = 0U;
    if (!parse_json_u64_id_array(c, jids, &ids, &cnt)) {
      nbase_json_free(root);
      return;
    }
    st = nbase_delete_batch(srv->db, ids, cnt, &deleted);
    free(ids);
  } else if (jflt != NULL && jflt->type != NBASE_JSON_NULL) {
    nbase_meta_cond_t *conds = NULL;
    nbase_meta_filter_t meta_flt;
    if (parse_meta_filter_json(jflt, &conds, &meta_flt) != NBASE_OK || meta_flt.count == 0U) {
      free(conds);
      nbase_json_free(root);
      send_error_json(c, 400, "non-empty metadata filters required");
      return;
    }
    st = nbase_delete_by_filter(srv->db, &meta_flt, &deleted);
    free(conds);
  } else {
    nbase_json_free(root);
    send_error_json(c, 400, "ids array or filters is required");
    return;
  }

  nbase_json_free(root);
  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"deleted\":%u}", deleted);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

typedef nbase_status_t (*http_batch_delete_ids_fn)(
    nbase_db_t *db,
    const uint64_t *ids,
    uint32_t count,
    uint32_t *out_deleted);

static void handle_delete_ids_attr_batch(
    const http_conn_t *c,
    const char *body,
    size_t body_len,
    http_batch_delete_ids_fn fn) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jids = nbase_json_obj_get(root, "ids");
  uint64_t *ids = NULL;
  uint32_t cnt = 0U;
  if (!parse_json_u64_id_array(c, jids, &ids, &cnt)) {
    nbase_json_free(root);
    return;
  }

  uint32_t deleted = 0U;
  nbase_status_t st = fn(srv->db, ids, cnt, &deleted);
  free(ids);
  nbase_json_free(root);
  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"deleted\":%u}", deleted);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_delete_text_batch(const http_conn_t *c, const char *body, size_t body_len) {
  handle_delete_ids_attr_batch(c, body, body_len, nbase_delete_text_batch);
}

static void handle_delete_meta_batch(const http_conn_t *c, const char *body, size_t body_len) {
  handle_delete_ids_attr_batch(c, body, body_len, nbase_delete_meta_batch);
}

static void handle_get_ids_attr_batch(
    const http_conn_t *c,
    const char *body,
    size_t body_len,
    bool is_meta) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jids = nbase_json_obj_get(root, "ids");
  uint64_t *ids = NULL;
  uint32_t cnt = 0U;
  if (!parse_json_u64_id_array(c, jids, &ids, &cnt)) {
    nbase_json_free(root);
    return;
  }
  nbase_json_free(root);

  char **texts = NULL;
  bool *found = NULL;
  nbase_status_t st = NBASE_OK;
  if (!is_meta && cnt > 0U) {
    texts = calloc(cnt, sizeof(char *));
    found = calloc(cnt, sizeof(bool));
    if (!texts || !found) {
      st = NBASE_ERR_NOMEM;
    } else {
      st = nbase_get_text_batch(srv->db, ids, cnt, texts, found);
    }
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_append(&sb, "{\"success\":true,\"results\":[");

  for (uint32_t i = 0U; i < cnt && st == NBASE_OK && !sb.oom; i++) {
    nbase_sbuf_appendf(&sb, i == 0U ? "{\"id\":%llu," : ",{\"id\":%llu,", (unsigned long long)ids[i]);
    if (is_meta) {
      if (!nbase_contains(srv->db, ids[i])) {
        nbase_sbuf_append(&sb, "\"found\":false,\"metadata\":null}");
      } else {
        nbase_sbuf_append(&sb, "\"found\":true,\"metadata\":");
        append_meta_json(&sb, srv->db, ids[i]);
        nbase_sbuf_append_n(&sb, "}", 1U);
      }
    } else if (found[i] && texts[i] != NULL) {
      nbase_sbuf_append(&sb, "\"found\":true,\"text\":");
      nbase_sbuf_append_json_str(&sb, texts[i]);
      nbase_sbuf_append_n(&sb, "}", 1U);
    } else {
      nbase_sbuf_append(&sb, "\"found\":false,\"text\":null}");
    }
  }

  nbase_free_texts(texts, cnt);
  free(texts);
  free(found);
  free(ids);
  if (st != NBASE_OK || sb.oom) {
    nbase_sbuf_free(&sb);
    send_error_json(c, http_code_for_status(st != NBASE_OK ? st : NBASE_ERR_NOMEM), nbase_strerror(st != NBASE_OK ? st : NBASE_ERR_NOMEM));
    return;
  }
  nbase_sbuf_append(&sb, "]}");
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_distance(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  nbase_metric_t metric = NBASE_METRIC_EUCLIDEAN;
  const nbase_json_t *jmetric = nbase_json_obj_get(root, "metric");
  if (jmetric != NULL) {
    if (jmetric->type != NBASE_JSON_STRING || !parse_metric_str(jmetric->str, &metric)) {
      nbase_json_free(root);
      send_error_json(c, 400, "invalid metric");
      return;
    }
  }

  float *query = NULL;
  uint32_t dim = 0U;
  if (!parse_float_array(nbase_json_obj_get(root, "query"), &query, &dim)) {
    nbase_json_free(root);
    send_error_json(c, 400, "missing or invalid query vector");
    return;
  }

  const nbase_json_t *jvecs = nbase_json_obj_get(root, "vectors");
  if (!jvecs || jvecs->type != NBASE_JSON_ARRAY || jvecs->len > 10000U) {
    free(query);
    nbase_json_free(root);
    send_error_json(c, 400, "vectors must be an array with at most 10000 elements");
    return;
  }

  uint32_t count = (uint32_t)jvecs->len;
  if (count == 0U) {
    free(query);
    nbase_json_free(root);
    nbase_sbuf_t sb;
    nbase_sbuf_init(&sb);
    nbase_sbuf_appendf(
        &sb,
        "{\"success\":true,\"metric\":\"%s\",\"dimension\":%u,\"count\":0,\"distances\":[]}",
        metric_name_str(metric),
        dim);
    send_sbuf_json(c, 200, &sb);
    nbase_sbuf_free(&sb);
    return;
  }

  if ((size_t)count > SIZE_MAX / ((size_t)dim * sizeof(float))) {
    free(query);
    nbase_json_free(root);
    send_error_json(c, 400, "vectors payload too large");
    return;
  }

  float *flat_vecs = malloc((size_t)count * (size_t)dim * sizeof(float));
  float *dists = malloc((size_t)count * sizeof(float));
  if (!flat_vecs || !dists) {
    free(flat_vecs);
    free(dists);
    free(query);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  for (uint32_t n = 0U; n < count; n++) {
    const nbase_json_t *row = jvecs->items[n];
    if (!row || row->type != NBASE_JSON_ARRAY || row->len != (size_t)dim) {
      free(flat_vecs);
      free(dists);
      free(query);
      nbase_json_free(root);
      send_error_json(c, 400, "each vector must match query dimension");
      return;
    }
    float *dst = flat_vecs + (size_t)n * (size_t)dim;
    for (uint32_t i = 0U; i < dim; i++) {
      const nbase_json_t *it = row->items[i];
      if (!it || it->type != NBASE_JSON_NUMBER || !isfinite(it->num)) {
        free(flat_vecs);
        free(dists);
        free(query);
        nbase_json_free(root);
        send_error_json(c, 400, "vectors must contain finite numbers");
        return;
      }
      float f = (float)it->num;
      if (!isfinite(f)) {
        free(flat_vecs);
        free(dists);
        free(query);
        nbase_json_free(root);
        send_error_json(c, 400, "vectors must contain finite float32 values");
        return;
      }
      dst[i] = f;
    }
  }

  nbase_status_t st = nbase_distance_batch(metric, query, flat_vecs, count, dim, dists);
  free(flat_vecs);
  free(query);
  nbase_json_free(root);
  if (st != NBASE_OK) {
    free(dists);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(
      &sb,
      "{\"success\":true,\"metric\":\"%s\",\"dimension\":%u,\"count\":%u,\"distances\":[",
      metric_name_str(metric),
      dim,
      count);
  for (uint32_t i = 0U; i < count; i++) {
    nbase_sbuf_appendf(&sb, i == 0U ? "%.7g" : ",%.7g", (double)dists[i]);
  }
  nbase_sbuf_append(&sb, "]}");
  free(dists);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_rerank(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  nbase_json_t *root = NULL;
  if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
    nbase_json_free(root);
    send_error_json(c, 400, "invalid JSON request body");
    return;
  }

  const nbase_json_t *jcands = nbase_json_obj_get(root, "candidates");
  if (!jcands || jcands->type != NBASE_JSON_ARRAY || jcands->len > 10000U) {
    nbase_json_free(root);
    send_error_json(c, 400, "candidates must be an array of at most 10000 items");
    return;
  }
  uint32_t cand_cnt = (uint32_t)jcands->len;

  int rerank_method = 0; /* 0=unset, 1=diversity, 2=weighted, 3=standard */
  const nbase_json_t *jrm = nbase_json_obj_get(root, "method");
  if (jrm == NULL) {
    jrm = nbase_json_obj_get(root, "rerankingMethod");
  }
  if (jrm == NULL) {
    jrm = nbase_json_obj_get(root, "rerankMethod");
  }
  if (jrm != NULL) {
    if (jrm->type != NBASE_JSON_STRING) {
      nbase_json_free(root);
      send_error_json(c, 400, "method must be a string");
      return;
    }
    if (strcmp(jrm->str, "diversity") == 0) {
      rerank_method = 1;
    } else if (strcmp(jrm->str, "weighted") == 0) {
      rerank_method = 2;
    } else if (strcmp(jrm->str, "standard") == 0) {
      rerank_method = 3;
    } else {
      nbase_json_free(root);
      send_error_json(c, 400, "method must be standard, diversity, or weighted");
      return;
    }
  }

  nbase_rerank_weight_t *weights = NULL;
  uint32_t weight_cnt = 0U;
  const nbase_json_t *jrw = nbase_json_obj_get(root, "weights");
  if (jrw == NULL) {
    jrw = nbase_json_obj_get(root, "rerankWeights");
  }
  if (jrw != NULL) {
    const char *w_err = NULL;
    int w_code = 400;
    if (!parse_rerank_weights_obj(jrw, &weights, &weight_cnt, &w_err, &w_code)) {
      nbase_json_free(root);
      send_error_json(c, w_code, w_err);
      return;
    }
  }
  if (rerank_method == 0) {
    rerank_method = (weight_cnt > 0U) ? 2 : 1;
  }

  float lambda = 0.5f;
  const nbase_json_t *jlam = nbase_json_obj_get(root, "lambda");
  if (jlam == NULL) {
    jlam = nbase_json_obj_get(root, "rerankLambda");
  }
  if (jlam != NULL) {
    if (jlam->type != NBASE_JSON_NUMBER || !isfinite(jlam->num) ||
        jlam->num < 0.0 || jlam->num > 1.0) {
      free(weights);
      nbase_json_free(root);
      send_error_json(c, 400, "lambda must be in [0, 1]");
      return;
    }
    lambda = (float)jlam->num;
  }

  uint32_t k = (cand_cnt > 0U) ? cand_cnt : 10U;
  const nbase_json_t *jk = nbase_json_obj_get(root, "k");
  if (jk != NULL) {
    if (jk->type != NBASE_JSON_NUMBER || !jk->is_int || jk->i64 <= 0 || jk->i64 > 10000) {
      free(weights);
      nbase_json_free(root);
      send_error_json(c, 400, "k must be in 1..10000");
      return;
    }
    k = (uint32_t)jk->i64;
  }

  bool inc_meta = false;
  bool inc_vec = false;
  bool inc_text = false;
  if (!parse_include_flags(c, root, &inc_meta, &inc_vec, &inc_text)) {
    free(weights);
    nbase_json_free(root);
    return;
  }

  nbase_result_t *cands = (cand_cnt > 0U) ? malloc((size_t)cand_cnt * sizeof(nbase_result_t)) : NULL;
  nbase_result_t *out_res = (cand_cnt > 0U) ? malloc((size_t)k * sizeof(nbase_result_t)) : NULL;
  if (cand_cnt > 0U && (!cands || !out_res)) {
    free(cands);
    free(out_res);
    free(weights);
    nbase_json_free(root);
    send_error_json(c, 500, "out of memory");
    return;
  }

  for (uint32_t i = 0U; i < cand_cnt; i++) {
    const nbase_json_t *item = jcands->items[i];
    if (!item || item->type != NBASE_JSON_OBJECT) {
      free(cands);
      free(out_res);
      free(weights);
      nbase_json_free(root);
      send_error_json(c, 400, "each candidate must be a JSON object");
      return;
    }
    if (!parse_json_u64(nbase_json_obj_get(item, "id"), &cands[i].id)) {
      free(cands);
      free(out_res);
      free(weights);
      nbase_json_free(root);
      send_error_json(c, 400, "each candidate requires an unsigned 64-bit id");
      return;
    }
    const nbase_json_t *jd = nbase_json_obj_get(item, "distance");
    if (!jd || jd->type != NBASE_JSON_NUMBER || !isfinite(jd->num)) {
      free(cands);
      free(out_res);
      free(weights);
      nbase_json_free(root);
      send_error_json(c, 400, "each candidate requires a finite distance");
      return;
    }
    float f = (float)jd->num;
    if (!isfinite(f)) {
      free(cands);
      free(out_res);
      free(weights);
      nbase_json_free(root);
      send_error_json(c, 400, "candidate distance must be a finite float32");
      return;
    }
    cands[i].distance = f;
  }

  nbase_status_t st = NBASE_OK;
  uint32_t out_n = 0U;
  if (cand_cnt > 0U) {
    if (rerank_method == 2) {
      st = nbase_rerank_weighted(srv->db, cands, cand_cnt, weights, weight_cnt, k, out_res, &out_n);
    } else if (rerank_method == 3) {
      for (uint32_t i = 0U; i < cand_cnt; i++) {
        if (!nbase_contains(srv->db, cands[i].id)) {
          st = NBASE_ERR_NOT_FOUND;
          break;
        }
      }
      if (st == NBASE_OK) {
        qsort(cands, cand_cnt, sizeof(nbase_result_t), nbase_cmp_result_asc);
        out_n = (cand_cnt < k) ? cand_cnt : k;
        if (out_n > 0U) {
          memcpy(out_res, cands, (size_t)out_n * sizeof(nbase_result_t));
        }
      }
    } else {
      st = nbase_rerank_mmr(srv->db, cands, cand_cnt, k, lambda, out_res, &out_n);
    }
  }

  free(cands);
  free(weights);
  nbase_json_free(root);
  if (st != NBASE_OK) {
    free(out_res);
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  uint32_t vec_dim = 0U;
  if (inc_vec && out_n > 0U) {
    nbase_stats_t db_st;
    st = nbase_stats(srv->db, &db_st);
    if (st != NBASE_OK) {
      free(out_res);
      send_error_json(c, http_code_for_status(st), nbase_strerror(st));
      return;
    }
    vec_dim = db_st.dim;
  }

  const char *method_str =
      (rerank_method == 2) ? "weighted" : ((rerank_method == 3) ? "standard" : "diversity");
  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(&sb, "{\"success\":true,\"method\":\"%s\",\"count\":%u,\"results\":[",
                     method_str,
                     out_n);
  for (uint32_t i = 0U; i < out_n; i++) {
    nbase_sbuf_appendf(
        &sb,
        i == 0U ? "{\"id\":%llu,\"distance\":%.7g" : ",{\"id\":%llu,\"distance\":%.7g",
        (unsigned long long)out_res[i].id,
        (double)out_res[i].distance);
    if (inc_meta) {
      nbase_sbuf_append(&sb, ",\"metadata\":");
      append_meta_json(&sb, srv->db, out_res[i].id);
    }
    if (inc_vec) {
      nbase_sbuf_append(&sb, ",\"vector\":");
      append_vector_json(&sb, srv->db, out_res[i].id, vec_dim);
    }
    if (inc_text) {
      nbase_sbuf_append(&sb, ",\"text\":");
      append_text_json(&sb, srv->db, out_res[i].id);
    }
    nbase_sbuf_append_n(&sb, "}", 1U);
  }
  nbase_sbuf_append(&sb, "]}");
  free(out_res);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void handle_admin_cache(const http_conn_t *c, const char *body, size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  bool has_max = false;
  uint32_t max_entries = 0U;
  bool do_clear = false;

  if (body != NULL && body_len > 0U) {
    nbase_json_t *root = NULL;
    if (nbase_json_parse(body, body_len, &root) != NBASE_OK || root->type != NBASE_JSON_OBJECT) {
      nbase_json_free(root);
      send_error_json(c, 400, "invalid JSON request body");
      return;
    }
    const nbase_json_t *jmax = nbase_json_obj_get(root, "max_entries");
    if (jmax != NULL) {
      if (jmax->type != NBASE_JSON_NUMBER || !jmax->is_int || jmax->i64 < 0 ||
          (uint64_t)jmax->i64 > (uint64_t)UINT32_MAX) {
        nbase_json_free(root);
        send_error_json(c, 400, "max_entries must be a non-negative uint32 integer");
        return;
      }
      has_max = true;
      max_entries = (uint32_t)jmax->i64;
    }
    const nbase_json_t *jclr = nbase_json_obj_get(root, "clear");
    if (jclr != NULL) {
      if (jclr->type != NBASE_JSON_BOOL) {
        nbase_json_free(root);
        send_error_json(c, 400, "clear must be boolean");
        return;
      }
      do_clear = jclr->b;
    }
    nbase_json_free(root);
  }

  nbase_status_t st = NBASE_OK;
  if (has_max) {
    st = nbase_cache_configure(srv->db, max_entries);
  }
  if (st == NBASE_OK && do_clear) {
    st = nbase_cache_clear(srv->db);
  }
  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_stats_t db_st;
  st = nbase_stats(srv->db, &db_st);
  if (st != NBASE_OK) {
    send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    return;
  }

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);
  nbase_sbuf_appendf(
      &sb,
      "{\"success\":true,\"cache_capacity\":%u,\"cache_entries\":%u}",
      db_st.cache_capacity,
      db_st.cache_entries);
  send_sbuf_json(c, 200, &sb);
  nbase_sbuf_free(&sb);
}

static void dispatch_request(
    const http_conn_t *c,
    const char *method,
    const char *path,
    const char *query_str,
    const char *body,
    size_t body_len) {
  nbase_http_server_t *srv = c->srv;
  if (strcmp(path, "/health") == 0) {
    if (strcmp(method, "GET") == 0) {
      handle_health(c);
      return;
    }
    if (strcmp(method, "HEAD") == 0) {
      send_http_response(c, 200, "application/json", "", 0U);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(method, "GET") == 0 && strcmp(path, "/metrics") == 0) {
    handle_metrics(c);
    return;
  }
  if (strcmp(method, "GET") == 0 && strcmp(path, "/stats") == 0) {
    handle_stats(c);
    return;
  }
  if (strcmp(method, "GET") == 0 && strcmp(path, "/api/vectors") == 0) {
    handle_list_vectors(c, query_str);
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/vectors") == 0) {
    handle_post_vectors(c, body, body_len);
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/vectors/upsert") == 0) {
    handle_upsert_vectors(c, body, body_len);
    return;
  }
  if (strcmp(path, "/api/vectors/patch") == 0) {
    if (strcmp(method, "PATCH") == 0 || strcmp(method, "POST") == 0) {
      handle_patch_vectors_batch(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/vectors/get") == 0) {
    handle_get_vectors_batch(c, body, body_len);
    return;
  }
  if (strcmp(path, "/api/vectors/exists") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_vectors_exists(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if ((strcmp(method, "POST") == 0 && strcmp(path, "/api/vectors/delete") == 0) ||
      (strcmp(method, "DELETE") == 0 && strcmp(path, "/api/vectors") == 0)) {
    handle_delete_vectors(c, body, body_len);
    return;
  }
  if (strcmp(path, "/api/vectors/metadata") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_batch_metadata(c, body, body_len, false);
      return;
    }
    if (strcmp(method, "PATCH") == 0) {
      handle_batch_metadata(c, body, body_len, true);
      return;
    }
    if (strcmp(method, "DELETE") == 0) {
      handle_delete_meta_batch(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(path, "/api/vectors/metadata/get") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_get_ids_attr_batch(c, body, body_len, true);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(path, "/api/vectors/metadata/delete") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_delete_meta_batch(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(path, "/api/vectors/text") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_batch_text(c, body, body_len);
      return;
    }
    if (strcmp(method, "DELETE") == 0) {
      handle_delete_text_batch(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(path, "/api/vectors/text/get") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_get_ids_attr_batch(c, body, body_len, false);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(path, "/api/vectors/text/delete") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_delete_text_batch(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(path, "/api/vectors/count") == 0) {
    if (strcmp(method, "GET") == 0) {
      handle_count_vectors(c, NULL, 0U);
      return;
    }
    if (strcmp(method, "POST") == 0) {
      handle_count_vectors(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(path, "/api/vectors/export") == 0) {
    if (strcmp(method, "GET") == 0) {
      handle_export_vectors(c, query_str);
      return;
    }
    if (strcmp(method, "POST") == 0) {
      handle_export_vectors_post(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(path, "/api/vectors/import") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_import_vectors(c, query_str, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/search") == 0) {
    handle_search(c, body, body_len);
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/search/batch") == 0) {
    handle_search_batch(c, body, body_len);
    return;
  }
  if (strcmp(path, "/api/search/similar") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_search_similar_batch(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(path, "/api/search/radius") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_search_radius(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/search/metadata") == 0) {
    handle_search_metadata(c, body, body_len);
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/search/text") == 0) {
    handle_search_text(c, body, body_len);
    return;
  }
  if (strcmp(path, "/api/search/text/batch") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_search_text_batch(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/search/hybrid") == 0) {
    handle_search_hybrid(c, body, body_len);
    return;
  }
  if (strcmp(path, "/api/search/hybrid/batch") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_search_hybrid_batch(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/search/relationships") == 0) {
    handle_search_relationships(c, body, body_len);
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/search/communities") == 0) {
    handle_search_communities(c, body, body_len);
    return;
  }
  if (strcmp(path, "/api/distance") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_distance(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(path, "/api/rerank") == 0) {
    if (strcmp(method, "POST") == 0) {
      handle_rerank(c, body, body_len);
      return;
    }
    send_error_json(c, 405, "method not allowed");
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/admin/save") == 0) {
    if (!srv->save_dir) {
      send_error_json(c, 400, "save_dir is not configured");
      return;
    }
    nbase_status_t st = nbase_save(srv->db, srv->save_dir);
    if (st != NBASE_OK) {
      send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    } else {
      send_json_response(c, 200, "{\"success\":true}", 16U);
    }
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/admin/checkpoint") == 0) {
    nbase_status_t st = nbase_wal_checkpoint(srv->db);
    if (st != NBASE_OK) {
      send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    } else {
      send_json_response(c, 200, "{\"success\":true}", 16U);
    }
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/admin/compact") == 0) {
    nbase_status_t st = nbase_compact(srv->db);
    if (st != NBASE_OK) {
      send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    } else {
      send_json_response(c, 200, "{\"success\":true}", 16U);
    }
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/admin/rebuild") == 0) {
    nbase_status_t st = nbase_rebuild_hnsw_mt(srv->db, srv->num_workers);
    if (st != NBASE_OK) {
      send_error_json(c, http_code_for_status(st), nbase_strerror(st));
    } else {
      send_json_response(c, 200, "{\"success\":true}", 16U);
    }
    return;
  }
  if (strcmp(method, "POST") == 0 && strcmp(path, "/api/admin/cache") == 0) {
    handle_admin_cache(c, body, body_len);
    return;
  }

  static const char prefix[] = "/api/vectors/";
  size_t plen = sizeof(prefix) - 1U;
  if (strncmp(path, prefix, plen) == 0) {
    const char *rest = path + plen;
    const char *slash = strchr(rest, '/');
    size_t id_len = slash ? (size_t)(slash - rest) : strlen(rest);
    uint64_t id = 0ULL;
    if (!parse_u64_str(rest, id_len, &id)) {
      send_error_json(c, 400, "invalid vector id");
      return;
    }

    if (!slash) {
      if (strcmp(method, "GET") == 0) {
        handle_get_vector_id(c, id, query_str);
        return;
      }
      if (strcmp(method, "HEAD") == 0) {
        bool ex = nbase_contains(srv->db, id);
        send_http_response(c, ex ? 200 : 404, "application/json", "", 0U);
        return;
      }
      if (strcmp(method, "PUT") == 0) {
        handle_put_vector_id(c, id, body, body_len);
        return;
      }
      if (strcmp(method, "PATCH") == 0) {
        handle_patch_vector_id(c, id, body, body_len);
        return;
      }
      if (strcmp(method, "DELETE") == 0) {
        nbase_status_t st = nbase_delete(srv->db, id);
        if (st != NBASE_OK) {
          send_error_json(c, http_code_for_status(st), nbase_strerror(st));
        } else {
          send_json_response(c, 200, "{\"success\":true}", 16U);
        }
        return;
      }
      send_error_json(c, 405, "method not allowed");
      return;
    }

    if (strcmp(slash, "/exists") == 0 && strcmp(method, "GET") == 0) {
      bool exists = nbase_contains(srv->db, id);
      if (exists) {
        send_json_response(c, 200, "{\"success\":true,\"exists\":true}", 30U);
      } else {
        send_json_response(c, 200, "{\"success\":true,\"exists\":false}", 31U);
      }
      return;
    }
    if (strcmp(slash, "/similar") == 0) {
      if (strcmp(method, "GET") == 0) {
        handle_get_similar(c, id, query_str);
        return;
      }
      if (strcmp(method, "POST") == 0) {
        handle_post_similar(c, id, body, body_len);
        return;
      }
      send_error_json(c, 405, "method not allowed");
      return;
    }
    if (strcmp(slash, "/metadata") == 0) {
      if (strcmp(method, "HEAD") == 0) {
        bool ex = nbase_contains(srv->db, id);
        send_http_response(c, ex ? 200 : 404, "application/json", "", 0U);
        return;
      }
      if (strcmp(method, "GET") == 0) {
        if (!nbase_contains(srv->db, id)) {
          send_error_json(c, 404, nbase_strerror(NBASE_ERR_NOT_FOUND));
          return;
        }
        nbase_sbuf_t sb;
        nbase_sbuf_init(&sb);
        nbase_sbuf_appendf(&sb, "{\"success\":true,\"id\":%llu,\"metadata\":", (unsigned long long)id);
        append_meta_json(&sb, srv->db, id);
        nbase_sbuf_append_n(&sb, "}", 1U);
        send_sbuf_json(c, 200, &sb);
        nbase_sbuf_free(&sb);
        return;
      }
      if (strcmp(method, "PUT") == 0) {
        handle_id_metadata_write(c, id, body, body_len, false);
        return;
      }
      if (strcmp(method, "PATCH") == 0) {
        handle_id_metadata_write(c, id, body, body_len, true);
        return;
      }
      if (strcmp(method, "DELETE") == 0) {
        nbase_status_t st = nbase_delete_meta(srv->db, id);
        if (st != NBASE_OK) {
          send_error_json(c, http_code_for_status(st), nbase_strerror(st));
        } else {
          send_json_response(c, 200, "{\"success\":true}", 16U);
        }
        return;
      }
      send_error_json(c, 405, "method not allowed");
      return;
    }
    if (strcmp(slash, "/text") == 0) {
      if (strcmp(method, "HEAD") == 0) {
        size_t tlen = 0U;
        bool ex = nbase_contains(srv->db, id) &&
                  nbase_get_text(srv->db, id, NULL, 0U, &tlen) == NBASE_OK;
        send_http_response(c, ex ? 200 : 404, "application/json", "", 0U);
        return;
      }
      if (strcmp(method, "GET") == 0) {
        if (!nbase_contains(srv->db, id)) {
          send_error_json(c, 404, nbase_strerror(NBASE_ERR_NOT_FOUND));
          return;
        }
        nbase_sbuf_t sb;
        nbase_sbuf_init(&sb);
        nbase_sbuf_appendf(&sb, "{\"success\":true,\"id\":%llu,\"text\":", (unsigned long long)id);
        append_text_json(&sb, srv->db, id);
        nbase_sbuf_append_n(&sb, "}", 1U);
        send_sbuf_json(c, 200, &sb);
        nbase_sbuf_free(&sb);
        return;
      }
      if (strcmp(method, "PUT") == 0) {
        handle_put_vector_text(c, id, body, body_len);
        return;
      }
      if (strcmp(method, "DELETE") == 0) {
        nbase_status_t st = nbase_delete_text(srv->db, id);
        if (st != NBASE_OK) {
          send_error_json(c, http_code_for_status(st), nbase_strerror(st));
        } else {
          send_json_response(c, 200, "{\"success\":true}", 16U);
        }
        return;
      }
      send_error_json(c, 405, "method not allowed");
      return;
    }
  }

  send_error_json(c, 404, "not found");
}

static bool hdr_has_token_ci(const char *val, const char *tok) {
  size_t tlen = strlen(tok);
  const char *p = val;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t' || *p == ',') {
      p++;
    }
    if (*p == '\0') {
      break;
    }
    const char *start = p;
    while (*p != '\0' && *p != ',') {
      p++;
    }
    const char *end = p;
    while (end > start && (end[-1] == ' ' || end[-1] == '\t')) {
      end--;
    }
    if ((size_t)(end - start) == tlen && strncasecmp(start, tok, tlen) == 0) {
      return true;
    }
  }
  return false;
}

static ssize_t recv_with_poll(
    nbase_http_server_t *srv,
    int fd,
    char *dst,
    size_t cap,
    uint32_t timeout_ms) {
  uint64_t deadline = monotonic_us() + (uint64_t)timeout_ms * 1000ULL;
  while (true) {
    if (atomic_load_explicit(&srv->stop_flag, memory_order_acquire)) {
      return 0;
    }
    uint64_t now = monotonic_us();
    if (now >= deadline) {
      return -2;
    }
    uint64_t rem_ms = (deadline - now + 999ULL) / 1000ULL;
    int wait_ms = (rem_ms > 50ULL) ? 50 : (int)rem_ms;
    if (wait_ms < 1) {
      wait_ms = 1;
    }

    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    int pr = poll(&pfd, 1U, wait_ms);
    if (pr < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    if (pr == 0) {
      continue;
    }
    ssize_t n = recv(fd, dst, cap, 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    return n;
  }
}

static void linger_drain_fd(nbase_http_server_t *srv, int fd) {
  (void)shutdown(fd, SHUT_WR);
  char trash[4096];
  size_t drained = 0U;
  uint64_t deadline = monotonic_us() + 50000ULL;
  while (drained < 65536U) {
    if (atomic_load_explicit(&srv->stop_flag, memory_order_acquire)) {
      break;
    }
    uint64_t now = monotonic_us();
    if (now >= deadline) {
      break;
    }
    uint64_t rem_ms = (deadline - now + 999ULL) / 1000ULL;
    int wait_ms = (rem_ms > 50ULL) ? 50 : (int)rem_ms;
    if (wait_ms < 1) {
      wait_ms = 1;
    }
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    int pr = poll(&pfd, 1U, wait_ms);
    if (pr < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    if (pr == 0) {
      break;
    }
    ssize_t n = recv(fd, trash, sizeof(trash), 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    if (n == 0) {
      break;
    }
    drained += (size_t)n;
  }
}

static void handle_client(nbase_http_server_t *srv, int fd) {
  struct timeval tv;
  tv.tv_sec = (time_t)(srv->socket_timeout_ms / 1000U);
  tv.tv_usec = (suseconds_t)((srv->socket_timeout_ms % 1000U) * 1000U);
  (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  char *cbuf = malloc(NBASE_HTTP_MAX_HDR + 1U);
  if (!cbuf) {
    http_conn_t err_c = {srv, fd, false};
    send_error_json(&err_c, 500, "out of memory");
    return;
  }

  bool need_linger = false;
  size_t buf_len = 0U;
  for (uint32_t req_idx = 0U; req_idx < NBASE_HTTP_MAX_KEEPALIVE_REQS; req_idx++) {
    if (atomic_load_explicit(&srv->stop_flag, memory_order_acquire)) {
      break;
    }

    cbuf[buf_len] = '\0';
    char *hdr_end = strstr(cbuf, "\r\n\r\n");
    while (hdr_end == NULL && buf_len < NBASE_HTTP_MAX_HDR) {
      ssize_t n = recv_with_poll(srv, fd, cbuf + buf_len, NBASE_HTTP_MAX_HDR - buf_len, srv->socket_timeout_ms);
      if (n <= 0) {
        goto done;
      }
      buf_len += (size_t)n;
      cbuf[buf_len] = '\0';
      hdr_end = strstr(cbuf, "\r\n\r\n");
    }

    if (hdr_end == NULL) {
      http_conn_t err_c = {srv, fd, false};
      send_error_json(&err_c, 400, "request header too large or malformed");
      need_linger = true;
      break;
    }

    size_t hdr_len = (size_t)(hdr_end - cbuf) + 4U;
    size_t avail_body = buf_len - hdr_len;

    char *line_end = strstr(cbuf, "\r\n");
    if (!line_end || line_end > hdr_end) {
      http_conn_t err_c = {srv, fd, false};
      send_error_json(&err_c, 400, "malformed request line");
      need_linger = true;
      break;
    }
    *line_end = '\0';

    char method[16];
    char target[2048];
    char version[16];
    if (sscanf(cbuf, "%15s %2047s %15s", method, target, version) != 3) {
      http_conn_t err_c = {srv, fd, false};
      send_error_json(&err_c, 400, "malformed request line");
      need_linger = true;
      break;
    }

    atomic_fetch_add_explicit(&srv->req_total, 1ULL, memory_order_relaxed);
    if (req_idx >= 1U) {
      atomic_fetch_add_explicit(&srv->keepalive_reqs, 1ULL, memory_order_relaxed);
    }

    const char *query_str = NULL;
    char *qmark = strchr(target, '?');
    if (qmark != NULL) {
      *qmark = '\0';
      query_str = qmark + 1;
    }

    size_t content_length = 0U;
    bool has_chunked = false;
    bool conn_close = false;
    bool conn_ka = false;
    char auth_val[1024];
    auth_val[0] = '\0';

    char *cur = line_end + 2;
    while (cur < hdr_end) {
      char *next = strstr(cur, "\r\n");
      if (!next || next == cur) {
        break;
      }
      *next = '\0';
      char *colon = strchr(cur, ':');
      if (colon != NULL) {
        *colon = '\0';
        const char *val = colon + 1;
        while (*val == ' ' || *val == '\t') {
          val++;
        }
        if (strcasecmp(cur, "Content-Length") == 0) {
          uint64_t cl64 = 0ULL;
          size_t vlen = strlen(val);
          while (vlen > 0U && (val[vlen - 1U] == ' ' || val[vlen - 1U] == '\t')) {
            vlen--;
          }
          if (!parse_u64_str(val, vlen, &cl64)) {
            http_conn_t err_c = {srv, fd, false};
            send_error_json(&err_c, 400, "invalid Content-Length");
            need_linger = true;
            goto done;
          }
          if (cl64 > (uint64_t)srv->max_body_bytes) {
            http_conn_t err_c = {srv, fd, false};
            send_error_json(&err_c, 413, "payload too large");
            need_linger = true;
            goto done;
          }
          content_length = (size_t)cl64;
        } else if (strcasecmp(cur, "Authorization") == 0) {
          snprintf(auth_val, sizeof(auth_val), "%s", val);
        } else if (strcasecmp(cur, "Transfer-Encoding") == 0 && strcasecmp(val, "chunked") == 0) {
          has_chunked = true;
        } else if (strcasecmp(cur, "Connection") == 0) {
          if (hdr_has_token_ci(val, "close")) {
            conn_close = true;
          }
          if (hdr_has_token_ci(val, "keep-alive")) {
            conn_ka = true;
          }
        }
      }
      cur = next + 2;
    }

    bool keep_alive = false;
    if (strcmp(version, "HTTP/1.1") == 0) {
      keep_alive = !conn_close;
    } else if (strcmp(version, "HTTP/1.0") == 0) {
      keep_alive = (conn_ka && !conn_close);
    }
    if (req_idx + 1U >= NBASE_HTTP_MAX_KEEPALIVE_REQS) {
      keep_alive = false;
    }

    http_conn_t c = {srv, fd, keep_alive};
    if (has_chunked) {
      c.keep_alive = false;
      send_error_json(&c, 400, "chunked transfer encoding not supported");
      need_linger = true;
      break;
    }

    if (strcmp(method, "OPTIONS") == 0) {
      if (content_length > 0U) {
        c.keep_alive = false;
        need_linger = true;
      } else {
        size_t rem = avail_body;
        if (rem > 0U) {
          memmove(cbuf, cbuf + hdr_len, rem);
        }
        buf_len = rem;
      }
      send_http_response(&c, 204, "text/plain", "", 0U);
      if (!c.keep_alive) {
        break;
      }
      continue;
    }

    bool is_public = (strcmp(target, "/health") == 0 || strcmp(target, "/metrics") == 0);
    if (srv->api_key != NULL && srv->api_key[0] != '\0' && !is_public) {
      if (!const_time_bearer_eq(auth_val, srv->api_key)) {
        if (content_length > 0U) {
          c.keep_alive = false;
          need_linger = true;
        } else {
          if (avail_body > 0U) {
            memmove(cbuf, cbuf + hdr_len, avail_body);
          }
          buf_len = avail_body;
        }
        send_error_json(&c, 401, "unauthorized");
        if (!c.keep_alive) {
          break;
        }
        continue;
      }
    }

    if (strncmp(target, "/api/", 5U) == 0 && !rate_limit_allow(srv)) {
      if (content_length > 0U) {
        c.keep_alive = false;
        need_linger = true;
      } else {
        if (avail_body > 0U) {
          memmove(cbuf, cbuf + hdr_len, avail_body);
        }
        buf_len = avail_body;
      }
      send_error_json(&c, 429, "rate limit exceeded");
      if (!c.keep_alive) {
        break;
      }
      continue;
    }

    char *body = NULL;
    if (content_length > 0U) {
      body = malloc(content_length + 1U);
      if (!body) {
        c.keep_alive = false;
        send_error_json(&c, 500, "out of memory");
        need_linger = true;
        break;
      }
      size_t copied = (avail_body <= content_length) ? avail_body : content_length;
      if (copied > 0U) {
        memcpy(body, cbuf + hdr_len, copied);
      }
      if (avail_body >= content_length) {
        size_t rem = avail_body - content_length;
        if (rem > 0U) {
          memmove(cbuf, cbuf + hdr_len + content_length, rem);
        }
        buf_len = rem;
      } else {
        buf_len = 0U;
      }
      while (copied < content_length) {
        ssize_t n = recv_with_poll(srv, fd, body + copied, content_length - copied, srv->socket_timeout_ms);
        if (n <= 0) {
          free(body);
          c.keep_alive = false;
          send_error_json(&c, 400, "incomplete request body");
          goto done;
        }
        copied += (size_t)n;
      }
      body[content_length] = '\0';
    } else {
      if (avail_body > 0U) {
        memmove(cbuf, cbuf + hdr_len, avail_body);
      }
      buf_len = avail_body;
    }

    dispatch_request(&c, method, target, query_str, body, content_length);
    free(body);
    if (!c.keep_alive) {
      break;
    }
  }

done:
  free(cbuf);
  if (need_linger) {
    linger_drain_fd(srv, fd);
  }
}

static void *worker_loop(void *raw) {
  nbase_http_server_t *srv = (nbase_http_server_t *)raw;
  while (true) {
    pthread_mutex_lock(&srv->q_lock);
    while (!srv->stop && srv->q_count == 0U) {
      pthread_cond_wait(&srv->q_not_empty, &srv->q_lock);
    }
    if (srv->stop) {
      pthread_mutex_unlock(&srv->q_lock);
      break;
    }
    int client_fd = srv->queue[srv->q_head];
    srv->q_head = (srv->q_head + 1U) % NBASE_HTTP_QUEUE_CAP;
    srv->q_count--;
    pthread_cond_signal(&srv->q_not_full);
    pthread_mutex_unlock(&srv->q_lock);

    handle_client(srv, client_fd);
    close(client_fd);
  }
  return NULL;
}

static void *accept_loop(void *raw) {
  nbase_http_server_t *srv = (nbase_http_server_t *)raw;
  struct pollfd pfds[2];
  pfds[0].fd = srv->listen_fd;
  pfds[0].events = POLLIN;
  pfds[1].fd = srv->wakeup_pipe[0];
  pfds[1].events = POLLIN;

  while (true) {
    pthread_mutex_lock(&srv->q_lock);
    bool stopping = srv->stop;
    pthread_mutex_unlock(&srv->q_lock);
    if (stopping) {
      break;
    }

    pfds[0].revents = 0;
    pfds[1].revents = 0;
    int pr = poll(pfds, 2U, -1);
    if (pr < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    if (pfds[1].revents != 0) {
      break;
    }
    if ((pfds[0].revents & POLLIN) == 0) {
      continue;
    }

    int client_fd = accept(srv->listen_fd, NULL, NULL);
    if (client_fd < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      break;
    }

    pthread_mutex_lock(&srv->q_lock);
    while (!srv->stop && srv->q_count == NBASE_HTTP_QUEUE_CAP) {
      pthread_cond_wait(&srv->q_not_full, &srv->q_lock);
    }
    if (srv->stop) {
      pthread_mutex_unlock(&srv->q_lock);
      close(client_fd);
      break;
    }
    srv->queue[srv->q_tail] = client_fd;
    srv->q_tail = (srv->q_tail + 1U) % NBASE_HTTP_QUEUE_CAP;
    srv->q_count++;
    pthread_cond_signal(&srv->q_not_empty);
    pthread_mutex_unlock(&srv->q_lock);
  }
  return NULL;
}

nbase_http_config_t nbase_http_config_default(void) {
  nbase_http_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.bind_host = "127.0.0.1";
  cfg.port = 3000U;
  cfg.num_workers = 4U;
  cfg.max_body_bytes = 8U * 1024U * 1024U;
  cfg.socket_timeout_ms = 5000U;
  cfg.api_key = NULL;
  cfg.save_dir = NULL;
  cfg.max_requests_per_sec = 0U;
  cfg.cors_origin = NULL;
  return cfg;
}

nbase_status_t nbase_http_server_start(
    nbase_db_t *db,
    const nbase_http_config_t *cfg,
    nbase_http_server_t **out_srv) {
  if (!db || !out_srv) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_srv = NULL;

  nbase_http_config_t eff = cfg ? *cfg : nbase_http_config_default();
  uint32_t workers = (eff.num_workers == 0U) ? 4U : ((eff.num_workers > 32U) ? 32U : eff.num_workers);
  size_t max_body = (eff.max_body_bytes == 0U) ? (8U * 1024U * 1024U) : eff.max_body_bytes;
  uint32_t timeout_ms = (eff.socket_timeout_ms == 0U) ? 5000U : eff.socket_timeout_ms;
  const char *host = (eff.bind_host && eff.bind_host[0] != '\0') ? eff.bind_host : "127.0.0.1";

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(eff.port);
  if (strcmp(host, "0.0.0.0") == 0) {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else if (strcmp(host, "localhost") == 0) {
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  } else if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
    return NBASE_ERR_INVALID_ARG;
  }

  int lfd = socket(AF_INET, SOCK_STREAM, 0);
  if (lfd < 0) {
    return NBASE_ERR_IO;
  }

  int yes = 1;
  (void)setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  if (bind(lfd, (const struct sockaddr *)&addr, sizeof(addr)) != 0 || listen(lfd, 128) != 0) {
    close(lfd);
    return NBASE_ERR_IO;
  }

  struct sockaddr_in bound;
  socklen_t blen = sizeof(bound);
  if (getsockname(lfd, (struct sockaddr *)&bound, &blen) != 0) {
    close(lfd);
    return NBASE_ERR_IO;
  }

  int pfd[2];
  if (pipe(pfd) != 0) {
    close(lfd);
    return NBASE_ERR_IO;
  }

  nbase_http_server_t *srv = calloc(1, sizeof(*srv));
  pthread_t *wtids = calloc((size_t)workers, sizeof(pthread_t));
  if (!srv || !wtids) {
    free(srv);
    free(wtids);
    close(pfd[0]);
    close(pfd[1]);
    close(lfd);
    return NBASE_ERR_NOMEM;
  }

  srv->db = db;
  srv->listen_fd = lfd;
  srv->wakeup_pipe[0] = pfd[0];
  srv->wakeup_pipe[1] = pfd[1];
  srv->bound_port = ntohs(bound.sin_port);
  srv->num_workers = workers;
  srv->max_body_bytes = max_body;
  srv->socket_timeout_ms = timeout_ms;
  srv->max_rps = eff.max_requests_per_sec;
  srv->worker_threads = wtids;

  atomic_init(&srv->req_total, 0ULL);
  atomic_init(&srv->err_total, 0ULL);
  atomic_init(&srv->search_total, 0ULL);
  atomic_init(&srv->search_latency_us_sum, 0ULL);
  atomic_init(&srv->keepalive_reqs, 0ULL);
  atomic_init(&srv->stop_flag, false);

  pthread_mutex_init(&srv->rl_lock, NULL);
  pthread_mutex_init(&srv->lat_lock, NULL);
  pthread_mutex_init(&srv->q_lock, NULL);
  pthread_cond_init(&srv->q_not_empty, NULL);
  pthread_cond_init(&srv->q_not_full, NULL);

  if (eff.api_key && eff.api_key[0] != '\0') {
    srv->api_key = dup_cstr(eff.api_key);
    if (!srv->api_key) {
      nbase_http_server_stop(srv);
      return NBASE_ERR_NOMEM;
    }
  }
  if (eff.save_dir && eff.save_dir[0] != '\0') {
    srv->save_dir = dup_cstr(eff.save_dir);
    if (!srv->save_dir) {
      nbase_http_server_stop(srv);
      return NBASE_ERR_NOMEM;
    }
  }
  if (eff.cors_origin && eff.cors_origin[0] != '\0') {
    srv->cors_origin = dup_cstr(eff.cors_origin);
    if (!srv->cors_origin) {
      nbase_http_server_stop(srv);
      return NBASE_ERR_NOMEM;
    }
  }

  for (uint32_t t = 0U; t < workers; t++) {
    if (pthread_create(&srv->worker_threads[t], NULL, worker_loop, srv) != 0) {
      nbase_http_server_stop(srv);
      return NBASE_ERR_NOMEM;
    }
    srv->workers_started++;
  }

  if (pthread_create(&srv->accept_thread, NULL, accept_loop, srv) != 0) {
    nbase_http_server_stop(srv);
    return NBASE_ERR_NOMEM;
  }
  srv->accept_started = true;

  *out_srv = srv;
  return NBASE_OK;
}

uint16_t nbase_http_server_port(const nbase_http_server_t *srv) {
  return srv ? srv->bound_port : 0U;
}

void nbase_http_server_stop(nbase_http_server_t *srv) {
  if (!srv) {
    return;
  }

  atomic_store_explicit(&srv->stop_flag, true, memory_order_release);
  pthread_mutex_lock(&srv->q_lock);
  srv->stop = true;
  pthread_cond_broadcast(&srv->q_not_empty);
  pthread_cond_broadcast(&srv->q_not_full);
  pthread_mutex_unlock(&srv->q_lock);

  if (srv->wakeup_pipe[1] >= 0) {
    char sig = 1;
    ssize_t wr = write(srv->wakeup_pipe[1], &sig, 1U);
    (void)wr;
  }

  if (srv->accept_started) {
    pthread_join(srv->accept_thread, NULL);
    srv->accept_started = false;
  }

  for (uint32_t t = 0U; t < srv->workers_started; t++) {
    pthread_join(srv->worker_threads[t], NULL);
  }
  srv->workers_started = 0U;

  pthread_mutex_lock(&srv->q_lock);
  while (srv->q_count > 0U) {
    int cfd = srv->queue[srv->q_head];
    srv->q_head = (srv->q_head + 1U) % NBASE_HTTP_QUEUE_CAP;
    srv->q_count--;
    close(cfd);
  }
  pthread_mutex_unlock(&srv->q_lock);

  if (srv->listen_fd >= 0) {
    close(srv->listen_fd);
    srv->listen_fd = -1;
  }
  if (srv->wakeup_pipe[0] >= 0) {
    close(srv->wakeup_pipe[0]);
    srv->wakeup_pipe[0] = -1;
  }
  if (srv->wakeup_pipe[1] >= 0) {
    close(srv->wakeup_pipe[1]);
    srv->wakeup_pipe[1] = -1;
  }

  pthread_cond_destroy(&srv->q_not_full);
  pthread_cond_destroy(&srv->q_not_empty);
  pthread_mutex_destroy(&srv->q_lock);
  pthread_mutex_destroy(&srv->lat_lock);
  pthread_mutex_destroy(&srv->rl_lock);
  free(srv->worker_threads);
  free(srv->api_key);
  free(srv->save_dir);
  free(srv->cors_origin);
  free(srv);
}
