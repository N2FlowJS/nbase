#include "nbase_json.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NBASE_JSON_MAX_DEPTH 64U

typedef struct json_parser {
  const char *src;
  size_t len;
  size_t pos;
  uint32_t depth;
} json_parser_t;

void nbase_sbuf_init(nbase_sbuf_t *sb) {
  sb->data = NULL;
  sb->len = 0U;
  sb->cap = 0U;
  sb->oom = false;
}

void nbase_sbuf_free(nbase_sbuf_t *sb) {
  if (!sb) {
    return;
  }
  free(sb->data);
  sb->data = NULL;
  sb->len = 0U;
  sb->cap = 0U;
  sb->oom = false;
}

static bool sbuf_ensure(nbase_sbuf_t *sb, size_t extra) {
  if (sb->oom) {
    return false;
  }
  if (extra > SIZE_MAX - sb->len - 1U) {
    sb->oom = true;
    return false;
  }
  size_t needed = sb->len + extra + 1U;
  if (needed <= sb->cap) {
    return true;
  }
  size_t new_cap = (sb->cap == 0U) ? 128U : sb->cap;
  while (new_cap < needed) {
    if (new_cap > SIZE_MAX / 2U) {
      new_cap = needed;
      break;
    }
    new_cap *= 2U;
  }
  char *next = realloc(sb->data, new_cap);
  if (!next) {
    sb->oom = true;
    return false;
  }
  sb->data = next;
  sb->cap = new_cap;
  sb->data[sb->len] = '\0';
  return true;
}

void nbase_sbuf_append_n(nbase_sbuf_t *sb, const char *s, size_t n) {
  if (!sb || sb->oom || n == 0U) {
    return;
  }
  if (!sbuf_ensure(sb, n)) {
    return;
  }
  memcpy(sb->data + sb->len, s, n);
  sb->len += n;
  sb->data[sb->len] = '\0';
}

void nbase_sbuf_append(nbase_sbuf_t *sb, const char *s) {
  if (!s) {
    return;
  }
  nbase_sbuf_append_n(sb, s, strlen(s));
}

void nbase_sbuf_appendf(nbase_sbuf_t *sb, const char *fmt, ...) {
  if (!sb || sb->oom || !fmt) {
    return;
  }
  va_list ap1;
  va_list ap2;
  va_start(ap1, fmt);
  va_copy(ap2, ap1);
  int req = vsnprintf(NULL, 0U, fmt, ap1);
  va_end(ap1);
  if (req < 0) {
    va_end(ap2);
    sb->oom = true;
    return;
  }
  size_t n = (size_t)req;
  if (!sbuf_ensure(sb, n)) {
    va_end(ap2);
    return;
  }
  (void)vsnprintf(sb->data + sb->len, n + 1U, fmt, ap2);
  va_end(ap2);
  sb->len += n;
}

void nbase_sbuf_append_json_str(nbase_sbuf_t *sb, const char *s) {
  nbase_sbuf_append_n(sb, "\"", 1U);
  if (s) {
    const unsigned char *p = (const unsigned char *)s;
    while (*p != '\0' && !sb->oom) {
      unsigned char ch = *p++;
      switch (ch) {
        case '"':
          nbase_sbuf_append_n(sb, "\\\"", 2U);
          break;
        case '\\':
          nbase_sbuf_append_n(sb, "\\\\", 2U);
          break;
        case '\b':
          nbase_sbuf_append_n(sb, "\\b", 2U);
          break;
        case '\f':
          nbase_sbuf_append_n(sb, "\\f", 2U);
          break;
        case '\n':
          nbase_sbuf_append_n(sb, "\\n", 2U);
          break;
        case '\r':
          nbase_sbuf_append_n(sb, "\\r", 2U);
          break;
        case '\t':
          nbase_sbuf_append_n(sb, "\\t", 2U);
          break;
        default:
          if (ch < 0x20U) {
            nbase_sbuf_appendf(sb, "\\u%04x", (unsigned int)ch);
          } else {
            char c = (char)ch;
            nbase_sbuf_append_n(sb, &c, 1U);
          }
          break;
      }
    }
  }
  nbase_sbuf_append_n(sb, "\"", 1U);
}

static void skip_ws(json_parser_t *p) {
  while (p->pos < p->len) {
    char c = p->src[p->pos];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      p->pos++;
    } else {
      break;
    }
  }
}

static bool parse_hex4(json_parser_t *p, uint32_t *out) {
  if (p->pos + 4U > p->len) {
    return false;
  }
  uint32_t val = 0U;
  for (uint32_t i = 0U; i < 4U; i++) {
    char c = p->src[p->pos++];
    uint32_t d;
    if (c >= '0' && c <= '9') {
      d = (uint32_t)(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      d = (uint32_t)(c - 'a' + 10);
    } else if (c >= 'A' && c <= 'F') {
      d = (uint32_t)(c - 'A' + 10);
    } else {
      return false;
    }
    val = (val << 4) | d;
  }
  *out = val;
  return true;
}

static void append_utf8(nbase_sbuf_t *sb, uint32_t cp) {
  char buf[4];
  size_t n;
  if (cp <= 0x7FU) {
    buf[0] = (char)cp;
    n = 1U;
  } else if (cp <= 0x7FFU) {
    buf[0] = (char)(0xC0U | (cp >> 6));
    buf[1] = (char)(0x80U | (cp & 0x3FU));
    n = 2U;
  } else if (cp <= 0xFFFFU) {
    buf[0] = (char)(0xE0U | (cp >> 12));
    buf[1] = (char)(0x80U | ((cp >> 6) & 0x3FU));
    buf[2] = (char)(0x80U | (cp & 0x3FU));
    n = 3U;
  } else {
    buf[0] = (char)(0xF0U | (cp >> 18));
    buf[1] = (char)(0x80U | ((cp >> 12) & 0x3FU));
    buf[2] = (char)(0x80U | ((cp >> 6) & 0x3FU));
    buf[3] = (char)(0x80U | (cp & 0x3FU));
    n = 4U;
  }
  nbase_sbuf_append_n(sb, buf, n);
}

static nbase_status_t parse_raw_string(json_parser_t *p, char **out_str) {
  if (p->pos >= p->len || p->src[p->pos] != '"') {
    return NBASE_ERR_INVALID_ARG;
  }
  p->pos++;

  nbase_sbuf_t sb;
  nbase_sbuf_init(&sb);

  while (p->pos < p->len) {
    unsigned char ch = (unsigned char)p->src[p->pos++];
    if (ch == '"') {
      if (sb.oom) {
        nbase_sbuf_free(&sb);
        return NBASE_ERR_NOMEM;
      }
      if (!sb.data) {
        char *empty = malloc(1U);
        if (!empty) {
          return NBASE_ERR_NOMEM;
        }
        empty[0] = '\0';
        *out_str = empty;
      } else {
        *out_str = sb.data;
      }
      return NBASE_OK;
    }
    if (ch < 0x20U) {
      nbase_sbuf_free(&sb);
      return NBASE_ERR_INVALID_ARG;
    }
    if (ch == '\\') {
      if (p->pos >= p->len) {
        nbase_sbuf_free(&sb);
        return NBASE_ERR_INVALID_ARG;
      }
      char esc = p->src[p->pos++];
      switch (esc) {
        case '"':
        case '\\':
        case '/':
          nbase_sbuf_append_n(&sb, &esc, 1U);
          break;
        case 'b':
          nbase_sbuf_append_n(&sb, "\b", 1U);
          break;
        case 'f':
          nbase_sbuf_append_n(&sb, "\f", 1U);
          break;
        case 'n':
          nbase_sbuf_append_n(&sb, "\n", 1U);
          break;
        case 'r':
          nbase_sbuf_append_n(&sb, "\r", 1U);
          break;
        case 't':
          nbase_sbuf_append_n(&sb, "\t", 1U);
          break;
        case 'u': {
          uint32_t cp = 0U;
          if (!parse_hex4(p, &cp)) {
            nbase_sbuf_free(&sb);
            return NBASE_ERR_INVALID_ARG;
          }
          if (cp >= 0xD800U && cp <= 0xDBFFU) {
            if (p->pos + 2U > p->len || p->src[p->pos] != '\\' || p->src[p->pos + 1U] != 'u') {
              nbase_sbuf_free(&sb);
              return NBASE_ERR_INVALID_ARG;
            }
            p->pos += 2U;
            uint32_t lo = 0U;
            if (!parse_hex4(p, &lo) || lo < 0xDC00U || lo > 0xDFFFU) {
              nbase_sbuf_free(&sb);
              return NBASE_ERR_INVALID_ARG;
            }
            cp = 0x10000U + (((cp - 0xD800U) << 10) | (lo - 0xDC00U));
          } else if (cp >= 0xDC00U && cp <= 0xDFFFU) {
            nbase_sbuf_free(&sb);
            return NBASE_ERR_INVALID_ARG;
          }
          append_utf8(&sb, cp);
          break;
        }
        default:
          nbase_sbuf_free(&sb);
          return NBASE_ERR_INVALID_ARG;
      }
    } else {
      char c = (char)ch;
      nbase_sbuf_append_n(&sb, &c, 1U);
    }
    if (sb.oom) {
      nbase_sbuf_free(&sb);
      return NBASE_ERR_NOMEM;
    }
  }

  nbase_sbuf_free(&sb);
  return NBASE_ERR_INVALID_ARG;
}

static nbase_status_t parse_value(json_parser_t *p, nbase_json_t **out);

static nbase_status_t parse_number(json_parser_t *p, nbase_json_t **out) {
  size_t start = p->pos;
  if (p->src[p->pos] == '-') {
    p->pos++;
  }
  if (p->pos >= p->len) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (p->src[p->pos] == '0') {
    p->pos++;
    if (p->pos < p->len && isdigit((unsigned char)p->src[p->pos])) {
      return NBASE_ERR_INVALID_ARG;
    }
  } else if (p->src[p->pos] >= '1' && p->src[p->pos] <= '9') {
    while (p->pos < p->len && isdigit((unsigned char)p->src[p->pos])) {
      p->pos++;
    }
  } else {
    return NBASE_ERR_INVALID_ARG;
  }

  bool is_float = false;
  if (p->pos < p->len && p->src[p->pos] == '.') {
    is_float = true;
    p->pos++;
    if (p->pos >= p->len || !isdigit((unsigned char)p->src[p->pos])) {
      return NBASE_ERR_INVALID_ARG;
    }
    while (p->pos < p->len && isdigit((unsigned char)p->src[p->pos])) {
      p->pos++;
    }
  }
  if (p->pos < p->len && (p->src[p->pos] == 'e' || p->src[p->pos] == 'E')) {
    is_float = true;
    p->pos++;
    if (p->pos < p->len && (p->src[p->pos] == '+' || p->src[p->pos] == '-')) {
      p->pos++;
    }
    if (p->pos >= p->len || !isdigit((unsigned char)p->src[p->pos])) {
      return NBASE_ERR_INVALID_ARG;
    }
    while (p->pos < p->len && isdigit((unsigned char)p->src[p->pos])) {
      p->pos++;
    }
  }

  size_t tok_len = p->pos - start;
  char stack_buf[64];
  char *num_buf = (tok_len < sizeof(stack_buf)) ? stack_buf : malloc(tok_len + 1U);
  if (!num_buf) {
    return NBASE_ERR_NOMEM;
  }
  memcpy(num_buf, p->src + start, tok_len);
  num_buf[tok_len] = '\0';

  nbase_json_t *v = calloc(1, sizeof(*v));
  if (!v) {
    if (num_buf != stack_buf) {
      free(num_buf);
    }
    return NBASE_ERR_NOMEM;
  }
  v->type = NBASE_JSON_NUMBER;

  if (!is_float) {
    errno = 0;
    char *endp = NULL;
    long long iv = strtoll(num_buf, &endp, 10);
    if (errno == 0 && endp == num_buf + tok_len) {
      v->is_int = true;
      v->i64 = (int64_t)iv;
      v->num = (double)iv;
      if (num_buf != stack_buf) {
        free(num_buf);
      }
      *out = v;
      return NBASE_OK;
    }
  }

  errno = 0;
  char *endd = NULL;
  double dv = strtod(num_buf, &endd);
  if (num_buf != stack_buf) {
    free(num_buf);
  }
  if (endd == NULL || errno == ERANGE || !isfinite(dv)) {
    free(v);
    return NBASE_ERR_INVALID_ARG;
  }
  v->is_int = false;
  v->num = dv;
  *out = v;
  return NBASE_OK;
}

static nbase_status_t parse_array(json_parser_t *p, nbase_json_t **out) {
  p->pos++; /* '[' */
  skip_ws(p);

  nbase_json_t *arr = calloc(1, sizeof(*arr));
  if (!arr) {
    return NBASE_ERR_NOMEM;
  }
  arr->type = NBASE_JSON_ARRAY;

  if (p->pos < p->len && p->src[p->pos] == ']') {
    p->pos++;
    *out = arr;
    return NBASE_OK;
  }

  while (true) {
    skip_ws(p);
    nbase_json_t *elem = NULL;
    nbase_status_t st = parse_value(p, &elem);
    if (st != NBASE_OK) {
      nbase_json_free(arr);
      return st;
    }

    if (arr->len == arr->cap) {
      size_t new_cap = (arr->cap == 0U) ? 8U : arr->cap * 2U;
      nbase_json_t **next = realloc(arr->items, new_cap * sizeof(*next));
      if (!next) {
        nbase_json_free(elem);
        nbase_json_free(arr);
        return NBASE_ERR_NOMEM;
      }
      arr->items = next;
      arr->cap = new_cap;
    }
    arr->items[arr->len++] = elem;

    skip_ws(p);
    if (p->pos >= p->len) {
      nbase_json_free(arr);
      return NBASE_ERR_INVALID_ARG;
    }
    if (p->src[p->pos] == ']') {
      p->pos++;
      *out = arr;
      return NBASE_OK;
    }
    if (p->src[p->pos] != ',') {
      nbase_json_free(arr);
      return NBASE_ERR_INVALID_ARG;
    }
    p->pos++;
  }
}

static nbase_status_t parse_object(json_parser_t *p, nbase_json_t **out) {
  p->pos++; /* '{' */
  skip_ws(p);

  nbase_json_t *obj = calloc(1, sizeof(*obj));
  if (!obj) {
    return NBASE_ERR_NOMEM;
  }
  obj->type = NBASE_JSON_OBJECT;

  if (p->pos < p->len && p->src[p->pos] == '}') {
    p->pos++;
    *out = obj;
    return NBASE_OK;
  }

  while (true) {
    skip_ws(p);
    char *key = NULL;
    nbase_status_t st = parse_raw_string(p, &key);
    if (st != NBASE_OK) {
      nbase_json_free(obj);
      return st;
    }

    skip_ws(p);
    if (p->pos >= p->len || p->src[p->pos] != ':') {
      free(key);
      nbase_json_free(obj);
      return NBASE_ERR_INVALID_ARG;
    }
    p->pos++;

    skip_ws(p);
    nbase_json_t *val = NULL;
    st = parse_value(p, &val);
    if (st != NBASE_OK) {
      free(key);
      nbase_json_free(obj);
      return st;
    }

    bool replaced = false;
    for (size_t i = 0U; i < obj->len; i++) {
      if (strcmp(obj->keys[i], key) == 0) {
        free(key);
        nbase_json_free(obj->vals[i]);
        obj->vals[i] = val;
        replaced = true;
        break;
      }
    }

    if (!replaced) {
      if (obj->len == obj->cap) {
        size_t new_cap = (obj->cap == 0U) ? 8U : obj->cap * 2U;
        char **next_keys = realloc(obj->keys, new_cap * sizeof(char *));
        if (!next_keys) {
          free(key);
          nbase_json_free(val);
          nbase_json_free(obj);
          return NBASE_ERR_NOMEM;
        }
        obj->keys = next_keys;
        nbase_json_t **next_vals = realloc(obj->vals, new_cap * sizeof(nbase_json_t *));
        if (!next_vals) {
          free(key);
          nbase_json_free(val);
          nbase_json_free(obj);
          return NBASE_ERR_NOMEM;
        }
        obj->vals = next_vals;
        obj->cap = new_cap;
      }
      obj->keys[obj->len] = key;
      obj->vals[obj->len] = val;
      obj->len++;
    }

    skip_ws(p);
    if (p->pos >= p->len) {
      nbase_json_free(obj);
      return NBASE_ERR_INVALID_ARG;
    }
    if (p->src[p->pos] == '}') {
      p->pos++;
      *out = obj;
      return NBASE_OK;
    }
    if (p->src[p->pos] != ',') {
      nbase_json_free(obj);
      return NBASE_ERR_INVALID_ARG;
    }
    p->pos++;
  }
}

static nbase_status_t parse_value(json_parser_t *p, nbase_json_t **out) {
  skip_ws(p);
  if (p->pos >= p->len) {
    return NBASE_ERR_INVALID_ARG;
  }

  char c = p->src[p->pos];
  if (c == 'n') {
    if (p->pos + 4U > p->len || memcmp(p->src + p->pos, "null", 4U) != 0) {
      return NBASE_ERR_INVALID_ARG;
    }
    p->pos += 4U;
    nbase_json_t *v = calloc(1, sizeof(*v));
    if (!v) {
      return NBASE_ERR_NOMEM;
    }
    v->type = NBASE_JSON_NULL;
    *out = v;
    return NBASE_OK;
  }
  if (c == 't') {
    if (p->pos + 4U > p->len || memcmp(p->src + p->pos, "true", 4U) != 0) {
      return NBASE_ERR_INVALID_ARG;
    }
    p->pos += 4U;
    nbase_json_t *v = calloc(1, sizeof(*v));
    if (!v) {
      return NBASE_ERR_NOMEM;
    }
    v->type = NBASE_JSON_BOOL;
    v->b = true;
    *out = v;
    return NBASE_OK;
  }
  if (c == 'f') {
    if (p->pos + 5U > p->len || memcmp(p->src + p->pos, "false", 5U) != 0) {
      return NBASE_ERR_INVALID_ARG;
    }
    p->pos += 5U;
    nbase_json_t *v = calloc(1, sizeof(*v));
    if (!v) {
      return NBASE_ERR_NOMEM;
    }
    v->type = NBASE_JSON_BOOL;
    v->b = false;
    *out = v;
    return NBASE_OK;
  }
  if (c == '"') {
    char *s = NULL;
    nbase_status_t st = parse_raw_string(p, &s);
    if (st != NBASE_OK) {
      return st;
    }
    nbase_json_t *v = calloc(1, sizeof(*v));
    if (!v) {
      free(s);
      return NBASE_ERR_NOMEM;
    }
    v->type = NBASE_JSON_STRING;
    v->str = s;
    *out = v;
    return NBASE_OK;
  }
  if (c == '[' || c == '{') {
    if (p->depth >= NBASE_JSON_MAX_DEPTH) {
      return NBASE_ERR_INVALID_ARG;
    }
    p->depth++;
    nbase_status_t st = (c == '[') ? parse_array(p, out) : parse_object(p, out);
    p->depth--;
    return st;
  }
  if (c == '-' || (c >= '0' && c <= '9')) {
    return parse_number(p, out);
  }
  return NBASE_ERR_INVALID_ARG;
}

nbase_status_t nbase_json_parse(const char *src, size_t len, nbase_json_t **out_val) {
  if (!out_val || (!src && len > 0U)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_val = NULL;
  if (len == 0U || !src) {
    return NBASE_ERR_INVALID_ARG;
  }

  json_parser_t p = {src, len, 0U, 0U};
  nbase_json_t *val = NULL;
  nbase_status_t st = parse_value(&p, &val);
  if (st != NBASE_OK) {
    return st;
  }
  skip_ws(&p);
  if (p.pos != p.len) {
    nbase_json_free(val);
    return NBASE_ERR_INVALID_ARG;
  }
  *out_val = val;
  return NBASE_OK;
}

void nbase_json_free(nbase_json_t *val) {
  if (!val) {
    return;
  }
  if (val->type == NBASE_JSON_STRING) {
    free(val->str);
  } else if (val->type == NBASE_JSON_ARRAY) {
    for (size_t i = 0U; i < val->len; i++) {
      nbase_json_free(val->items[i]);
    }
    free(val->items);
  } else if (val->type == NBASE_JSON_OBJECT) {
    for (size_t i = 0U; i < val->len; i++) {
      free(val->keys[i]);
      nbase_json_free(val->vals[i]);
    }
    free(val->keys);
    free(val->vals);
  }
  free(val);
}

const nbase_json_t *nbase_json_obj_get(const nbase_json_t *obj, const char *key) {
  if (!obj || obj->type != NBASE_JSON_OBJECT || !key) {
    return NULL;
  }
  for (size_t i = 0U; i < obj->len; i++) {
    if (strcmp(obj->keys[i], key) == 0) {
      return obj->vals[i];
    }
  }
  return NULL;
}
