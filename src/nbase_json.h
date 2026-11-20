#ifndef NBASE_JSON_H
#define NBASE_JSON_H

#include "nbase.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum nbase_json_type {
  NBASE_JSON_NULL = 0,
  NBASE_JSON_BOOL = 1,
  NBASE_JSON_NUMBER = 2,
  NBASE_JSON_STRING = 3,
  NBASE_JSON_ARRAY = 4,
  NBASE_JSON_OBJECT = 5
} nbase_json_type_t;

typedef struct nbase_json nbase_json_t;

struct nbase_json {
  nbase_json_type_t type;
  bool b;
  double num;
  int64_t i64;
  bool is_int;
  char *str;
  nbase_json_t **items;
  size_t len;
  size_t cap;
  char **keys;
  nbase_json_t **vals;
};

typedef struct nbase_sbuf {
  char *data;
  size_t len;
  size_t cap;
  bool oom;
} nbase_sbuf_t;

nbase_status_t nbase_json_parse(const char *src, size_t len, nbase_json_t **out_val);
void nbase_json_free(nbase_json_t *val);
const nbase_json_t *nbase_json_obj_get(const nbase_json_t *obj, const char *key);

void nbase_sbuf_init(nbase_sbuf_t *sb);
void nbase_sbuf_free(nbase_sbuf_t *sb);
void nbase_sbuf_append_n(nbase_sbuf_t *sb, const char *s, size_t n);
void nbase_sbuf_append(nbase_sbuf_t *sb, const char *s);
void nbase_sbuf_appendf(nbase_sbuf_t *sb, const char *fmt, ...);
void nbase_sbuf_append_json_str(nbase_sbuf_t *sb, const char *s);

#endif /* NBASE_JSON_H */
