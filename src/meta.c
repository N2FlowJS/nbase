#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define META_EMPTY 0U
#define META_OCCUPIED 1U
#define META_DELETED 2U

typedef struct meta_entry {
  char *key;
  uint32_t key_len;
  nbase_meta_type_t type;
  union {
    int64_t i64;
    double f64;
    bool b;
    struct {
      char *ptr;
      uint32_t len;
    } str;
  } as;
} meta_entry_t;

typedef struct meta_record {
  uint64_t id;
  meta_entry_t *entries;
  uint32_t count;
  uint32_t cap;
} meta_record_t;

struct nbase_meta_store {
  size_t cap;
  size_t count;
  size_t tombstones;
  uint64_t *keys;
  meta_record_t **records;
  uint8_t *states;
};

static inline uint64_t meta_hash_u64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

static void entry_free_contents(meta_entry_t *e) {
  free(e->key);
  if (e->type == NBASE_META_STRING) {
    free(e->as.str.ptr);
  }
  memset(e, 0, sizeof(*e));
}

static void record_destroy(meta_record_t *rec) {
  if (!rec) {
    return;
  }
  for (uint32_t i = 0U; i < rec->count; i++) {
    entry_free_contents(&rec->entries[i]);
  }
  free(rec->entries);
  free(rec);
}

static nbase_status_t validate_fields(const nbase_meta_field_t *fields, uint32_t count) {
  if (count > 0U && !fields) {
    return NBASE_ERR_INVALID_ARG;
  }
  for (uint32_t i = 0U; i < count; i++) {
    const nbase_meta_field_t *f = &fields[i];
    if (!f->key || f->key[0] == '\0') {
      return NBASE_ERR_INVALID_ARG;
    }
    switch (f->value.type) {
      case NBASE_META_INT64:
      case NBASE_META_BOOL:
        break;
      case NBASE_META_DOUBLE:
        if (!isfinite(f->value.as.f64)) {
          return NBASE_ERR_INVALID_ARG;
        }
        break;
      case NBASE_META_STRING:
        if (!f->value.as.str) {
          return NBASE_ERR_INVALID_ARG;
        }
        break;
      default:
        return NBASE_ERR_INVALID_ARG;
    }
  }
  return NBASE_OK;
}

static nbase_status_t entry_init_from_field(meta_entry_t *dst, const nbase_meta_field_t *src) {
  memset(dst, 0, sizeof(*dst));
  size_t klen = strlen(src->key);
  if (klen > (size_t)UINT32_MAX - 1U) {
    return NBASE_ERR_INVALID_ARG;
  }
  dst->key = malloc(klen + 1U);
  if (!dst->key) {
    return NBASE_ERR_NOMEM;
  }
  memcpy(dst->key, src->key, klen + 1U);
  dst->key_len = (uint32_t)klen;
  dst->type = src->value.type;

  switch (src->value.type) {
    case NBASE_META_INT64:
      dst->as.i64 = src->value.as.i64;
      break;
    case NBASE_META_DOUBLE:
      dst->as.f64 = src->value.as.f64;
      break;
    case NBASE_META_BOOL:
      dst->as.b = src->value.as.b;
      break;
    case NBASE_META_STRING: {
      size_t slen = strlen(src->value.as.str);
      if (slen > (size_t)UINT32_MAX - 1U) {
        free(dst->key);
        dst->key = NULL;
        return NBASE_ERR_INVALID_ARG;
      }
      dst->as.str.ptr = malloc(slen + 1U);
      if (!dst->as.str.ptr) {
        free(dst->key);
        dst->key = NULL;
        return NBASE_ERR_NOMEM;
      }
      memcpy(dst->as.str.ptr, src->value.as.str, slen + 1U);
      dst->as.str.len = (uint32_t)slen;
      break;
    }
    default:
      free(dst->key);
      dst->key = NULL;
      return NBASE_ERR_INVALID_ARG;
  }
  return NBASE_OK;
}

static const meta_entry_t *record_find(const meta_record_t *rec, const char *key) {
  for (uint32_t i = 0U; i < rec->count; i++) {
    if (strcmp(rec->entries[i].key, key) == 0) {
      return &rec->entries[i];
    }
  }
  return NULL;
}

static nbase_status_t record_upsert_field(meta_record_t *rec, const nbase_meta_field_t *field) {
  meta_entry_t next_val;
  nbase_status_t st = entry_init_from_field(&next_val, field);
  if (st != NBASE_OK) {
    return st;
  }

  for (uint32_t i = 0U; i < rec->count; i++) {
    if (strcmp(rec->entries[i].key, field->key) == 0) {
      entry_free_contents(&rec->entries[i]);
      rec->entries[i] = next_val;
      return NBASE_OK;
    }
  }

  if (rec->count == rec->cap) {
    uint32_t new_cap = (rec->cap == 0U) ? 4U : rec->cap * 2U;
    meta_entry_t *grown = realloc(rec->entries, (size_t)new_cap * sizeof(meta_entry_t));
    if (!grown) {
      entry_free_contents(&next_val);
      return NBASE_ERR_NOMEM;
    }
    rec->entries = grown;
    rec->cap = new_cap;
  }
  rec->entries[rec->count++] = next_val;
  return NBASE_OK;
}

static bool meta_table_init(nbase_meta_store_t *meta, size_t initial_cap) {
  size_t cap = 16U;
  while (cap < initial_cap) {
    if (cap > SIZE_MAX / 2U) {
      return false;
    }
    cap *= 2U;
  }
  meta->keys = calloc(cap, sizeof(uint64_t));
  meta->records = calloc(cap, sizeof(meta_record_t *));
  meta->states = calloc(cap, sizeof(uint8_t));
  if (!meta->keys || !meta->records || !meta->states) {
    free(meta->keys);
    free(meta->records);
    free(meta->states);
    memset(meta, 0, sizeof(*meta));
    return false;
  }
  meta->cap = cap;
  meta->count = 0U;
  meta->tombstones = 0U;
  return true;
}

static bool meta_table_rehash(nbase_meta_store_t *meta, size_t new_cap) {
  nbase_meta_store_t next;
  if (!meta_table_init(&next, new_cap)) {
    return false;
  }
  size_t mask = next.cap - 1U;
  for (size_t i = 0U; i < meta->cap; i++) {
    if (meta->states[i] == META_OCCUPIED) {
      size_t idx = (size_t)meta_hash_u64(meta->keys[i]) & mask;
      while (next.states[idx] == META_OCCUPIED) {
        idx = (idx + 1U) & mask;
      }
      next.keys[idx] = meta->keys[i];
      next.records[idx] = meta->records[i];
      next.states[idx] = META_OCCUPIED;
      next.count++;
    }
  }
  free(meta->keys);
  free(meta->records);
  free(meta->states);
  *meta = next;
  return true;
}

static meta_record_t *meta_find_record(const nbase_meta_store_t *meta, uint64_t id) {
  if (!meta || meta->cap == 0U) {
    return NULL;
  }
  size_t mask = meta->cap - 1U;
  size_t idx = (size_t)meta_hash_u64(id) & mask;
  for (size_t probe = 0U; probe < meta->cap; probe++) {
    uint8_t st = meta->states[idx];
    if (st == META_EMPTY) {
      return NULL;
    }
    if (st == META_OCCUPIED && meta->keys[idx] == id) {
      return meta->records[idx];
    }
    idx = (idx + 1U) & mask;
  }
  return NULL;
}

static nbase_status_t meta_put_record(nbase_meta_store_t *meta, meta_record_t *rec) {
  if ((meta->count + meta->tombstones + 1U) * 100U >= meta->cap * 65U) {
    size_t target = (meta->count * 2U >= meta->cap) ? meta->cap * 2U : meta->cap;
    if (!meta_table_rehash(meta, target)) {
      return NBASE_ERR_NOMEM;
    }
  }
  size_t mask = meta->cap - 1U;
  size_t idx = (size_t)meta_hash_u64(rec->id) & mask;
  size_t first_deleted = SIZE_MAX;

  for (size_t probe = 0U; probe < meta->cap; probe++) {
    uint8_t st = meta->states[idx];
    if (st == META_EMPTY) {
      size_t dst = (first_deleted != SIZE_MAX) ? first_deleted : idx;
      if (meta->states[dst] == META_DELETED) {
        meta->tombstones--;
      }
      meta->keys[dst] = rec->id;
      meta->records[dst] = rec;
      meta->states[dst] = META_OCCUPIED;
      meta->count++;
      return NBASE_OK;
    }
    if (st == META_OCCUPIED && meta->keys[idx] == rec->id) {
      record_destroy(meta->records[idx]);
      meta->records[idx] = rec;
      return NBASE_OK;
    }
    if (st == META_DELETED && first_deleted == SIZE_MAX) {
      first_deleted = idx;
    }
    idx = (idx + 1U) & mask;
  }
  return NBASE_ERR_NOMEM;
}

nbase_status_t nbase_meta_create(nbase_meta_store_t **out_meta) {
  if (!out_meta) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_meta_store_t *meta = calloc(1, sizeof(*meta));
  if (!meta) {
    return NBASE_ERR_NOMEM;
  }
  if (!meta_table_init(meta, 16U)) {
    free(meta);
    return NBASE_ERR_NOMEM;
  }
  *out_meta = meta;
  return NBASE_OK;
}

void nbase_meta_destroy(nbase_meta_store_t *meta) {
  if (!meta) {
    return;
  }
  for (size_t i = 0U; i < meta->cap; i++) {
    if (meta->states[i] == META_OCCUPIED) {
      record_destroy(meta->records[i]);
    }
  }
  free(meta->keys);
  free(meta->records);
  free(meta->states);
  free(meta);
}

uint64_t nbase_meta_count(const nbase_meta_store_t *meta) {
  return meta ? (uint64_t)meta->count : 0ULL;
}

nbase_status_t nbase_meta_set(nbase_meta_store_t *meta, uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
  if (!meta) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_status_t st = validate_fields(fields, count);
  if (st != NBASE_OK) {
    return st;
  }
  if (count == 0U) {
    return nbase_meta_delete(meta, id);
  }

  meta_record_t *rec = calloc(1, sizeof(*rec));
  if (!rec) {
    return NBASE_ERR_NOMEM;
  }
  rec->id = id;
  for (uint32_t i = 0U; i < count; i++) {
    st = record_upsert_field(rec, &fields[i]);
    if (st != NBASE_OK) {
      record_destroy(rec);
      return st;
    }
  }

  st = meta_put_record(meta, rec);
  if (st != NBASE_OK) {
    record_destroy(rec);
  }
  return st;
}

nbase_status_t nbase_meta_update(nbase_meta_store_t *meta, uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
  if (!meta) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_status_t st = validate_fields(fields, count);
  if (st != NBASE_OK) {
    return st;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  meta_record_t *existing = meta_find_record(meta, id);
  if (!existing) {
    return nbase_meta_set(meta, id, fields, count);
  }

  for (uint32_t i = 0U; i < count; i++) {
    st = record_upsert_field(existing, &fields[i]);
    if (st != NBASE_OK) {
      return st;
    }
  }
  return NBASE_OK;
}

nbase_status_t nbase_meta_get(const nbase_meta_store_t *meta, uint64_t id, const char *key, nbase_meta_value_t *out_val, char *str_buf, size_t str_buf_len) {
  if (!meta || !key || key[0] == '\0' || !out_val) {
    return NBASE_ERR_INVALID_ARG;
  }
  const meta_record_t *rec = meta_find_record(meta, id);
  if (!rec) {
    return NBASE_ERR_NOT_FOUND;
  }
  const meta_entry_t *entry = record_find(rec, key);
  if (!entry) {
    return NBASE_ERR_NOT_FOUND;
  }

  memset(out_val, 0, sizeof(*out_val));
  out_val->type = entry->type;
  switch (entry->type) {
    case NBASE_META_INT64:
      out_val->as.i64 = entry->as.i64;
      break;
    case NBASE_META_DOUBLE:
      out_val->as.f64 = entry->as.f64;
      break;
    case NBASE_META_BOOL:
      out_val->as.b = entry->as.b;
      break;
    case NBASE_META_STRING:
      if (!str_buf || str_buf_len <= (size_t)entry->as.str.len) {
        return NBASE_ERR_INVALID_ARG;
      }
      memcpy(str_buf, entry->as.str.ptr, (size_t)entry->as.str.len + 1U);
      out_val->as.str = str_buf;
      break;
    default:
      return NBASE_ERR_CORRUPT;
  }
  return NBASE_OK;
}

nbase_status_t nbase_meta_delete(nbase_meta_store_t *meta, uint64_t id) {
  if (!meta) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (meta->cap == 0U) {
    return NBASE_OK;
  }
  size_t mask = meta->cap - 1U;
  size_t idx = (size_t)meta_hash_u64(id) & mask;
  for (size_t probe = 0U; probe < meta->cap; probe++) {
    uint8_t st = meta->states[idx];
    if (st == META_EMPTY) {
      return NBASE_OK;
    }
    if (st == META_OCCUPIED && meta->keys[idx] == id) {
      record_destroy(meta->records[idx]);
      meta->records[idx] = NULL;
      meta->states[idx] = META_DELETED;
      meta->count--;
      meta->tombstones++;
      return NBASE_OK;
    }
    idx = (idx + 1U) & mask;
  }
  return NBASE_OK;
}

static bool eval_condition(const meta_entry_t *entry, const nbase_meta_cond_t *cond) {
  if (cond->op == NBASE_OP_EXISTS) {
    return true;
  }

  /* Strict type bracketing: numeric only matches numeric, bool only matches bool, string only matches string. */
  if (entry->type == NBASE_META_INT64 && cond->value.type == NBASE_META_INT64) {
    int64_t a = entry->as.i64;
    int64_t b = cond->value.as.i64;
    switch (cond->op) {
      case NBASE_OP_EQ:
        return a == b;
      case NBASE_OP_NE:
        return a != b;
      case NBASE_OP_GT:
        return a > b;
      case NBASE_OP_GTE:
        return a >= b;
      case NBASE_OP_LT:
        return a < b;
      case NBASE_OP_LTE:
        return a <= b;
      default:
        return false;
    }
  }

  if ((entry->type == NBASE_META_INT64 || entry->type == NBASE_META_DOUBLE) && (cond->value.type == NBASE_META_INT64 || cond->value.type == NBASE_META_DOUBLE)) {
    double a = (entry->type == NBASE_META_INT64) ? (double)entry->as.i64 : entry->as.f64;
    double b = (cond->value.type == NBASE_META_INT64) ? (double)cond->value.as.i64 : cond->value.as.f64;
    if (!isfinite(b)) {
      return false;
    }
    switch (cond->op) {
      case NBASE_OP_EQ:
        return a == b;
      case NBASE_OP_NE:
        return a != b;
      case NBASE_OP_GT:
        return a > b;
      case NBASE_OP_GTE:
        return a >= b;
      case NBASE_OP_LT:
        return a < b;
      case NBASE_OP_LTE:
        return a <= b;
      default:
        return false;
    }
  }

  if (entry->type == NBASE_META_BOOL && cond->value.type == NBASE_META_BOOL) {
    bool a = entry->as.b;
    bool b = cond->value.as.b;
    switch (cond->op) {
      case NBASE_OP_EQ:
        return a == b;
      case NBASE_OP_NE:
        return a != b;
      default:
        return false;
    }
  }

  if (entry->type == NBASE_META_STRING && cond->value.type == NBASE_META_STRING) {
    if (!cond->value.as.str) {
      return false;
    }
    if (cond->op == NBASE_OP_CONTAINS) {
      return strstr(entry->as.str.ptr, cond->value.as.str) != NULL;
    }
    int cmp = strcmp(entry->as.str.ptr, cond->value.as.str);
    switch (cond->op) {
      case NBASE_OP_EQ:
        return cmp == 0;
      case NBASE_OP_NE:
        return cmp != 0;
      case NBASE_OP_GT:
        return cmp > 0;
      case NBASE_OP_GTE:
        return cmp >= 0;
      case NBASE_OP_LT:
        return cmp < 0;
      case NBASE_OP_LTE:
        return cmp <= 0;
      default:
        return false;
    }
  }

  return false;
}

bool nbase_meta_matches(const nbase_meta_store_t *meta, uint64_t id, const nbase_meta_filter_t *filter) {
  if (!filter || filter->count == 0U) {
    return true;
  }
  if (!filter->conds || !meta) {
    return false;
  }
  const meta_record_t *rec = meta_find_record(meta, id);
  if (!rec) {
    return false;
  }
  for (uint32_t i = 0U; i < filter->count; i++) {
    const nbase_meta_cond_t *cond = &filter->conds[i];
    if (!cond->key || cond->key[0] == '\0') {
      return false;
    }
    const meta_entry_t *entry = record_find(rec, cond->key);
    if (!entry || !eval_condition(entry, cond)) {
      return false;
    }
  }
  return true;
}

nbase_status_t nbase_meta_foreach(
    const nbase_meta_store_t *meta,
    uint64_t id,
    nbase_meta_iter_fn cb,
    void *user_ctx) {
  if (!meta || !cb) {
    return NBASE_ERR_INVALID_ARG;
  }
  const meta_record_t *rec = meta_find_record(meta, id);
  if (!rec) {
    return NBASE_OK;
  }
  for (uint32_t i = 0U; i < rec->count; i++) {
    const meta_entry_t *e = &rec->entries[i];
    nbase_meta_value_t val;
    val.type = e->type;
    switch (e->type) {
      case NBASE_META_INT64:
        val.as.i64 = e->as.i64;
        break;
      case NBASE_META_DOUBLE:
        val.as.f64 = e->as.f64;
        break;
      case NBASE_META_BOOL:
        val.as.b = e->as.b;
        break;
      case NBASE_META_STRING:
        val.as.str = e->as.str.ptr;
        break;
    }
    if (!cb(e->key, &val, user_ctx)) {
      break;
    }
  }
  return NBASE_OK;
}

nbase_status_t nbase_meta_save(const nbase_meta_store_t *meta, const char *file_path) {
  if (!meta || !file_path) {
    return NBASE_ERR_INVALID_ARG;
  }

  size_t payload_bytes = 0U;
  for (size_t i = 0U; i < meta->cap; i++) {
    if (meta->states[i] != META_OCCUPIED) {
      continue;
    }
    const meta_record_t *rec = meta->records[i];
    payload_bytes += sizeof(uint64_t) + sizeof(uint32_t);
    for (uint32_t f = 0U; f < rec->count; f++) {
      const meta_entry_t *e = &rec->entries[f];
      payload_bytes += sizeof(uint32_t) + (size_t)e->key_len + sizeof(uint8_t);
      switch (e->type) {
        case NBASE_META_INT64:
          payload_bytes += sizeof(int64_t);
          break;
        case NBASE_META_DOUBLE:
          payload_bytes += sizeof(double);
          break;
        case NBASE_META_BOOL:
          payload_bytes += sizeof(uint8_t);
          break;
        case NBASE_META_STRING:
          payload_bytes += sizeof(uint32_t) + (size_t)e->as.str.len;
          break;
      }
    }
  }

  uint8_t *buf = NULL;
  if (payload_bytes > 0U) {
    buf = malloc(payload_bytes);
    if (!buf) {
      return NBASE_ERR_NOMEM;
    }
    size_t pos = 0U;
    for (size_t i = 0U; i < meta->cap; i++) {
      if (meta->states[i] != META_OCCUPIED) {
        continue;
      }
      const meta_record_t *rec = meta->records[i];
      memcpy(buf + pos, &rec->id, sizeof(uint64_t));
      pos += sizeof(uint64_t);
      memcpy(buf + pos, &rec->count, sizeof(uint32_t));
      pos += sizeof(uint32_t);
      for (uint32_t f = 0U; f < rec->count; f++) {
        const meta_entry_t *e = &rec->entries[f];
        uint8_t type_byte = (uint8_t)e->type;
        memcpy(buf + pos, &e->key_len, sizeof(uint32_t));
        pos += sizeof(uint32_t);
        memcpy(buf + pos, e->key, e->key_len);
        pos += e->key_len;
        memcpy(buf + pos, &type_byte, sizeof(uint8_t));
        pos += sizeof(uint8_t);
        switch (e->type) {
          case NBASE_META_INT64:
            memcpy(buf + pos, &e->as.i64, sizeof(int64_t));
            pos += sizeof(int64_t);
            break;
          case NBASE_META_DOUBLE:
            memcpy(buf + pos, &e->as.f64, sizeof(double));
            pos += sizeof(double);
            break;
          case NBASE_META_BOOL: {
            uint8_t b = e->as.b ? 1U : 0U;
            memcpy(buf + pos, &b, sizeof(uint8_t));
            pos += sizeof(uint8_t);
            break;
          }
          case NBASE_META_STRING:
            memcpy(buf + pos, &e->as.str.len, sizeof(uint32_t));
            pos += sizeof(uint32_t);
            if (e->as.str.len > 0U) {
              memcpy(buf + pos, e->as.str.ptr, e->as.str.len);
              pos += e->as.str.len;
            }
            break;
        }
      }
    }
  }

  nbm_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBM_MAGIC, 8);
  hdr.version = NBASE_NBM_VERSION;
  hdr.flags = 0U;
  hdr.record_count = meta->count;
  hdr.payload_bytes = (uint64_t)payload_bytes;
  hdr.payload_crc32 = nbase_crc32(buf, payload_bytes);
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nbm_header_t, header_crc32));

  nbase_io_chunk_t chunk = {buf, payload_bytes};
  nbase_status_t st = nbase_write_file_atomic(file_path, &hdr, sizeof(hdr), &chunk, 1);
  free(buf);
  return st;
}

nbase_status_t nbase_meta_load(const char *file_path, nbase_meta_store_t **out_meta) {
  if (!file_path || !out_meta) {
    return NBASE_ERR_INVALID_ARG;
  }

  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    return (errno == ENOENT) ? NBASE_ERR_NOT_FOUND : NBASE_ERR_IO;
  }

  struct stat st_buf;
  if (fstat(fd, &st_buf) != 0) {
    close(fd);
    return NBASE_ERR_IO;
  }
  if (st_buf.st_size < (off_t)sizeof(nbm_header_t)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;

  nbm_header_t hdr;
  nbase_status_t st = nbase_read_exact(fd, &hdr, sizeof(hdr));
  if (st != NBASE_OK) {
    close(fd);
    return st;
  }

  if (memcmp(hdr.magic, NBASE_NBM_MAGIC, 8) != 0 || hdr.version != NBASE_NBM_VERSION) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.header_crc32 != nbase_crc32(&hdr, offsetof(nbm_header_t, header_crc32))) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (file_size != sizeof(nbm_header_t) + hdr.payload_bytes || hdr.payload_bytes > (uint64_t)SIZE_MAX) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint8_t *buf = NULL;
  if (hdr.payload_bytes > 0U) {
    buf = malloc((size_t)hdr.payload_bytes);
    if (!buf) {
      close(fd);
      return NBASE_ERR_NOMEM;
    }
    st = nbase_read_exact(fd, buf, (size_t)hdr.payload_bytes);
    close(fd);
    if (st != NBASE_OK) {
      free(buf);
      return st;
    }
    if (nbase_crc32(buf, (size_t)hdr.payload_bytes) != hdr.payload_crc32) {
      free(buf);
      return NBASE_ERR_CORRUPT;
    }
  } else {
    close(fd);
  }

  nbase_meta_store_t *meta = NULL;
  st = nbase_meta_create(&meta);
  if (st != NBASE_OK) {
    free(buf);
    return st;
  }

  size_t pos = 0U;
  size_t payload_len = (size_t)hdr.payload_bytes;
  for (uint64_t r = 0ULL; r < hdr.record_count; r++) {
    if (pos + sizeof(uint64_t) + sizeof(uint32_t) > payload_len) {
      free(buf);
      nbase_meta_destroy(meta);
      return NBASE_ERR_CORRUPT;
    }
    uint64_t id = 0ULL;
    uint32_t fcount = 0U;
    memcpy(&id, buf + pos, sizeof(uint64_t));
    pos += sizeof(uint64_t);
    memcpy(&fcount, buf + pos, sizeof(uint32_t));
    pos += sizeof(uint32_t);

    if (meta_find_record(meta, id) != NULL) {
      free(buf);
      nbase_meta_destroy(meta);
      return NBASE_ERR_CORRUPT;
    }

    meta_record_t *rec = calloc(1, sizeof(*rec));
    if (!rec) {
      free(buf);
      nbase_meta_destroy(meta);
      return NBASE_ERR_NOMEM;
    }
    rec->id = id;

    for (uint32_t f = 0U; f < fcount; f++) {
      if (pos + sizeof(uint32_t) > payload_len) {
        record_destroy(rec);
        free(buf);
        nbase_meta_destroy(meta);
        return NBASE_ERR_CORRUPT;
      }
      uint32_t klen = 0U;
      memcpy(&klen, buf + pos, sizeof(uint32_t));
      pos += sizeof(uint32_t);
      if (klen == 0U || klen > (uint32_t)(payload_len - pos) || pos + klen + sizeof(uint8_t) > payload_len) {
        record_destroy(rec);
        free(buf);
        nbase_meta_destroy(meta);
        return NBASE_ERR_CORRUPT;
      }

      char *key_tmp = malloc((size_t)klen + 1U);
      if (!key_tmp) {
        record_destroy(rec);
        free(buf);
        nbase_meta_destroy(meta);
        return NBASE_ERR_NOMEM;
      }
      memcpy(key_tmp, buf + pos, klen);
      key_tmp[klen] = '\0';
      pos += klen;

      if (memchr(key_tmp, '\0', klen) != NULL || record_find(rec, key_tmp) != NULL) {
        free(key_tmp);
        record_destroy(rec);
        free(buf);
        nbase_meta_destroy(meta);
        return NBASE_ERR_CORRUPT;
      }

      uint8_t type_byte = buf[pos++];
      nbase_meta_field_t fld;
      fld.key = key_tmp;
      fld.value.type = (nbase_meta_type_t)type_byte;
      char *str_tmp = NULL;

      switch (fld.value.type) {
        case NBASE_META_INT64:
          if (pos + sizeof(int64_t) > payload_len) {
            free(key_tmp);
            record_destroy(rec);
            free(buf);
            nbase_meta_destroy(meta);
            return NBASE_ERR_CORRUPT;
          }
          memcpy(&fld.value.as.i64, buf + pos, sizeof(int64_t));
          pos += sizeof(int64_t);
          break;
        case NBASE_META_DOUBLE:
          if (pos + sizeof(double) > payload_len) {
            free(key_tmp);
            record_destroy(rec);
            free(buf);
            nbase_meta_destroy(meta);
            return NBASE_ERR_CORRUPT;
          }
          memcpy(&fld.value.as.f64, buf + pos, sizeof(double));
          pos += sizeof(double);
          if (!isfinite(fld.value.as.f64)) {
            free(key_tmp);
            record_destroy(rec);
            free(buf);
            nbase_meta_destroy(meta);
            return NBASE_ERR_CORRUPT;
          }
          break;
        case NBASE_META_BOOL: {
          if (pos + sizeof(uint8_t) > payload_len) {
            free(key_tmp);
            record_destroy(rec);
            free(buf);
            nbase_meta_destroy(meta);
            return NBASE_ERR_CORRUPT;
          }
          uint8_t b = buf[pos++];
          if (b > 1U) {
            free(key_tmp);
            record_destroy(rec);
            free(buf);
            nbase_meta_destroy(meta);
            return NBASE_ERR_CORRUPT;
          }
          fld.value.as.b = (b != 0U);
          break;
        }
        case NBASE_META_STRING: {
          if (pos + sizeof(uint32_t) > payload_len) {
            free(key_tmp);
            record_destroy(rec);
            free(buf);
            nbase_meta_destroy(meta);
            return NBASE_ERR_CORRUPT;
          }
          uint32_t slen = 0U;
          memcpy(&slen, buf + pos, sizeof(uint32_t));
          pos += sizeof(uint32_t);
          if ((size_t)slen > payload_len - pos) {
            free(key_tmp);
            record_destroy(rec);
            free(buf);
            nbase_meta_destroy(meta);
            return NBASE_ERR_CORRUPT;
          }
          str_tmp = malloc((size_t)slen + 1U);
          if (!str_tmp) {
            free(key_tmp);
            record_destroy(rec);
            free(buf);
            nbase_meta_destroy(meta);
            return NBASE_ERR_NOMEM;
          }
          if (slen > 0U) {
            memcpy(str_tmp, buf + pos, slen);
          }
          str_tmp[slen] = '\0';
          pos += slen;
          fld.value.as.str = str_tmp;
          break;
        }
        default:
          free(key_tmp);
          record_destroy(rec);
          free(buf);
          nbase_meta_destroy(meta);
          return NBASE_ERR_CORRUPT;
      }

      st = record_upsert_field(rec, &fld);
      free(key_tmp);
      free(str_tmp);
      if (st != NBASE_OK) {
        record_destroy(rec);
        free(buf);
        nbase_meta_destroy(meta);
        return st;
      }
    }

    st = meta_put_record(meta, rec);
    if (st != NBASE_OK) {
      record_destroy(rec);
      free(buf);
      nbase_meta_destroy(meta);
      return st;
    }
  }

  free(buf);
  if (pos != payload_len || meta->count != hdr.record_count) {
    nbase_meta_destroy(meta);
    return NBASE_ERR_CORRUPT;
  }

  *out_meta = meta;
  return NBASE_OK;
}

uint64_t nbase_meta_memory_bytes(const nbase_meta_store_t *meta) {
  if (!meta) {
    return 0ULL;
  }
  uint64_t total = (uint64_t)sizeof(*meta);
  total += (uint64_t)meta->cap *
           ((uint64_t)sizeof(uint64_t) + (uint64_t)sizeof(meta_record_t *) + (uint64_t)sizeof(uint8_t));
  for (size_t i = 0U; i < meta->cap; i++) {
    if (meta->states != NULL && meta->states[i] == META_OCCUPIED && meta->records[i] != NULL) {
      const meta_record_t *rec = meta->records[i];
      total += (uint64_t)sizeof(*rec) + (uint64_t)rec->cap * (uint64_t)sizeof(meta_entry_t);
      for (uint32_t j = 0U; j < rec->count; j++) {
        total += (uint64_t)rec->entries[j].key_len + 1ULL;
        if (rec->entries[j].type == NBASE_META_STRING) {
          total += (uint64_t)rec->entries[j].as.str.len + 1ULL;
        }
      }
    }
  }
  return total;
}
