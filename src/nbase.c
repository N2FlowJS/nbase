#include "nbase_internal.h"

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct db_cache_entry {
  uint64_t hash;
  float *query;
  uint32_t dim;
  uint32_t k;
  uint32_t ef_search;
  nbase_search_mode_t mode;
  nbase_metric_t metric;
  uint8_t *filter_blob;
  size_t filter_blob_len;
  nbase_result_t *results;
  uint32_t result_count;
  struct db_cache_entry *prev;
  struct db_cache_entry *next;
} db_cache_entry_t;

struct nbase_db {
  nbase_store_t *store;
  nbase_hnsw_t *hnsw;
  nbase_meta_store_t *meta;
  nbase_bm25_t *bm25;
  nbase_wal_t *wal;
  char *wal_dir;
  pthread_rwlock_t rwlock;
  pthread_mutex_t cache_lock;
  uint32_t cache_cap;
  uint32_t cache_count;
  uint64_t cache_hits;
  uint64_t cache_misses;
  db_cache_entry_t *cache_head;
  db_cache_entry_t *cache_tail;
  bool enable_hnsw;
  uint64_t seed;
};

typedef struct combined_filter_ctx {
  const nbase_meta_store_t *meta;
  const nbase_meta_filter_t *meta_filter;
  nbase_filter_fn user_filter;
  void *user_ctx;
} combined_filter_ctx_t;

typedef struct batch_worker_ctx {
  const nbase_db_t *db;
  const float *queries;
  uint32_t query_count;
  uint32_t dim;
  uint32_t k;
  nbase_search_opts_t opts;
  nbase_result_t *out_results;
  uint32_t *out_counts;
  atomic_uint next_idx;
  atomic_int first_err;
} batch_worker_ctx_t;

typedef struct text_batch_worker_ctx {
  const nbase_db_t *db;
  const char *const *queries;
  uint32_t query_count;
  uint32_t k;
  const nbase_meta_filter_t *meta_filter;
  nbase_scored_id_t *out_results;
  uint32_t *out_counts;
  atomic_uint next_idx;
  atomic_int first_err;
} text_batch_worker_ctx_t;

typedef struct hybrid_batch_worker_ctx {
  const nbase_db_t *db;
  const float *query_vectors;
  uint32_t dim;
  const char *const *query_texts;
  uint32_t query_count;
  uint32_t k;
  float alpha;
  float rrf_k;
  nbase_search_opts_t opts;
  nbase_scored_id_t *out_results;
  uint32_t *out_counts;
  atomic_uint next_idx;
  atomic_int first_err;
} hybrid_batch_worker_ctx_t;

static bool combined_filter_cb(uint64_t id, void *raw_ctx) {
  const combined_filter_ctx_t *ctx = (const combined_filter_ctx_t *)raw_ctx;
  if (!nbase_meta_matches(ctx->meta, id, ctx->meta_filter)) {
    return false;
  }
  if (ctx->user_filter != NULL && !ctx->user_filter(id, ctx->user_ctx)) {
    return false;
  }
  return true;
}

static char *join_path(const char *dir, const char *file) {
  size_t dlen = strlen(dir);
  size_t flen = strlen(file);
  bool need_slash = (dlen > 0U && dir[dlen - 1U] != '/');
  size_t total = dlen + (need_slash ? 1U : 0U) + flen + 1U;
  char *out = malloc(total);
  if (!out) {
    return NULL;
  }
  if (need_slash) {
    snprintf(out, total, "%s/%s", dir, file);
  } else {
    snprintf(out, total, "%s%s", dir, file);
  }
  return out;
}

const char *nbase_version(void) {
  return "1.14.0";
}

const char *nbase_strerror(nbase_status_t status) {
  switch (status) {
    case NBASE_OK:
      return "ok";
    case NBASE_ERR_INVALID_ARG:
      return "invalid argument";
    case NBASE_ERR_NOMEM:
      return "out of memory";
    case NBASE_ERR_NOT_FOUND:
      return "not found";
    case NBASE_ERR_ALREADY_EXISTS:
      return "already exists";
    case NBASE_ERR_DIM_MISMATCH:
      return "dimension mismatch";
    case NBASE_ERR_IO:
      return "I/O error";
    case NBASE_ERR_CORRUPT:
      return "corrupt database file";
    case NBASE_ERR_UNSUPPORTED:
      return "unsupported operation";
    case NBASE_ERR_INVALID_VECTOR:
      return "invalid vector (NaN or Inf)";
    default:
      return "unknown error";
  }
}

nbase_config_t nbase_config_default(uint32_t dim) {
  nbase_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.dim = dim;
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.initial_capacity = 64U;
  cfg.enable_hnsw = true;
  cfg.hnsw_m = 16U;
  cfg.hnsw_ef_construction = 200U;
  cfg.hnsw_ef_search = 50U;
  cfg.seed = 0ULL;
  cfg.cache_capacity = 0U;
  return cfg;
}

nbase_search_opts_t nbase_search_opts_default(void) {
  nbase_search_opts_t opts;
  memset(&opts, 0, sizeof(opts));
  opts.mode = NBASE_SEARCH_AUTO;
  opts.metric = NBASE_METRIC_EUCLIDEAN;
  opts.use_metric_override = false;
  opts.ef_search = 0U;
  opts.filter = NULL;
  opts.filter_ctx = NULL;
  opts.meta_filter = NULL;
  opts.num_threads = 0U;
  opts.skip_cache = false;
  return opts;
}

nbase_status_t nbase_create(const nbase_config_t *cfg, nbase_db_t **out_db) {
  if (!cfg || !out_db) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (cfg->dim == 0U || cfg->dim > NBASE_MAX_DIM || !nbase_metric_is_valid(cfg->metric)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (cfg->enable_hnsw && (cfg->hnsw_m < 2U || cfg->hnsw_ef_construction == 0U || cfg->hnsw_ef_search == 0U)) {
    return NBASE_ERR_INVALID_ARG;
  }

  nbase_db_t *db = calloc(1, sizeof(*db));
  if (!db) {
    return NBASE_ERR_NOMEM;
  }

  nbase_status_t st = nbase_store_create(cfg->dim, cfg->metric, cfg->initial_capacity, &db->store);
  if (st != NBASE_OK) {
    free(db);
    return st;
  }

  st = nbase_meta_create(&db->meta);
  if (st != NBASE_OK) {
    nbase_store_destroy(db->store);
    free(db);
    return st;
  }

  uint64_t seed = cfg->seed;
  if (seed == 0ULL) {
    seed = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)db;
    if (seed == 0ULL) {
      seed = 1ULL;
    }
  }
  db->seed = seed;
  db->enable_hnsw = cfg->enable_hnsw;

  if (cfg->enable_hnsw) {
    st = nbase_hnsw_create(cfg->hnsw_m, cfg->hnsw_ef_construction, cfg->hnsw_ef_search, seed, &db->hnsw);
    if (st != NBASE_OK) {
      nbase_meta_destroy(db->meta);
      nbase_store_destroy(db->store);
      free(db);
      return st;
    }
  }

  if (pthread_rwlock_init(&db->rwlock, NULL) != 0) {
    nbase_hnsw_destroy(db->hnsw);
    nbase_meta_destroy(db->meta);
    nbase_store_destroy(db->store);
    free(db);
    return NBASE_ERR_NOMEM;
  }
  if (pthread_mutex_init(&db->cache_lock, NULL) != 0) {
    pthread_rwlock_destroy(&db->rwlock);
    nbase_hnsw_destroy(db->hnsw);
    nbase_meta_destroy(db->meta);
    nbase_store_destroy(db->store);
    free(db);
    return NBASE_ERR_NOMEM;
  }
  db->cache_cap = cfg->cache_capacity;

  *out_db = db;
  return NBASE_OK;
}

static int cmp_u64(const void *pa, const void *pb) {
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

static uint32_t default_worker_count(uint32_t requested) {
  if (requested > 0U) {
    return requested > 64U ? 64U : requested;
  }
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpu <= 1L) {
    return 1U;
  }
  return (ncpu > 8L) ? 8U : (uint32_t)ncpu;
}

static nbase_status_t db_ensure_bm25_unlocked(nbase_db_t *db) {
  if (db->bm25 != NULL) {
    return NBASE_OK;
  }
  return nbase_bm25_create(1.5f, 0.75f, &db->bm25);
}

static nbase_status_t db_save_unlocked(const nbase_db_t *db, const char *dir_path) {
  nbase_status_t st = nbase_ensure_dir(dir_path);
  if (st != NBASE_OK) {
    return st;
  }

  char *vec_path = join_path(dir_path, "vectors.nbv");
  char *idx_path = join_path(dir_path, "hnsw.nbi");
  char *meta_path = join_path(dir_path, "meta.nbm");
  char *bm25_path = join_path(dir_path, "bm25.nbbm");
  if (!vec_path || !idx_path || !meta_path || !bm25_path) {
    free(vec_path);
    free(idx_path);
    free(meta_path);
    free(bm25_path);
    return NBASE_ERR_NOMEM;
  }

  st = nbase_store_save(db->store, vec_path);
  if (st == NBASE_OK) {
    if (db->hnsw != NULL) {
      st = nbase_hnsw_save(db->hnsw, idx_path);
    } else if (unlink(idx_path) != 0 && errno != ENOENT) {
      st = NBASE_ERR_IO;
    }
  }
  if (st == NBASE_OK) {
    if (nbase_meta_count(db->meta) > 0ULL) {
      st = nbase_meta_save(db->meta, meta_path);
    } else if (unlink(meta_path) != 0 && errno != ENOENT) {
      st = NBASE_ERR_IO;
    }
  }
  if (st == NBASE_OK) {
    if (db->bm25 != NULL && nbase_bm25_doc_count(db->bm25) > 0ULL) {
      st = nbase_bm25_save(db->bm25, bm25_path);
    } else if (unlink(bm25_path) != 0 && errno != ENOENT) {
      st = NBASE_ERR_IO;
    }
  }

  free(vec_path);
  free(idx_path);
  free(meta_path);
  free(bm25_path);
  return st;
}

static nbase_status_t open_internal(const char *dir_path, bool mmap_mode, bool attach_wal, nbase_db_t **out_db) {
  if (!dir_path || dir_path[0] == '\0' || !out_db) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (mmap_mode && attach_wal) {
    return NBASE_ERR_INVALID_ARG;
  }

  char *vec_path = join_path(dir_path, "vectors.nbv");
  char *idx_path = join_path(dir_path, "hnsw.nbi");
  char *meta_path = join_path(dir_path, "meta.nbm");
  char *bm25_path = join_path(dir_path, "bm25.nbbm");
  char *wal_path = join_path(dir_path, "wal.nbw");
  if (!vec_path || !idx_path || !meta_path || !bm25_path || !wal_path) {
    free(vec_path);
    free(idx_path);
    free(meta_path);
    free(bm25_path);
    free(wal_path);
    return NBASE_ERR_NOMEM;
  }

  nbase_store_t *store = NULL;
  nbase_status_t st = nbase_store_load(vec_path, mmap_mode, &store);
  if (st != NBASE_OK) {
    free(vec_path);
    free(idx_path);
    free(meta_path);
    free(bm25_path);
    free(wal_path);
    return st;
  }

  nbase_hnsw_t *hnsw = NULL;
  nbase_status_t idx_st = nbase_hnsw_load(idx_path, &hnsw);
  if (idx_st != NBASE_OK && idx_st != NBASE_ERR_NOT_FOUND) {
    nbase_store_destroy(store);
    free(vec_path);
    free(idx_path);
    free(meta_path);
    free(bm25_path);
    free(wal_path);
    return idx_st;
  }

  if (hnsw != NULL && (hnsw->node_count > store->slot_count || hnsw->active_nodes != store->active_count)) {
    nbase_hnsw_destroy(hnsw);
    nbase_store_destroy(store);
    free(vec_path);
    free(idx_path);
    free(meta_path);
    free(bm25_path);
    free(wal_path);
    return NBASE_ERR_CORRUPT;
  }

  nbase_meta_store_t *meta = NULL;
  nbase_status_t meta_st = nbase_meta_load(meta_path, &meta);
  if (meta_st == NBASE_ERR_NOT_FOUND) {
    meta_st = nbase_meta_create(&meta);
  }
  if (meta_st != NBASE_OK) {
    nbase_hnsw_destroy(hnsw);
    nbase_store_destroy(store);
    free(vec_path);
    free(idx_path);
    free(meta_path);
    free(bm25_path);
    free(wal_path);
    return meta_st;
  }

  nbase_bm25_t *bm25 = NULL;
  nbase_status_t bm25_st = nbase_bm25_load(bm25_path, &bm25);
  if (bm25_st != NBASE_OK && bm25_st != NBASE_ERR_NOT_FOUND) {
    nbase_meta_destroy(meta);
    nbase_hnsw_destroy(hnsw);
    nbase_store_destroy(store);
    free(vec_path);
    free(idx_path);
    free(meta_path);
    free(bm25_path);
    free(wal_path);
    return bm25_st;
  }

  free(vec_path);
  free(idx_path);
  free(meta_path);
  free(bm25_path);

  nbase_db_t *db = calloc(1, sizeof(*db));
  if (!db) {
    nbase_bm25_destroy(bm25);
    nbase_meta_destroy(meta);
    nbase_hnsw_destroy(hnsw);
    nbase_store_destroy(store);
    free(wal_path);
    return NBASE_ERR_NOMEM;
  }
  db->store = store;
  db->hnsw = hnsw;
  db->meta = meta;
  db->bm25 = bm25;
  db->enable_hnsw = (hnsw != NULL);
  db->seed = (hnsw != NULL) ? hnsw->rng_state : 0x4E42415345534545ULL;

  if (pthread_rwlock_init(&db->rwlock, NULL) != 0) {
    nbase_bm25_destroy(bm25);
    nbase_meta_destroy(meta);
    nbase_hnsw_destroy(hnsw);
    nbase_store_destroy(store);
    free(wal_path);
    free(db);
    return NBASE_ERR_NOMEM;
  }
  if (pthread_mutex_init(&db->cache_lock, NULL) != 0) {
    pthread_rwlock_destroy(&db->rwlock);
    nbase_bm25_destroy(bm25);
    nbase_meta_destroy(meta);
    nbase_hnsw_destroy(hnsw);
    nbase_store_destroy(store);
    free(wal_path);
    free(db);
    return NBASE_ERR_NOMEM;
  }

  struct stat wal_stat;
  bool wal_exists = (stat(wal_path, &wal_stat) == 0);
  if (wal_exists || attach_wal) {
    if (mmap_mode) {
      if (wal_stat.st_size > (off_t)sizeof(nbw_header_t)) {
        free(wal_path);
        nbase_close(db);
        return NBASE_ERR_UNSUPPORTED;
      }
      free(wal_path);
      *out_db = db;
      return NBASE_OK;
    }

    nbase_wal_t *wal = NULL;
    st = nbase_wal_open(wal_path, &wal);
    free(wal_path);
    if (st != NBASE_OK) {
      nbase_close(db);
      return st;
    }

    uint64_t applied = 0ULL;
    st = nbase_wal_replay(wal, db, &applied);
    if (st != NBASE_OK) {
      nbase_wal_close(wal);
      nbase_close(db);
      return st;
    }

    if (attach_wal) {
      size_t dlen = strlen(dir_path);
      db->wal_dir = malloc(dlen + 1U);
      if (!db->wal_dir) {
        nbase_wal_close(wal);
        nbase_close(db);
        return NBASE_ERR_NOMEM;
      }
      memcpy(db->wal_dir, dir_path, dlen + 1U);
      db->wal = wal;
    } else {
      nbase_wal_close(wal);
    }
  } else {
    free(wal_path);
  }

  *out_db = db;
  return NBASE_OK;
}

nbase_status_t nbase_open(const char *dir_path, nbase_db_t **out_db) {
  return open_internal(dir_path, false, false, out_db);
}

nbase_status_t nbase_open_mmap(const char *dir_path, nbase_db_t **out_db) {
  return open_internal(dir_path, true, false, out_db);
}

nbase_status_t nbase_open_wal(const char *dir_path, nbase_db_t **out_db) {
  return open_internal(dir_path, false, true, out_db);
}

nbase_status_t nbase_wal_enable(nbase_db_t *db, const char *dir_path) {
  if (!db || !dir_path || dir_path[0] == '\0') {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  if (db->store->mmap_backed) {
    pthread_rwlock_unlock(&db->rwlock);
    return NBASE_ERR_UNSUPPORTED;
  }
  if (db->wal != NULL) {
    pthread_rwlock_unlock(&db->rwlock);
    return NBASE_ERR_ALREADY_EXISTS;
  }

  nbase_status_t st = db_save_unlocked(db, dir_path);
  if (st != NBASE_OK) {
    pthread_rwlock_unlock(&db->rwlock);
    return st;
  }

  char *wal_path = join_path(dir_path, "wal.nbw");
  size_t dlen = strlen(dir_path);
  char *dir_copy = malloc(dlen + 1U);
  if (!wal_path || !dir_copy) {
    free(wal_path);
    free(dir_copy);
    pthread_rwlock_unlock(&db->rwlock);
    return NBASE_ERR_NOMEM;
  }
  memcpy(dir_copy, dir_path, dlen + 1U);

  nbase_wal_t *wal = NULL;
  st = nbase_wal_open(wal_path, &wal);
  free(wal_path);
  if (st == NBASE_OK) {
    st = nbase_wal_reset(wal);
  }
  if (st != NBASE_OK) {
    nbase_wal_close(wal);
    free(dir_copy);
    pthread_rwlock_unlock(&db->rwlock);
    return st;
  }

  db->wal = wal;
  db->wal_dir = dir_copy;
  pthread_rwlock_unlock(&db->rwlock);
  return NBASE_OK;
}

nbase_status_t nbase_wal_checkpoint(nbase_db_t *db) {
  if (!db) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  if (db->store->mmap_backed || db->wal == NULL || db->wal_dir == NULL) {
    pthread_rwlock_unlock(&db->rwlock);
    return NBASE_ERR_UNSUPPORTED;
  }

  nbase_status_t st = db_save_unlocked(db, db->wal_dir);
  if (st == NBASE_OK) {
    st = nbase_wal_reset(db->wal);
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_save(const nbase_db_t *db, const char *dir_path) {
  if (!db || !dir_path || dir_path[0] == '\0') {
    return NBASE_ERR_INVALID_ARG;
  }

  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);
  nbase_status_t st = db_save_unlocked(db, dir_path);
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

static uint64_t hash_query_cache_key(
    uint32_t dim,
    uint32_t k,
    uint32_t ef_search,
    nbase_search_mode_t mode,
    nbase_metric_t metric,
    const float *query,
    const uint8_t *fblob,
    size_t flen) {
  uint64_t h = 14695981039346656037ULL;
  uint32_t hdr_vals[5] = {dim, k, ef_search, (uint32_t)mode, (uint32_t)metric};
  const uint8_t *p = (const uint8_t *)hdr_vals;
  for (size_t i = 0U; i < sizeof(hdr_vals); i++) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ULL;
  }
  const uint8_t *qp = (const uint8_t *)query;
  size_t qbytes = (size_t)dim * sizeof(float);
  for (size_t i = 0U; i < qbytes; i++) {
    h ^= (uint64_t)qp[i];
    h *= 1099511628211ULL;
  }
  for (size_t i = 0U; i < flen; i++) {
    h ^= (uint64_t)fblob[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static bool serialize_meta_filter_key(
    const nbase_meta_filter_t *flt,
    uint8_t **out_blob,
    size_t *out_len) {
  *out_blob = NULL;
  *out_len = 0U;
  if (flt == NULL || flt->count == 0U || flt->conds == NULL) {
    return true;
  }

  size_t total = sizeof(uint32_t);
  for (uint32_t i = 0U; i < flt->count; i++) {
    const nbase_meta_cond_t *c = &flt->conds[i];
    size_t klen = c->key ? strlen(c->key) : 0U;
    total += sizeof(uint32_t) + klen + sizeof(uint32_t) * 2U;
    switch (c->value.type) {
      case NBASE_META_INT64:
        total += sizeof(int64_t);
        break;
      case NBASE_META_DOUBLE:
        total += sizeof(double);
        break;
      case NBASE_META_BOOL:
        total += 1U;
        break;
      case NBASE_META_STRING: {
        size_t slen = c->value.as.str ? strlen(c->value.as.str) : 0U;
        total += sizeof(uint32_t) + slen;
        break;
      }
      default:
        break;
    }
  }

  uint8_t *buf = malloc(total);
  if (!buf) {
    return false;
  }
  size_t pos = 0U;
  uint32_t cnt = flt->count;
  memcpy(buf + pos, &cnt, sizeof(uint32_t));
  pos += sizeof(uint32_t);

  for (uint32_t i = 0U; i < flt->count; i++) {
    const nbase_meta_cond_t *c = &flt->conds[i];
    size_t klen = c->key ? strlen(c->key) : 0U;
    uint32_t klen32 = (uint32_t)klen;
    uint32_t op32 = (uint32_t)c->op;
    uint32_t vt32 = (uint32_t)c->value.type;
    memcpy(buf + pos, &klen32, sizeof(uint32_t));
    pos += sizeof(uint32_t);
    if (klen > 0U) {
      memcpy(buf + pos, c->key, klen);
      pos += klen;
    }
    memcpy(buf + pos, &op32, sizeof(uint32_t));
    pos += sizeof(uint32_t);
    memcpy(buf + pos, &vt32, sizeof(uint32_t));
    pos += sizeof(uint32_t);
    switch (c->value.type) {
      case NBASE_META_INT64: {
        int64_t v = c->value.as.i64;
        memcpy(buf + pos, &v, sizeof(int64_t));
        pos += sizeof(int64_t);
        break;
      }
      case NBASE_META_DOUBLE: {
        double d = (c->value.as.f64 == 0.0) ? 0.0 : c->value.as.f64;
        memcpy(buf + pos, &d, sizeof(double));
        pos += sizeof(double);
        break;
      }
      case NBASE_META_BOOL: {
        buf[pos++] = c->value.as.b ? 1U : 0U;
        break;
      }
      case NBASE_META_STRING: {
        size_t slen = c->value.as.str ? strlen(c->value.as.str) : 0U;
        uint32_t slen32 = (uint32_t)slen;
        memcpy(buf + pos, &slen32, sizeof(uint32_t));
        pos += sizeof(uint32_t);
        if (slen > 0U) {
          memcpy(buf + pos, c->value.as.str, slen);
          pos += slen;
        }
        break;
      }
      default:
        break;
    }
  }

  *out_blob = buf;
  *out_len = total;
  return true;
}

static void db_cache_free_entry(db_cache_entry_t *e) {
  if (!e) {
    return;
  }
  free(e->query);
  free(e->filter_blob);
  free(e->results);
  free(e);
}

static void db_cache_unlink_locked(nbase_db_t *db, db_cache_entry_t *e) {
  if (e->prev != NULL) {
    e->prev->next = e->next;
  } else if (db->cache_head == e) {
    db->cache_head = e->next;
  }
  if (e->next != NULL) {
    e->next->prev = e->prev;
  } else if (db->cache_tail == e) {
    db->cache_tail = e->prev;
  }
  e->prev = NULL;
  e->next = NULL;
}

static void db_cache_push_front_locked(nbase_db_t *db, db_cache_entry_t *e) {
  e->prev = NULL;
  e->next = db->cache_head;
  if (db->cache_head != NULL) {
    db->cache_head->prev = e;
  }
  db->cache_head = e;
  if (db->cache_tail == NULL) {
    db->cache_tail = e;
  }
}

static void db_cache_clear_locked(nbase_db_t *db) {
  db_cache_entry_t *cur = db->cache_head;
  while (cur != NULL) {
    db_cache_entry_t *nxt = cur->next;
    db_cache_free_entry(cur);
    cur = nxt;
  }
  db->cache_head = NULL;
  db->cache_tail = NULL;
  db->cache_count = 0U;
}

static void db_cache_invalidate(nbase_db_t *db) {
  pthread_mutex_lock(&db->cache_lock);
  if (db->cache_count > 0U) {
    db_cache_clear_locked(db);
  }
  pthread_mutex_unlock(&db->cache_lock);
}

static uint64_t db_cache_memory_bytes_locked(const nbase_db_t *db) {
  uint64_t bytes = 0ULL;
  for (const db_cache_entry_t *e = db->cache_head; e != NULL; e = e->next) {
    bytes += (uint64_t)sizeof(*e);
    bytes += (uint64_t)e->dim * (uint64_t)sizeof(float);
    bytes += (uint64_t)e->filter_blob_len;
    bytes += (uint64_t)e->result_count * (uint64_t)sizeof(nbase_result_t);
  }
  return bytes;
}

static db_cache_entry_t *db_cache_find_locked(
    const nbase_db_t *db,
    uint64_t hash,
    uint32_t dim,
    uint32_t k,
    uint32_t ef_search,
    nbase_search_mode_t mode,
    nbase_metric_t metric,
    const float *query,
    const uint8_t *fblob,
    size_t flen) {
  for (db_cache_entry_t *e = db->cache_head; e != NULL; e = e->next) {
    if (e->hash != hash || e->dim != dim || e->k != k ||
        e->ef_search != ef_search || e->mode != mode || e->metric != metric ||
        e->filter_blob_len != flen) {
      continue;
    }
    if (memcmp(e->query, query, (size_t)dim * sizeof(float)) != 0) {
      continue;
    }
    if (flen > 0U && memcmp(e->filter_blob, fblob, flen) != 0) {
      continue;
    }
    return e;
  }
  return NULL;
}

static void db_cache_evict_to_cap_locked(nbase_db_t *db) {
  if (db->cache_cap == 0U) {
    db_cache_clear_locked(db);
    return;
  }
  while (db->cache_count > db->cache_cap && db->cache_tail != NULL) {
    db_cache_entry_t *victim = db->cache_tail;
    db_cache_unlink_locked(db, victim);
    db_cache_free_entry(victim);
    db->cache_count--;
  }
}

void nbase_close(nbase_db_t *db) {
  if (!db) {
    return;
  }
  if (db->wal != NULL) {
    nbase_wal_close(db->wal);
  }
  free(db->wal_dir);
  db_cache_clear_locked(db);
  pthread_mutex_destroy(&db->cache_lock);
  nbase_bm25_destroy(db->bm25);
  nbase_hnsw_destroy(db->hnsw);
  nbase_meta_destroy(db->meta);
  nbase_store_destroy(db->store);
  pthread_rwlock_destroy(&db->rwlock);
  free(db);
}

static nbase_status_t db_add_unlocked(nbase_db_t *db, uint64_t id, const float *vec, uint32_t dim) {
  if (dim != db->store->dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (db->store->mmap_backed) {
    return NBASE_ERR_UNSUPPORTED;
  }

  uint32_t slot = NBASE_INVALID_SLOT;
  nbase_status_t st = nbase_store_add(db->store, id, vec, &slot);
  if (st != NBASE_OK) {
    return st;
  }

  if (db->hnsw != NULL) {
    st = nbase_hnsw_insert(db->hnsw, db->store, slot);
    if (st != NBASE_OK) {
      nbase_store_rollback_slot(db->store, slot);
      return st;
    }
  }
  db_cache_invalidate(db);
  return NBASE_OK;
}

static nbase_status_t db_upsert_unlocked(nbase_db_t *db, uint64_t id, const float *vec, uint32_t dim) {
  if (dim != db->store->dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (db->store->mmap_backed) {
    return NBASE_ERR_UNSUPPORTED;
  }
  if (!nbase_vec_is_finite(vec, dim)) {
    return NBASE_ERR_INVALID_VECTOR;
  }

  uint32_t existing_slot = NBASE_INVALID_SLOT;
  if (!nbase_store_find_slot(db->store, id, &existing_slot)) {
    return db_add_unlocked(db, id, vec, dim);
  }
  if (db->hnsw == NULL) {
    nbase_status_t st_up = nbase_store_update_inplace(db->store, existing_slot, vec);
    if (st_up == NBASE_OK) {
      db_cache_invalidate(db);
    }
    return st_up;
  }

  uint32_t deleted_slot = NBASE_INVALID_SLOT;
  nbase_status_t st = nbase_store_delete(db->store, id, &deleted_slot);
  if (st == NBASE_OK) {
    nbase_hnsw_mark_deleted(db->hnsw, deleted_slot);
    st = db_add_unlocked(db, id, vec, dim);
    if (st == NBASE_OK) {
      db_cache_invalidate(db);
    }
  }
  return st;
}

static nbase_status_t db_delete_unlocked(nbase_db_t *db, uint64_t id) {
  if (db->store->mmap_backed) {
    return NBASE_ERR_UNSUPPORTED;
  }
  uint32_t slot = NBASE_INVALID_SLOT;
  nbase_status_t st = nbase_store_delete(db->store, id, &slot);
  if (st == NBASE_OK) {
    if (db->hnsw != NULL) {
      nbase_hnsw_mark_deleted(db->hnsw, slot);
    }
    (void)nbase_meta_delete(db->meta, id);
    if (db->bm25 != NULL) {
      (void)nbase_bm25_delete(db->bm25, id);
    }
    db_cache_invalidate(db);
  }
  return st;
}

nbase_status_t nbase_add(nbase_db_t *db, uint64_t id, const float *vec, uint32_t dim) {
  if (!db || !vec) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = db_add_unlocked(db, id, vec, dim);
  if (st == NBASE_OK && db->wal != NULL) {
    st = nbase_wal_append_vec(db->wal, NBASE_WAL_OP_ADD, id, vec, dim);
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_add_batch(
    nbase_db_t *db,
    const uint64_t *ids,
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    uint32_t num_threads) {
  if (!db || (count > 0U && (!ids || !vectors))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  pthread_rwlock_wrlock(&db->rwlock);

  if (dim != db->store->dim) {
    pthread_rwlock_unlock(&db->rwlock);
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (db->store->mmap_backed) {
    pthread_rwlock_unlock(&db->rwlock);
    return NBASE_ERR_UNSUPPORTED;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)dim, dim)) {
      pthread_rwlock_unlock(&db->rwlock);
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  uint64_t *sorted_ids = malloc((size_t)count * sizeof(uint64_t));
  uint32_t *slots = malloc((size_t)count * sizeof(uint32_t));
  if (!sorted_ids || !slots) {
    free(sorted_ids);
    free(slots);
    pthread_rwlock_unlock(&db->rwlock);
    return NBASE_ERR_NOMEM;
  }

  for (uint32_t i = 0U; i < count; i++) {
    if (nbase_store_find_slot(db->store, ids[i], NULL)) {
      free(sorted_ids);
      free(slots);
      pthread_rwlock_unlock(&db->rwlock);
      return NBASE_ERR_ALREADY_EXISTS;
    }
    sorted_ids[i] = ids[i];
  }

  qsort(sorted_ids, count, sizeof(uint64_t), cmp_u64);
  for (uint32_t i = 1U; i < count; i++) {
    if (sorted_ids[i] == sorted_ids[i - 1U]) {
      free(sorted_ids);
      free(slots);
      pthread_rwlock_unlock(&db->rwlock);
      return NBASE_ERR_ALREADY_EXISTS;
    }
  }
  free(sorted_ids);

  nbase_status_t st = NBASE_OK;
  uint32_t added = 0U;
  for (uint32_t i = 0U; i < count; i++) {
    const float *v = vectors + (size_t)i * (size_t)dim;
    st = nbase_store_add(db->store, ids[i], v, &slots[i]);
    if (st != NBASE_OK) {
      break;
    }
    added++;
  }

  if (st != NBASE_OK) {
    for (uint32_t j = 0U; j < added; j++) {
      nbase_store_rollback_slot(db->store, slots[j]);
    }
    free(slots);
    pthread_rwlock_unlock(&db->rwlock);
    return st;
  }

  if (db->wal != NULL) {
    st = nbase_wal_append_vec_batch(db->wal, ids, vectors, count, dim);
    if (st != NBASE_OK) {
      for (uint32_t j = 0U; j < added; j++) {
        nbase_store_rollback_slot(db->store, slots[j]);
      }
      free(slots);
      pthread_rwlock_unlock(&db->rwlock);
      return st;
    }
  }

  if (db->hnsw != NULL) {
    uint32_t workers = default_worker_count(num_threads);
    st = nbase_hnsw_insert_batch_mt(db->hnsw, db->store, slots, count, workers);
  }

  db_cache_invalidate(db);
  free(slots);
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_upsert(nbase_db_t *db, uint64_t id, const float *vec, uint32_t dim) {
  if (!db || !vec) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = db_upsert_unlocked(db, id, vec, dim);
  if (st == NBASE_OK && db->wal != NULL) {
    st = nbase_wal_append_vec(db->wal, NBASE_WAL_OP_UPSERT, id, vec, dim);
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_upsert_batch(
    nbase_db_t *db,
    const uint64_t *ids,
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    uint32_t num_threads) {
  if (!db || (count > 0U && (!ids || !vectors))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  pthread_rwlock_wrlock(&db->rwlock);

  if (dim != db->store->dim) {
    pthread_rwlock_unlock(&db->rwlock);
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (db->store->mmap_backed) {
    pthread_rwlock_unlock(&db->rwlock);
    return NBASE_ERR_UNSUPPORTED;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)dim, dim)) {
      pthread_rwlock_unlock(&db->rwlock);
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  uint64_t *sorted_ids = malloc((size_t)count * sizeof(uint64_t));
  uint32_t *hnsw_slots = malloc((size_t)count * sizeof(uint32_t));
  if (!sorted_ids || !hnsw_slots) {
    free(sorted_ids);
    free(hnsw_slots);
    pthread_rwlock_unlock(&db->rwlock);
    return NBASE_ERR_NOMEM;
  }
  memcpy(sorted_ids, ids, (size_t)count * sizeof(uint64_t));
  qsort(sorted_ids, count, sizeof(uint64_t), cmp_u64);
  for (uint32_t i = 1U; i < count; i++) {
    if (sorted_ids[i] == sorted_ids[i - 1U]) {
      free(sorted_ids);
      free(hnsw_slots);
      pthread_rwlock_unlock(&db->rwlock);
      return NBASE_ERR_INVALID_ARG;
    }
  }
  free(sorted_ids);

  nbase_status_t st = NBASE_OK;
  uint32_t hnsw_cnt = 0U;
  uint32_t applied = 0U;
  for (uint32_t i = 0U; i < count; i++) {
    const float *v = vectors + (size_t)i * (size_t)dim;
    uint32_t existing_slot = NBASE_INVALID_SLOT;
    if (!nbase_store_find_slot(db->store, ids[i], &existing_slot)) {
      uint32_t new_slot = NBASE_INVALID_SLOT;
      st = nbase_store_add(db->store, ids[i], v, &new_slot);
      if (st != NBASE_OK) {
        break;
      }
      if (db->hnsw != NULL) {
        hnsw_slots[hnsw_cnt++] = new_slot;
      }
    } else if (db->hnsw == NULL) {
      st = nbase_store_update_inplace(db->store, existing_slot, v);
      if (st != NBASE_OK) {
        break;
      }
    } else {
      uint32_t del_slot = NBASE_INVALID_SLOT;
      st = nbase_store_delete(db->store, ids[i], &del_slot);
      if (st != NBASE_OK) {
        break;
      }
      nbase_hnsw_mark_deleted(db->hnsw, del_slot);
      uint32_t new_slot = NBASE_INVALID_SLOT;
      st = nbase_store_add(db->store, ids[i], v, &new_slot);
      if (st != NBASE_OK) {
        break;
      }
      hnsw_slots[hnsw_cnt++] = new_slot;
    }
    applied++;
  }

  if (db->wal != NULL && applied > 0U) {
    nbase_status_t wal_st = nbase_wal_append_vec_batch_op(
        db->wal, NBASE_WAL_OP_UPSERT, ids, vectors, applied, dim);
    if (st == NBASE_OK) {
      st = wal_st;
    }
  }

  if (st == NBASE_OK && db->hnsw != NULL && hnsw_cnt > 0U) {
    uint32_t workers = default_worker_count(num_threads);
    st = nbase_hnsw_insert_batch_mt(db->hnsw, db->store, hnsw_slots, hnsw_cnt, workers);
  }

  if (applied > 0U) {
    db_cache_invalidate(db);
  }
  free(hnsw_slots);
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_delete(nbase_db_t *db, uint64_t id) {
  if (!db) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = db_delete_unlocked(db, id);
  if (st == NBASE_OK && db->wal != NULL) {
    st = nbase_wal_append_id(db->wal, NBASE_WAL_OP_DELETE, id);
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_delete_batch(
    nbase_db_t *db,
    const uint64_t *ids,
    uint32_t count,
    uint32_t *out_deleted) {
  if (!db || (count > 0U && !ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (out_deleted) {
    *out_deleted = 0U;
  }

  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = NBASE_OK;
  uint32_t deleted = 0U;
  if (db->store->mmap_backed) {
    st = NBASE_ERR_UNSUPPORTED;
  } else {
    for (uint32_t i = 0U; i < count; i++) {
      nbase_status_t ds = db_delete_unlocked(db, ids[i]);
      if (ds == NBASE_ERR_NOT_FOUND) {
        continue;
      }
      if (ds != NBASE_OK) {
        st = ds;
        break;
      }
      if (db->wal != NULL) {
        ds = nbase_wal_append_id(db->wal, NBASE_WAL_OP_DELETE, ids[i]);
        if (ds != NBASE_OK) {
          st = ds;
          break;
        }
      }
      deleted++;
    }
  }
  pthread_rwlock_unlock(&db->rwlock);
  if (out_deleted) {
    *out_deleted = deleted;
  }
  return st;
}

nbase_status_t nbase_delete_by_filter(
    nbase_db_t *db,
    const nbase_meta_filter_t *filter,
    uint32_t *out_deleted) {
  if (!db || !filter || filter->count == 0U || !filter->conds) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (out_deleted) {
    *out_deleted = 0U;
  }

  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = NBASE_OK;
  uint32_t deleted = 0U;
  if (db->store->mmap_backed) {
    st = NBASE_ERR_UNSUPPORTED;
  } else {
    for (uint32_t s = 0U; s < db->store->slot_count; s++) {
      if (db->store->slot_flags[s] != NBASE_SLOT_ACTIVE) {
        continue;
      }
      uint64_t id = db->store->ids[s];
      if (!nbase_meta_matches(db->meta, id, filter)) {
        continue;
      }
      st = db_delete_unlocked(db, id);
      if (st != NBASE_OK) {
        break;
      }
      if (db->wal != NULL) {
        st = nbase_wal_append_id(db->wal, NBASE_WAL_OP_DELETE, id);
        if (st != NBASE_OK) {
          break;
        }
      }
      deleted++;
    }
  }
  pthread_rwlock_unlock(&db->rwlock);
  if (out_deleted) {
    *out_deleted = deleted;
  }
  return st;
}

nbase_status_t nbase_get(const nbase_db_t *db, uint64_t id, float *out_vec, uint32_t dim) {
  if (!db || !out_vec) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);

  nbase_status_t st = NBASE_OK;
  if (dim != db->store->dim) {
    st = NBASE_ERR_DIM_MISMATCH;
  } else {
    uint32_t slot = NBASE_INVALID_SLOT;
    if (!nbase_store_find_slot(db->store, id, &slot)) {
      st = NBASE_ERR_NOT_FOUND;
    } else {
      const float *vec = nbase_store_get_vector(db->store, slot);
      if (!vec) {
        st = NBASE_ERR_NOT_FOUND;
      } else {
        memcpy(out_vec, vec, (size_t)dim * sizeof(float));
      }
    }
  }

  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

nbase_status_t nbase_get_batch(
    const nbase_db_t *db,
    const uint64_t *ids,
    uint32_t count,
    uint32_t dim,
    float *out_vectors,
    bool *out_found) {
  if (!db || (count > 0U && (!ids || !out_vectors || !out_found))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);

  nbase_status_t st = NBASE_OK;
  if (dim != db->store->dim) {
    st = NBASE_ERR_DIM_MISMATCH;
  } else {
    size_t vec_bytes = (size_t)dim * sizeof(float);
    for (uint32_t i = 0U; i < count; i++) {
      uint32_t slot = NBASE_INVALID_SLOT;
      float *dst = out_vectors + (size_t)i * (size_t)dim;
      if (nbase_store_find_slot(db->store, ids[i], &slot)) {
        const float *vec = nbase_store_get_vector(db->store, slot);
        if (vec != NULL) {
          memcpy(dst, vec, vec_bytes);
          out_found[i] = true;
          continue;
        }
      }
      memset(dst, 0, vec_bytes);
      out_found[i] = false;
    }
  }

  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

bool nbase_contains(const nbase_db_t *db, uint64_t id) {
  if (!db) {
    return false;
  }
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);
  bool found = nbase_store_find_slot(db->store, id, NULL);
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return found;
}

nbase_status_t nbase_count(
    const nbase_db_t *db,
    const nbase_meta_filter_t *filter,
    uint64_t *out_count) {
  if (!db || !out_count || (filter != NULL && filter->count > 0U && !filter->conds)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0ULL;
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);
  if (!filter || filter->count == 0U) {
    *out_count = db->store->active_count;
  } else {
    uint64_t matched = 0ULL;
    for (uint32_t s = 0U; s < db->store->slot_count; s++) {
      if (db->store->slot_flags[s] == NBASE_SLOT_ACTIVE &&
          nbase_meta_matches(db->meta, db->store->ids[s], filter)) {
        matched++;
      }
    }
    *out_count = matched;
  }
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return NBASE_OK;
}

nbase_status_t nbase_list_ids(
    const nbase_db_t *db,
    uint64_t offset,
    uint32_t limit,
    uint64_t *out_ids,
    uint32_t *out_count,
    uint64_t *out_total) {
  if (!db || !out_count || (limit > 0U && !out_ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  if (out_total) {
    *out_total = 0ULL;
  }

  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);

  uint64_t total = db->store->active_count;
  if (out_total) {
    *out_total = total;
  }
  if (limit == 0U || total == 0ULL || offset >= total) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_OK;
  }

  uint64_t *all = malloc((size_t)total * sizeof(uint64_t));
  if (!all) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_ERR_NOMEM;
  }

  size_t pos = 0U;
  for (uint32_t s = 0U; s < db->store->slot_count && pos < (size_t)total; s++) {
    if (db->store->slot_flags[s] == NBASE_SLOT_ACTIVE) {
      all[pos++] = db->store->ids[s];
    }
  }
  pthread_rwlock_unlock(&mutable_db->rwlock);

  if (pos > 1U) {
    qsort(all, pos, sizeof(uint64_t), cmp_u64);
  }
  if (offset < (uint64_t)pos) {
    uint64_t rem = (uint64_t)pos - offset;
    uint32_t ret = (rem < (uint64_t)limit) ? (uint32_t)rem : limit;
    if (ret > 0U) {
      memcpy(out_ids, all + (size_t)offset, (size_t)ret * sizeof(uint64_t));
    }
    *out_count = ret;
  }
  free(all);
  return NBASE_OK;
}

nbase_status_t nbase_set_meta(nbase_db_t *db, uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
  if (!db) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st;
  if (db->store->mmap_backed) {
    st = NBASE_ERR_UNSUPPORTED;
  } else if (!nbase_store_find_slot(db->store, id, NULL)) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    st = nbase_meta_set(db->meta, id, fields, count);
    if (st == NBASE_OK) {
      db_cache_invalidate(db);
    }
    if (st == NBASE_OK && db->wal != NULL) {
      st = nbase_wal_append_meta(db->wal, NBASE_WAL_OP_SET_META, id, fields, count);
    }
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_update_meta(nbase_db_t *db, uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
  if (!db) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st;
  if (db->store->mmap_backed) {
    st = NBASE_ERR_UNSUPPORTED;
  } else if (!nbase_store_find_slot(db->store, id, NULL)) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    st = nbase_meta_update(db->meta, id, fields, count);
    if (st == NBASE_OK) {
      db_cache_invalidate(db);
    }
    if (st == NBASE_OK && db->wal != NULL) {
      st = nbase_wal_append_meta(db->wal, NBASE_WAL_OP_UPDATE_META, id, fields, count);
    }
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

static nbase_status_t db_meta_batch_op(
    nbase_db_t *db,
    const nbase_meta_batch_item_t *items,
    uint32_t count,
    bool is_update) {
  if (!db || (count > 0U && !items)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }
  for (uint32_t i = 0U; i < count; i++) {
    nbase_status_t vst = nbase_meta_validate_fields(items[i].fields, items[i].field_count);
    if (vst != NBASE_OK) {
      return vst;
    }
  }

  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = NBASE_OK;
  if (db->store->mmap_backed) {
    st = NBASE_ERR_UNSUPPORTED;
  } else {
    for (uint32_t i = 0U; i < count; i++) {
      if (!nbase_store_find_slot(db->store, items[i].id, NULL)) {
        st = NBASE_ERR_NOT_FOUND;
        break;
      }
    }
    uint32_t applied = 0U;
    for (uint32_t i = 0U; i < count && st == NBASE_OK; i++) {
      st = is_update
               ? nbase_meta_update(db->meta, items[i].id, items[i].fields, items[i].field_count)
               : nbase_meta_set(db->meta, items[i].id, items[i].fields, items[i].field_count);
      if (st == NBASE_OK) {
        applied++;
      }
    }
    if (applied > 0U) {
      db_cache_invalidate(db);
    }
    if (st == NBASE_OK && db->wal != NULL) {
      st = nbase_wal_append_meta_batch(
          db->wal,
          is_update ? NBASE_WAL_OP_UPDATE_META : NBASE_WAL_OP_SET_META,
          items,
          count);
    }
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_set_meta_batch(
    nbase_db_t *db,
    const nbase_meta_batch_item_t *items,
    uint32_t count) {
  return db_meta_batch_op(db, items, count, false);
}

nbase_status_t nbase_update_meta_batch(
    nbase_db_t *db,
    const nbase_meta_batch_item_t *items,
    uint32_t count) {
  return db_meta_batch_op(db, items, count, true);
}

nbase_status_t nbase_get_meta(const nbase_db_t *db, uint64_t id, const char *key, nbase_meta_value_t *out_val, char *str_buf, size_t str_buf_len) {
  if (!db || !key || key[0] == '\0' || !out_val) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);
  nbase_status_t st;
  if (!nbase_store_find_slot(db->store, id, NULL)) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    st = nbase_meta_get(db->meta, id, key, out_val, str_buf, str_buf_len);
  }
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

nbase_status_t nbase_delete_meta(nbase_db_t *db, uint64_t id) {
  if (!db) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st;
  if (db->store->mmap_backed) {
    st = NBASE_ERR_UNSUPPORTED;
  } else if (!nbase_store_find_slot(db->store, id, NULL)) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    st = nbase_meta_delete(db->meta, id);
    if (st == NBASE_OK) {
      db_cache_invalidate(db);
    }
    if (st == NBASE_OK && db->wal != NULL) {
      st = nbase_wal_append_id(db->wal, NBASE_WAL_OP_DELETE_META, id);
    }
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_delete_meta_batch(
    nbase_db_t *db,
    const uint64_t *ids,
    uint32_t count,
    uint32_t *out_deleted) {
  if (!db || (count > 0U && !ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (out_deleted != NULL) {
    *out_deleted = 0U;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = NBASE_OK;
  uint32_t deleted = 0U;
  if (db->store->mmap_backed) {
    st = NBASE_ERR_UNSUPPORTED;
  } else {
    for (uint32_t i = 0U; i < count; i++) {
      if (!nbase_store_find_slot(db->store, ids[i], NULL)) {
        continue;
      }
      nbase_status_t ds = nbase_meta_delete(db->meta, ids[i]);
      if (ds == NBASE_ERR_NOT_FOUND) {
        continue;
      }
      if (ds != NBASE_OK) {
        st = ds;
        break;
      }
      if (db->wal != NULL) {
        ds = nbase_wal_append_id(db->wal, NBASE_WAL_OP_DELETE_META, ids[i]);
        if (ds != NBASE_OK) {
          st = ds;
          break;
        }
      }
      deleted++;
    }
    if (deleted > 0U) {
      db_cache_invalidate(db);
    }
  }
  pthread_rwlock_unlock(&db->rwlock);
  if (out_deleted != NULL) {
    *out_deleted = deleted;
  }
  return st;
}

nbase_status_t nbase_set_text(nbase_db_t *db, uint64_t id, const char *text) {
  if (!db || !text) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st;
  if (db->store->mmap_backed) {
    st = NBASE_ERR_UNSUPPORTED;
  } else if (!nbase_store_find_slot(db->store, id, NULL)) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    st = db_ensure_bm25_unlocked(db);
    if (st == NBASE_OK) {
      st = nbase_bm25_index(db->bm25, id, text);
    }
    if (st == NBASE_OK && db->wal != NULL) {
      st = nbase_wal_append_text(db->wal, id, text);
    }
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_set_text_batch(
    nbase_db_t *db,
    const uint64_t *ids,
    const char *const *texts,
    uint32_t count) {
  if (!db || (count > 0U && (!ids || !texts))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!texts[i]) {
      return NBASE_ERR_INVALID_ARG;
    }
  }

  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = NBASE_OK;
  if (db->store->mmap_backed) {
    st = NBASE_ERR_UNSUPPORTED;
  } else {
    for (uint32_t i = 0U; i < count; i++) {
      if (!nbase_store_find_slot(db->store, ids[i], NULL)) {
        st = NBASE_ERR_NOT_FOUND;
        break;
      }
    }
    if (st == NBASE_OK) {
      st = db_ensure_bm25_unlocked(db);
    }
    for (uint32_t i = 0U; i < count && st == NBASE_OK; i++) {
      st = nbase_bm25_index(db->bm25, ids[i], texts[i]);
    }
    if (st == NBASE_OK && db->wal != NULL) {
      st = nbase_wal_append_text_batch(db->wal, ids, texts, count);
    }
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_get_text(
    const nbase_db_t *db,
    uint64_t id,
    char *out_buf,
    size_t buf_len,
    size_t *out_len) {
  if (!db || (!out_buf && !out_len)) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);
  nbase_status_t st;
  if (!nbase_store_find_slot(db->store, id, NULL) || db->bm25 == NULL) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    st = nbase_bm25_get(db->bm25, id, out_buf, buf_len, out_len);
  }
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

nbase_status_t nbase_delete_text(nbase_db_t *db, uint64_t id) {
  if (!db) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st;
  if (db->store->mmap_backed) {
    st = NBASE_ERR_UNSUPPORTED;
  } else if (!nbase_store_find_slot(db->store, id, NULL) || db->bm25 == NULL) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    st = nbase_bm25_delete(db->bm25, id);
    if (st == NBASE_OK && db->wal != NULL) {
      st = nbase_wal_append_id(db->wal, NBASE_WAL_OP_DELETE_TEXT, id);
    }
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_delete_text_batch(
    nbase_db_t *db,
    const uint64_t *ids,
    uint32_t count,
    uint32_t *out_deleted) {
  if (!db || (count > 0U && !ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (out_deleted != NULL) {
    *out_deleted = 0U;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = NBASE_OK;
  uint32_t deleted = 0U;
  if (db->store->mmap_backed) {
    st = NBASE_ERR_UNSUPPORTED;
  } else if (db->bm25 != NULL) {
    for (uint32_t i = 0U; i < count; i++) {
      if (!nbase_store_find_slot(db->store, ids[i], NULL)) {
        continue;
      }
      nbase_status_t ds = nbase_bm25_delete(db->bm25, ids[i]);
      if (ds == NBASE_ERR_NOT_FOUND) {
        continue;
      }
      if (ds != NBASE_OK) {
        st = ds;
        break;
      }
      if (db->wal != NULL) {
        ds = nbase_wal_append_id(db->wal, NBASE_WAL_OP_DELETE_TEXT, ids[i]);
        if (ds != NBASE_OK) {
          st = ds;
          break;
        }
      }
      deleted++;
    }
  }
  pthread_rwlock_unlock(&db->rwlock);
  if (out_deleted != NULL) {
    *out_deleted = deleted;
  }
  return st;
}

static nbase_status_t db_search_unlocked(
    const nbase_db_t *db,
    const float *query,
    uint32_t dim,
    uint32_t k,
    const nbase_search_opts_t *eff,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (dim != db->store->dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (!nbase_vec_is_finite(query, dim)) {
    return NBASE_ERR_INVALID_VECTOR;
  }

  nbase_metric_t metric = eff->use_metric_override ? eff->metric : db->store->metric;
  if (!nbase_metric_is_valid(metric)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (eff->mode != NBASE_SEARCH_EXACT &&
      eff->mode != NBASE_SEARCH_HNSW &&
      eff->mode != NBASE_SEARCH_AUTO) {
    return NBASE_ERR_INVALID_ARG;
  }

  bool can_cache = (eff->filter == NULL && k > 0U && !eff->skip_cache);
  uint8_t *fblob = NULL;
  size_t flen = 0U;
  uint64_t cache_hash = 0ULL;
  bool have_key = false;
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;

  if (can_cache) {
    pthread_mutex_lock(&mutable_db->cache_lock);
    if (mutable_db->cache_cap > 0U &&
        serialize_meta_filter_key(eff->meta_filter, &fblob, &flen)) {
      cache_hash = hash_query_cache_key(
          dim, k, eff->ef_search, eff->mode, metric, query, fblob, flen);
      have_key = true;
      db_cache_entry_t *hit = db_cache_find_locked(
          mutable_db, cache_hash, dim, k, eff->ef_search, eff->mode, metric,
          query, fblob, flen);
      if (hit != NULL) {
        if (mutable_db->cache_head != hit) {
          db_cache_unlink_locked(mutable_db, hit);
          db_cache_push_front_locked(mutable_db, hit);
        }
        mutable_db->cache_hits++;
        if (hit->result_count > 0U) {
          memcpy(out_results, hit->results, (size_t)hit->result_count * sizeof(nbase_result_t));
        }
        *out_count = hit->result_count;
        pthread_mutex_unlock(&mutable_db->cache_lock);
        free(fblob);
        return NBASE_OK;
      }
      mutable_db->cache_misses++;
    }
    pthread_mutex_unlock(&mutable_db->cache_lock);
  }

  combined_filter_ctx_t combined;
  nbase_filter_fn filter = eff->filter;
  void *filter_ctx = eff->filter_ctx;
  if (eff->meta_filter != NULL && eff->meta_filter->count > 0U) {
    combined.meta = db->meta;
    combined.meta_filter = eff->meta_filter;
    combined.user_filter = eff->filter;
    combined.user_ctx = eff->filter_ctx;
    filter = combined_filter_cb;
    filter_ctx = &combined;
  }

  nbase_status_t st;
  switch (eff->mode) {
    case NBASE_SEARCH_EXACT:
      st = nbase_store_exact_search(db->store, query, k, metric, filter, filter_ctx, out_results, out_count);
      break;
    case NBASE_SEARCH_HNSW:
      if (db->hnsw == NULL) {
        st = NBASE_ERR_UNSUPPORTED;
      } else {
        st = nbase_hnsw_search(db->hnsw, db->store, query, k, eff->ef_search, metric, filter, filter_ctx, out_results, out_count);
      }
      break;
    case NBASE_SEARCH_AUTO:
      if (db->hnsw != NULL) {
        st = nbase_hnsw_search(db->hnsw, db->store, query, k, eff->ef_search, metric, filter, filter_ctx, out_results, out_count);
      } else {
        st = nbase_store_exact_search(db->store, query, k, metric, filter, filter_ctx, out_results, out_count);
      }
      break;
    default:
      st = NBASE_ERR_INVALID_ARG;
      break;
  }

  if (st == NBASE_OK && can_cache) {
    pthread_mutex_lock(&mutable_db->cache_lock);
    if (mutable_db->cache_cap > 0U) {
      if (!have_key && serialize_meta_filter_key(eff->meta_filter, &fblob, &flen)) {
        cache_hash = hash_query_cache_key(
            dim, k, eff->ef_search, eff->mode, metric, query, fblob, flen);
        have_key = true;
      }
      if (have_key &&
          db_cache_find_locked(
              mutable_db, cache_hash, dim, k, eff->ef_search, eff->mode, metric,
              query, fblob, flen) == NULL) {
        db_cache_entry_t *ent = calloc(1, sizeof(*ent));
        float *qcopy = malloc((size_t)dim * sizeof(float));
        nbase_result_t *rcopy = (*out_count > 0U)
                                    ? malloc((size_t)(*out_count) * sizeof(nbase_result_t))
                                    : NULL;
        if (ent && qcopy && (*out_count == 0U || rcopy != NULL)) {
          memcpy(qcopy, query, (size_t)dim * sizeof(float));
          if (*out_count > 0U) {
            memcpy(rcopy, out_results, (size_t)(*out_count) * sizeof(nbase_result_t));
          }
          ent->hash = cache_hash;
          ent->query = qcopy;
          ent->dim = dim;
          ent->k = k;
          ent->ef_search = eff->ef_search;
          ent->mode = eff->mode;
          ent->metric = metric;
          ent->filter_blob = fblob;
          ent->filter_blob_len = flen;
          ent->results = rcopy;
          ent->result_count = *out_count;
          fblob = NULL;

          while (mutable_db->cache_count >= mutable_db->cache_cap &&
                 mutable_db->cache_tail != NULL) {
            db_cache_entry_t *victim = mutable_db->cache_tail;
            db_cache_unlink_locked(mutable_db, victim);
            db_cache_free_entry(victim);
            mutable_db->cache_count--;
          }
          db_cache_push_front_locked(mutable_db, ent);
          mutable_db->cache_count++;
        } else {
          free(rcopy);
          free(qcopy);
          free(ent);
        }
      }
    }
    pthread_mutex_unlock(&mutable_db->cache_lock);
  }

  free(fblob);
  return st;
}

nbase_status_t nbase_search(
    const nbase_db_t *db,
    const float *query,
    uint32_t dim,
    uint32_t k,
    const nbase_search_opts_t *opts,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!db || !query || !out_count || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_search_opts_t eff = opts ? *opts : nbase_search_opts_default();
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);
  nbase_status_t st = db_search_unlocked(db, query, dim, k, &eff, out_results, out_count);
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

nbase_status_t nbase_search_radius(
    const nbase_db_t *db,
    const float *query,
    uint32_t dim,
    float max_distance,
    uint32_t max_results,
    const nbase_search_opts_t *opts,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!db || !query || !out_count || (max_results > 0U && !out_results) || !isfinite(max_distance)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;

  nbase_search_opts_t eff = opts ? *opts : nbase_search_opts_default();
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);

  if (dim != db->store->dim) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (!nbase_vec_is_finite(query, dim)) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_ERR_INVALID_VECTOR;
  }

  nbase_metric_t metric = eff.use_metric_override ? eff.metric : db->store->metric;
  if (!nbase_metric_is_valid(metric) ||
      (eff.mode != NBASE_SEARCH_EXACT && eff.mode != NBASE_SEARCH_HNSW && eff.mode != NBASE_SEARCH_AUTO)) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_ERR_INVALID_ARG;
  }
  if (max_results == 0U) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_OK;
  }

  combined_filter_ctx_t combined;
  nbase_filter_fn filter = eff.filter;
  void *filter_ctx = eff.filter_ctx;
  if (eff.meta_filter != NULL && eff.meta_filter->count > 0U) {
    combined.meta = db->meta;
    combined.meta_filter = eff.meta_filter;
    combined.user_filter = eff.filter;
    combined.user_ctx = eff.filter_ctx;
    filter = combined_filter_cb;
    filter_ctx = &combined;
  }

  nbase_status_t st;
  if (eff.mode == NBASE_SEARCH_HNSW) {
    if (db->hnsw == NULL) {
      st = NBASE_ERR_UNSUPPORTED;
    } else {
      uint32_t fetch_k = (eff.ef_search > max_results) ? eff.ef_search : max_results;
      nbase_result_t *tmp = malloc((size_t)fetch_k * sizeof(nbase_result_t));
      if (!tmp) {
        st = NBASE_ERR_NOMEM;
      } else {
        uint32_t got = 0U;
        st = nbase_hnsw_search(
            db->hnsw, db->store, query, fetch_k, eff.ef_search, metric, filter, filter_ctx, tmp, &got);
        if (st == NBASE_OK) {
          uint32_t keep = 0U;
          for (uint32_t i = 0U; i < got && keep < max_results; i++) {
            if (tmp[i].distance <= max_distance) {
              out_results[keep++] = tmp[i];
            }
          }
          *out_count = keep;
        }
        free(tmp);
      }
    }
  } else {
    st = nbase_store_exact_search_radius(
        db->store, query, max_results, max_distance, metric, filter, filter_ctx, out_results, out_count);
  }

  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

nbase_status_t nbase_find_similar(
    const nbase_db_t *db,
    uint64_t id,
    uint32_t k,
    const nbase_search_opts_t *opts,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!db || !out_count || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;

  nbase_search_opts_t eff = opts ? *opts : nbase_search_opts_default();
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);

  uint32_t slot = NBASE_INVALID_SLOT;
  if (!nbase_store_find_slot(db->store, id, &slot)) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_ERR_NOT_FOUND;
  }
  if (k == 0U || db->store->active_count <= 1ULL) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_OK;
  }

  uint32_t max_active = (db->store->active_count > (uint64_t)UINT32_MAX - 1ULL)
                            ? UINT32_MAX
                            : (uint32_t)db->store->active_count;
  uint32_t fetch_k = (k < max_active) ? (k + 1U) : max_active;
  nbase_result_t *tmp = malloc((size_t)fetch_k * sizeof(nbase_result_t));
  if (!tmp) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_ERR_NOMEM;
  }

  const float *qvec = nbase_store_get_vector(db->store, slot);
  uint32_t got = 0U;
  nbase_status_t st = db_search_unlocked(db, qvec, db->store->dim, fetch_k, &eff, tmp, &got);
  pthread_rwlock_unlock(&mutable_db->rwlock);

  if (st == NBASE_OK) {
    uint32_t out_i = 0U;
    for (uint32_t i = 0U; i < got && out_i < k; i++) {
      if (tmp[i].id == id) {
        continue;
      }
      out_results[out_i++] = tmp[i];
    }
    *out_count = out_i;
  }
  free(tmp);
  return st;
}

nbase_status_t nbase_foreach_meta(
    const nbase_db_t *db,
    uint64_t id,
    nbase_meta_iter_fn cb,
    void *user_ctx) {
  if (!db || !cb) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);
  nbase_status_t st;
  if (!nbase_store_find_slot(db->store, id, NULL)) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    st = nbase_meta_foreach(db->meta, id, cb, user_ctx);
  }
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

nbase_status_t nbase_query_meta(
    const nbase_db_t *db,
    const nbase_meta_filter_t *filter,
    uint32_t limit,
    uint64_t *out_ids,
    uint32_t *out_count) {
  if (!db || !out_count || (limit > 0U && !out_ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  if (limit == 0U) {
    return NBASE_OK;
  }

  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);

  uint64_t active = db->store->active_count;
  if (active == 0ULL) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_OK;
  }
  uint64_t *matched = malloc((size_t)active * sizeof(uint64_t));
  if (!matched) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_ERR_NOMEM;
  }

  size_t m_cnt = 0U;
  for (uint32_t s = 0U; s < db->store->slot_count && m_cnt < (size_t)active; s++) {
    if (db->store->slot_flags[s] != NBASE_SLOT_ACTIVE) {
      continue;
    }
    uint64_t id = db->store->ids[s];
    if (filter == NULL || filter->count == 0U || nbase_meta_matches(db->meta, id, filter)) {
      matched[m_cnt++] = id;
    }
  }
  pthread_rwlock_unlock(&mutable_db->rwlock);

  if (m_cnt > 1U) {
    qsort(matched, m_cnt, sizeof(uint64_t), cmp_u64);
  }
  uint32_t ret = (m_cnt < (size_t)limit) ? (uint32_t)m_cnt : limit;
  if (ret > 0U) {
    memcpy(out_ids, matched, (size_t)ret * sizeof(uint64_t));
  }
  *out_count = ret;
  free(matched);
  return NBASE_OK;
}

static nbase_status_t db_search_text_unlocked(
    const nbase_db_t *db,
    const char *query,
    uint32_t k,
    const nbase_meta_filter_t *meta_filter,
    nbase_scored_id_t *out_results,
    uint32_t *out_count) {
  *out_count = 0U;
  if (k == 0U || db->bm25 == NULL) {
    return NBASE_OK;
  }

  combined_filter_ctx_t combined;
  nbase_filter_fn filter = NULL;
  void *filter_ctx = NULL;
  if (meta_filter != NULL && meta_filter->count > 0U) {
    combined.meta = db->meta;
    combined.meta_filter = meta_filter;
    combined.user_filter = NULL;
    combined.user_ctx = NULL;
    filter = combined_filter_cb;
    filter_ctx = &combined;
  }

  return nbase_bm25_search_filtered(
      db->bm25, query, k, filter, filter_ctx, out_results, out_count);
}

static void *text_batch_worker(void *raw) {
  text_batch_worker_ctx_t *ctx = (text_batch_worker_ctx_t *)raw;
  while (true) {
    if (atomic_load_explicit(&ctx->first_err, memory_order_relaxed) != (int)NBASE_OK) {
      break;
    }
    uint32_t idx = atomic_fetch_add_explicit(&ctx->next_idx, 1U, memory_order_relaxed);
    if (idx >= ctx->query_count) {
      break;
    }
    nbase_scored_id_t *dst =
        (ctx->k > 0U) ? (ctx->out_results + (size_t)idx * (size_t)ctx->k) : NULL;
    nbase_status_t st = db_search_text_unlocked(
        ctx->db, ctx->queries[idx], ctx->k, ctx->meta_filter, dst, &ctx->out_counts[idx]);
    if (st != NBASE_OK) {
      int expected = (int)NBASE_OK;
      (void)atomic_compare_exchange_strong_explicit(
          &ctx->first_err, &expected, (int)st, memory_order_relaxed, memory_order_relaxed);
      break;
    }
  }
  return NULL;
}

nbase_status_t nbase_search_text(
    const nbase_db_t *db,
    const char *query,
    uint32_t k,
    const nbase_meta_filter_t *meta_filter,
    nbase_scored_id_t *out_results,
    uint32_t *out_count) {
  if (!db || !query || !out_count || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;

  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);
  nbase_status_t st = db_search_text_unlocked(db, query, k, meta_filter, out_results, out_count);
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

nbase_status_t nbase_search_text_batch(
    const nbase_db_t *db,
    const char *const *queries,
    uint32_t query_count,
    uint32_t k,
    uint32_t num_threads,
    const nbase_meta_filter_t *meta_filter,
    nbase_scored_id_t *out_results,
    uint32_t *out_counts) {
  if (!db || (query_count > 0U && (!queries || !out_counts || (k > 0U && !out_results)))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (query_count == 0U) {
    return NBASE_OK;
  }
  for (uint32_t i = 0U; i < query_count; i++) {
    if (!queries[i]) {
      return NBASE_ERR_INVALID_ARG;
    }
  }
  memset(out_counts, 0, (size_t)query_count * sizeof(uint32_t));

  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);
  if (k == 0U || db->bm25 == NULL) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_OK;
  }

  uint32_t eff_threads = (num_threads == 0U) ? 4U : num_threads;
  if (eff_threads > query_count) {
    eff_threads = query_count;
  }
  if (eff_threads > 64U) {
    eff_threads = 64U;
  }

  if (eff_threads <= 1U) {
    nbase_status_t st = NBASE_OK;
    for (uint32_t i = 0U; i < query_count; i++) {
      st = db_search_text_unlocked(
          db,
          queries[i],
          k,
          meta_filter,
          out_results + (size_t)i * (size_t)k,
          &out_counts[i]);
      if (st != NBASE_OK) {
        break;
      }
    }
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return st;
  }

  text_batch_worker_ctx_t ctx;
  ctx.db = db;
  ctx.queries = queries;
  ctx.query_count = query_count;
  ctx.k = k;
  ctx.meta_filter = meta_filter;
  ctx.out_results = out_results;
  ctx.out_counts = out_counts;
  atomic_init(&ctx.next_idx, 0U);
  atomic_init(&ctx.first_err, (int)NBASE_OK);

  pthread_t tids[64];
  uint32_t spawned = 0U;
  for (uint32_t t = 0U; t < eff_threads; t++) {
    if (pthread_create(&tids[t], NULL, text_batch_worker, &ctx) != 0) {
      break;
    }
    spawned++;
  }
  if (spawned < eff_threads) {
    (void)text_batch_worker(&ctx);
  }
  for (uint32_t t = 0U; t < spawned; t++) {
    pthread_join(tids[t], NULL);
  }

  nbase_status_t st = (nbase_status_t)atomic_load_explicit(&ctx.first_err, memory_order_relaxed);
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

static nbase_status_t db_search_hybrid_unlocked(
    const nbase_db_t *db,
    const float *query_vec,
    uint32_t dim,
    const char *query_text,
    uint32_t k,
    float alpha,
    float rrf_k,
    const nbase_search_opts_t *eff,
    nbase_scored_id_t *out_results,
    uint32_t *out_count) {
  bool has_vec = (query_vec != NULL);
  bool has_text = (query_text != NULL && query_text[0] != '\0');
  *out_count = 0U;
  if (!has_vec && !has_text) {
    return NBASE_ERR_INVALID_ARG;
  }

  if (has_vec) {
    if (dim != db->store->dim) {
      return NBASE_ERR_DIM_MISMATCH;
    }
    if (!nbase_vec_is_finite(query_vec, dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }
  if (k == 0U) {
    return NBASE_OK;
  }

  uint32_t fetch_k = (k <= 2500U) ? (k * 4U) : 10000U;
  if (fetch_k < 64U) {
    fetch_k = 64U;
  }

  nbase_result_t *vec_res = has_vec ? malloc((size_t)fetch_k * sizeof(nbase_result_t)) : NULL;
  bool run_text = (has_text && db->bm25 != NULL);
  nbase_scored_id_t *text_res =
      run_text ? malloc((size_t)fetch_k * sizeof(nbase_scored_id_t)) : NULL;
  if ((has_vec && !vec_res) || (run_text && !text_res)) {
    free(vec_res);
    free(text_res);
    return NBASE_ERR_NOMEM;
  }

  nbase_status_t st = NBASE_OK;
  uint32_t vec_got = 0U;
  uint32_t text_got = 0U;

  if (has_vec) {
    st = db_search_unlocked(db, query_vec, dim, fetch_k, eff, vec_res, &vec_got);
  }
  if (st == NBASE_OK && run_text) {
    combined_filter_ctx_t combined;
    nbase_filter_fn filter = NULL;
    void *filter_ctx = NULL;
    if ((eff->meta_filter != NULL && eff->meta_filter->count > 0U) || eff->filter != NULL) {
      combined.meta = db->meta;
      combined.meta_filter = eff->meta_filter;
      combined.user_filter = eff->filter;
      combined.user_ctx = eff->filter_ctx;
      filter = combined_filter_cb;
      filter_ctx = &combined;
    }
    st = nbase_bm25_search_filtered(
        db->bm25, query_text, fetch_k, filter, filter_ctx, text_res, &text_got);
  }

  if (st == NBASE_OK) {
    st = nbase_hybrid_rrf(
        vec_got > 0U ? vec_res : NULL,
        vec_got,
        text_got > 0U ? text_res : NULL,
        text_got,
        alpha,
        rrf_k,
        k,
        out_results,
        out_count);
  }

  free(vec_res);
  free(text_res);
  return st;
}

static void *hybrid_batch_worker(void *raw) {
  hybrid_batch_worker_ctx_t *ctx = (hybrid_batch_worker_ctx_t *)raw;
  while (true) {
    if (atomic_load_explicit(&ctx->first_err, memory_order_relaxed) != (int)NBASE_OK) {
      break;
    }
    uint32_t idx = atomic_fetch_add_explicit(&ctx->next_idx, 1U, memory_order_relaxed);
    if (idx >= ctx->query_count) {
      break;
    }
    const float *qv = ctx->query_vectors
                          ? (ctx->query_vectors + (size_t)idx * (size_t)ctx->dim)
                          : NULL;
    const char *qt = ctx->query_texts ? ctx->query_texts[idx] : NULL;
    nbase_scored_id_t *dst =
        (ctx->k > 0U) ? (ctx->out_results + (size_t)idx * (size_t)ctx->k) : NULL;
    nbase_status_t st = db_search_hybrid_unlocked(
        ctx->db,
        qv,
        ctx->dim,
        qt,
        ctx->k,
        ctx->alpha,
        ctx->rrf_k,
        &ctx->opts,
        dst,
        &ctx->out_counts[idx]);
    if (st != NBASE_OK) {
      int expected = (int)NBASE_OK;
      (void)atomic_compare_exchange_strong_explicit(
          &ctx->first_err, &expected, (int)st, memory_order_relaxed, memory_order_relaxed);
      break;
    }
  }
  return NULL;
}

nbase_status_t nbase_search_hybrid(
    const nbase_db_t *db,
    const float *query_vec,
    uint32_t dim,
    const char *query_text,
    uint32_t k,
    float alpha,
    float rrf_k,
    const nbase_search_opts_t *opts,
    nbase_scored_id_t *out_results,
    uint32_t *out_count) {
  bool has_vec = (query_vec != NULL);
  bool has_text = (query_text != NULL && query_text[0] != '\0');
  if (!db || !out_count || (k > 0U && !out_results) || (!has_vec && !has_text) ||
      isnan(alpha) || alpha < 0.0f || alpha > 1.0f || !isfinite(rrf_k) || rrf_k <= 0.0f) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;

  nbase_search_opts_t eff = opts ? *opts : nbase_search_opts_default();
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);
  nbase_status_t st = db_search_hybrid_unlocked(
      db, query_vec, dim, query_text, k, alpha, rrf_k, &eff, out_results, out_count);
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

nbase_status_t nbase_search_hybrid_batch(
    const nbase_db_t *db,
    const float *query_vectors,
    uint32_t dim,
    const char *const *query_texts,
    uint32_t query_count,
    uint32_t k,
    float alpha,
    float rrf_k,
    const nbase_search_opts_t *opts,
    nbase_scored_id_t *out_results,
    uint32_t *out_counts) {
  if (!db || isnan(alpha) || alpha < 0.0f || alpha > 1.0f ||
      !isfinite(rrf_k) || rrf_k <= 0.0f ||
      (query_count > 0U &&
       (!out_counts || (!query_vectors && !query_texts) || (k > 0U && !out_results)))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (query_count == 0U) {
    return NBASE_OK;
  }
  if (query_vectors == NULL) {
    for (uint32_t i = 0U; i < query_count; i++) {
      if (!query_texts[i] || query_texts[i][0] == '\0') {
        return NBASE_ERR_INVALID_ARG;
      }
    }
  }
  memset(out_counts, 0, (size_t)query_count * sizeof(uint32_t));

  nbase_search_opts_t eff = opts ? *opts : nbase_search_opts_default();
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);

  if (query_vectors != NULL && dim != db->store->dim) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_ERR_DIM_MISMATCH;
  }

  if (eff.num_threads <= 1U || query_count <= 1U) {
    nbase_status_t st = NBASE_OK;
    for (uint32_t i = 0U; i < query_count; i++) {
      const float *qv = query_vectors ? (query_vectors + (size_t)i * (size_t)dim) : NULL;
      const char *qt = query_texts ? query_texts[i] : NULL;
      nbase_scored_id_t *dst = (k > 0U) ? (out_results + (size_t)i * (size_t)k) : NULL;
      st = db_search_hybrid_unlocked(
          db, qv, dim, qt, k, alpha, rrf_k, &eff, dst, &out_counts[i]);
      if (st != NBASE_OK) {
        break;
      }
    }
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return st;
  }

  uint32_t workers = (eff.num_threads < query_count) ? eff.num_threads : query_count;
  if (workers > 64U) {
    workers = 64U;
  }

  hybrid_batch_worker_ctx_t ctx;
  ctx.db = db;
  ctx.query_vectors = query_vectors;
  ctx.dim = dim;
  ctx.query_texts = query_texts;
  ctx.query_count = query_count;
  ctx.k = k;
  ctx.alpha = alpha;
  ctx.rrf_k = rrf_k;
  ctx.opts = eff;
  ctx.out_results = out_results;
  ctx.out_counts = out_counts;
  atomic_init(&ctx.next_idx, 0U);
  atomic_init(&ctx.first_err, (int)NBASE_OK);

  pthread_t tids[64];
  uint32_t spawned = 0U;
  for (uint32_t t = 0U; t < workers; t++) {
    if (pthread_create(&tids[t], NULL, hybrid_batch_worker, &ctx) != 0) {
      break;
    }
    spawned++;
  }
  if (spawned < workers) {
    (void)hybrid_batch_worker(&ctx);
  }
  for (uint32_t t = 0U; t < spawned; t++) {
    pthread_join(tids[t], NULL);
  }

  nbase_status_t st = (nbase_status_t)atomic_load_explicit(&ctx.first_err, memory_order_relaxed);
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

static void *batch_worker(void *raw) {
  batch_worker_ctx_t *ctx = (batch_worker_ctx_t *)raw;
  while (true) {
    if (atomic_load_explicit(&ctx->first_err, memory_order_relaxed) != (int)NBASE_OK) {
      break;
    }
    uint32_t idx = atomic_fetch_add_explicit(&ctx->next_idx, 1U, memory_order_relaxed);
    if (idx >= ctx->query_count) {
      break;
    }
    const float *q = ctx->queries + (size_t)idx * (size_t)ctx->dim;
    nbase_result_t *dst = (ctx->k > 0U) ? (ctx->out_results + (size_t)idx * (size_t)ctx->k) : NULL;
    nbase_status_t st = db_search_unlocked(ctx->db, q, ctx->dim, ctx->k, &ctx->opts, dst, &ctx->out_counts[idx]);
    if (st != NBASE_OK) {
      int expected = (int)NBASE_OK;
      (void)atomic_compare_exchange_strong_explicit(&ctx->first_err, &expected, (int)st, memory_order_relaxed, memory_order_relaxed);
      break;
    }
  }
  return NULL;
}

nbase_status_t nbase_search_batch(
    const nbase_db_t *db,
    const float *queries,
    uint32_t query_count,
    uint32_t dim,
    uint32_t k,
    const nbase_search_opts_t *opts,
    nbase_result_t *out_results,
    uint32_t *out_counts) {
  if (!db || !out_counts || (query_count > 0U && !queries) || (query_count > 0U && k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }

  nbase_search_opts_t eff = opts ? *opts : nbase_search_opts_default();
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);

  if (dim != db->store->dim) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (query_count == 0U) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_OK;
  }

  if (eff.num_threads <= 1U || query_count <= 1U) {
    nbase_status_t st = NBASE_OK;
    for (uint32_t i = 0U; i < query_count; i++) {
      const float *q = queries + (size_t)i * (size_t)dim;
      nbase_result_t *dst = (k > 0U) ? (out_results + (size_t)i * (size_t)k) : NULL;
      st = db_search_unlocked(db, q, dim, k, &eff, dst, &out_counts[i]);
      if (st != NBASE_OK) {
        break;
      }
    }
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return st;
  }

  uint32_t workers = (eff.num_threads < query_count) ? eff.num_threads : query_count;
  batch_worker_ctx_t ctx;
  ctx.db = db;
  ctx.queries = queries;
  ctx.query_count = query_count;
  ctx.dim = dim;
  ctx.k = k;
  ctx.opts = eff;
  ctx.out_results = out_results;
  ctx.out_counts = out_counts;
  atomic_init(&ctx.next_idx, 0U);
  atomic_init(&ctx.first_err, (int)NBASE_OK);

  pthread_t *tids = malloc((size_t)workers * sizeof(pthread_t));
  if (!tids) {
    pthread_rwlock_unlock(&mutable_db->rwlock);
    return NBASE_ERR_NOMEM;
  }

  uint32_t spawned = 0U;
  for (uint32_t t = 0U; t < workers; t++) {
    if (pthread_create(&tids[t], NULL, batch_worker, &ctx) != 0) {
      break;
    }
    spawned++;
  }

  /* Execute remaining work on the caller thread if fewer than requested threads spawned. */
  if (spawned < workers) {
    (void)batch_worker(&ctx);
  }
  for (uint32_t t = 0U; t < spawned; t++) {
    pthread_join(tids[t], NULL);
  }
  free(tids);

  nbase_status_t st = (nbase_status_t)atomic_load_explicit(&ctx.first_err, memory_order_relaxed);
  pthread_rwlock_unlock(&mutable_db->rwlock);
  return st;
}

static nbase_status_t db_rebuild_hnsw_unlocked(nbase_db_t *db, uint32_t num_threads) {
  if (db->store->mmap_backed || !db->enable_hnsw || db->hnsw == NULL) {
    return NBASE_ERR_UNSUPPORTED;
  }

  nbase_hnsw_t *fresh = NULL;
  nbase_status_t st = nbase_hnsw_create(db->hnsw->m, db->hnsw->ef_construction, db->hnsw->ef_search, db->seed, &fresh);
  if (st != NBASE_OK) {
    return st;
  }

  uint32_t active_cnt = 0U;
  for (uint32_t slot = 0U; slot < db->store->slot_count; slot++) {
    if (db->store->slot_flags[slot] == NBASE_SLOT_ACTIVE) {
      active_cnt++;
    }
  }

  if (active_cnt > 0U) {
    uint32_t *active_slots = malloc((size_t)active_cnt * sizeof(uint32_t));
    if (!active_slots) {
      nbase_hnsw_destroy(fresh);
      return NBASE_ERR_NOMEM;
    }
    uint32_t pos = 0U;
    for (uint32_t slot = 0U; slot < db->store->slot_count; slot++) {
      if (db->store->slot_flags[slot] == NBASE_SLOT_ACTIVE) {
        active_slots[pos++] = slot;
      }
    }
    st = nbase_hnsw_insert_batch_mt(fresh, db->store, active_slots, active_cnt, num_threads);
    free(active_slots);
    if (st != NBASE_OK) {
      nbase_hnsw_destroy(fresh);
      return st;
    }
  }

  nbase_hnsw_destroy(db->hnsw);
  db->hnsw = fresh;
  db_cache_invalidate(db);
  return NBASE_OK;
}

nbase_status_t nbase_rebuild_hnsw(nbase_db_t *db) {
  if (!db) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = db_rebuild_hnsw_unlocked(db, 1U);
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_rebuild_hnsw_mt(nbase_db_t *db, uint32_t num_threads) {
  if (!db) {
    return NBASE_ERR_INVALID_ARG;
  }
  uint32_t workers = default_worker_count(num_threads);
  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = db_rebuild_hnsw_unlocked(db, workers);
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_compact(nbase_db_t *db) {
  if (!db) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&db->rwlock);
  nbase_status_t st = nbase_store_compact(db->store);
  if (st == NBASE_OK && db->hnsw != NULL) {
    st = db_rebuild_hnsw_unlocked(db, 1U);
  }
  if (st == NBASE_OK) {
    db_cache_invalidate(db);
  }
  if (st == NBASE_OK && db->wal != NULL) {
    st = nbase_wal_append_compact(db->wal);
  }
  pthread_rwlock_unlock(&db->rwlock);
  return st;
}

nbase_status_t nbase_cache_configure(nbase_db_t *db, uint32_t max_entries) {
  if (!db) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&db->cache_lock);
  db->cache_cap = max_entries;
  db_cache_evict_to_cap_locked(db);
  pthread_mutex_unlock(&db->cache_lock);
  return NBASE_OK;
}

nbase_status_t nbase_cache_clear(nbase_db_t *db) {
  if (!db) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&db->cache_lock);
  db_cache_clear_locked(db);
  pthread_mutex_unlock(&db->cache_lock);
  return NBASE_OK;
}

void nbase_db_rdlock(const nbase_db_t *db) {
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);
}

void nbase_db_unlock(const nbase_db_t *db) {
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_unlock(&mutable_db->rwlock);
}

const nbase_store_t *nbase_db_store(const nbase_db_t *db) {
  return db->store;
}

const nbase_hnsw_t *nbase_db_hnsw(const nbase_db_t *db) {
  return db->hnsw;
}

const nbase_meta_store_t *nbase_db_meta(const nbase_db_t *db) {
  return db->meta;
}

bool nbase_db_hnsw_enabled(const nbase_db_t *db) {
  return db->enable_hnsw;
}

nbase_status_t nbase_db_replay_vec(nbase_db_t *db, nbase_wal_op_t op, uint64_t id, const float *vec, uint32_t dim) {
  if (op == NBASE_WAL_OP_ADD) {
    return db_add_unlocked(db, id, vec, dim);
  }
  return db_upsert_unlocked(db, id, vec, dim);
}

nbase_status_t nbase_db_replay_delete(nbase_db_t *db, uint64_t id) {
  return db_delete_unlocked(db, id);
}

nbase_status_t nbase_db_replay_meta(nbase_db_t *db, nbase_wal_op_t op, uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
  if (!nbase_store_find_slot(db->store, id, NULL)) {
    return NBASE_ERR_NOT_FOUND;
  }
  if (op == NBASE_WAL_OP_SET_META) {
    return nbase_meta_set(db->meta, id, fields, count);
  }
  if (op == NBASE_WAL_OP_UPDATE_META) {
    return nbase_meta_update(db->meta, id, fields, count);
  }
  return nbase_meta_delete(db->meta, id);
}

nbase_status_t nbase_db_replay_text(nbase_db_t *db, nbase_wal_op_t op, uint64_t id, const char *text) {
  if (!nbase_store_find_slot(db->store, id, NULL)) {
    return NBASE_ERR_NOT_FOUND;
  }
  if (op == NBASE_WAL_OP_SET_TEXT) {
    nbase_status_t st = db_ensure_bm25_unlocked(db);
    if (st != NBASE_OK) {
      return st;
    }
    return nbase_bm25_index(db->bm25, id, text ? text : "");
  }
  if (db->bm25 == NULL) {
    return NBASE_ERR_NOT_FOUND;
  }
  return nbase_bm25_delete(db->bm25, id);
}

nbase_status_t nbase_db_replay_compact(nbase_db_t *db) {
  nbase_status_t st = nbase_store_compact(db->store);
  if (st == NBASE_OK && db->hnsw != NULL) {
    st = db_rebuild_hnsw_unlocked(db, 1U);
  }
  return st;
}

nbase_status_t nbase_stats(const nbase_db_t *db, nbase_stats_t *out_stats) {
  if (!db || !out_stats) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_db_t *mutable_db = (nbase_db_t *)(uintptr_t)db;
  pthread_rwlock_rdlock(&mutable_db->rwlock);

  memset(out_stats, 0, sizeof(*out_stats));
  out_stats->dim = db->store->dim;
  out_stats->metric = db->store->metric;
  out_stats->active_vectors = db->store->active_count;
  out_stats->total_slots = db->store->slot_count;
  out_stats->tombstones = db->store->tombstone_count;
  out_stats->mmap_backed = db->store->mmap_backed;
  out_stats->hnsw_enabled = (db->hnsw != NULL);
  if (db->hnsw != NULL) {
    out_stats->hnsw_max_level = db->hnsw->max_level;
    out_stats->hnsw_active_nodes = db->hnsw->active_nodes;
    out_stats->hnsw_deleted_nodes = db->hnsw->deleted_nodes;
  } else {
    out_stats->hnsw_max_level = -1;
    out_stats->hnsw_active_nodes = 0ULL;
    out_stats->hnsw_deleted_nodes = 0ULL;
  }
  out_stats->text_docs = db->bm25 ? nbase_bm25_doc_count(db->bm25) : 0ULL;
  pthread_mutex_lock(&mutable_db->cache_lock);
  out_stats->cache_capacity = mutable_db->cache_cap;
  out_stats->cache_entries = mutable_db->cache_count;
  out_stats->cache_hits = mutable_db->cache_hits;
  out_stats->cache_misses = mutable_db->cache_misses;
  uint64_t cache_mem = db_cache_memory_bytes_locked(mutable_db);
  pthread_mutex_unlock(&mutable_db->cache_lock);
  out_stats->memory_bytes = (uint64_t)sizeof(*db) +
                            nbase_store_memory_bytes(db->store) +
                            nbase_hnsw_memory_bytes(db->hnsw) +
                            nbase_meta_memory_bytes(db->meta) +
                            nbase_bm25_memory_bytes(db->bm25) +
                            cache_mem;

  pthread_rwlock_unlock(&mutable_db->rwlock);
  return NBASE_OK;
}
