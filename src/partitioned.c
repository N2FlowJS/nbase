#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PDB_MAP_EMPTY 0U
#define PDB_MAP_OCCUPIED 1U
#define PDB_MAP_DELETED 2U

typedef struct pdb_part {
  uint32_t part_id;
  uint64_t active_vectors;
  nbase_db_t *db;
  bool dirty;
  struct pdb_part *lru_prev;
  struct pdb_part *lru_next;
} pdb_part_t;

typedef struct pdb_id_map {
  size_t cap;
  size_t count;
  size_t tombstones;
  uint64_t *keys;
  uint32_t *part_ids;
  uint8_t *states;
} pdb_id_map_t;

struct nbase_pdb {
  char *dir_path;
  nbase_pdb_config_t cfg;
  uint32_t next_partition_id;
  uint32_t active_partition_id;
  uint64_t total_vectors;
  pdb_part_t **parts;
  uint32_t part_count;
  uint32_t part_cap;
  pdb_part_t *lru_head;
  pdb_part_t *lru_tail;
  uint32_t loaded_count;
  uint64_t eviction_count;
  uint64_t cache_hits;
  uint64_t cache_misses;
  pdb_id_map_t id_map;
  bool manifest_dirty;
  pthread_mutex_t mu;
};

static int cmp_pdb_u64_asc(const void *pa, const void *pb) {
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

static int cmp_pdb_scored_desc(const void *pa, const void *pb) {
  const nbase_scored_id_t *a = (const nbase_scored_id_t *)pa;
  const nbase_scored_id_t *b = (const nbase_scored_id_t *)pb;
  if (a->score > b->score) {
    return -1;
  }
  if (a->score < b->score) {
    return 1;
  }
  if (a->id < b->id) {
    return -1;
  }
  if (a->id > b->id) {
    return 1;
  }
  return 0;
}

typedef struct pdb_part_ref {
  uint32_t part_id;
  uint32_t orig_idx;
} pdb_part_ref_t;

typedef struct pdb_weighted_cand {
  uint64_t id;
  float weighted_dist;
  float orig_dist;
} pdb_weighted_cand_t;

static int cmp_pdb_part_ref_asc(const void *pa, const void *pb) {
  const pdb_part_ref_t *a = (const pdb_part_ref_t *)pa;
  const pdb_part_ref_t *b = (const pdb_part_ref_t *)pb;
  if (a->part_id < b->part_id) {
    return -1;
  }
  if (a->part_id > b->part_id) {
    return 1;
  }
  if (a->orig_idx < b->orig_idx) {
    return -1;
  }
  if (a->orig_idx > b->orig_idx) {
    return 1;
  }
  return 0;
}

static int cmp_pdb_weighted_asc(const void *pa, const void *pb) {
  const pdb_weighted_cand_t *a = (const pdb_weighted_cand_t *)pa;
  const pdb_weighted_cand_t *b = (const pdb_weighted_cand_t *)pb;
  if (a->weighted_dist < b->weighted_dist) {
    return -1;
  }
  if (a->weighted_dist > b->weighted_dist) {
    return 1;
  }
  if (a->orig_dist < b->orig_dist) {
    return -1;
  }
  if (a->orig_dist > b->orig_dist) {
    return 1;
  }
  if (a->id < b->id) {
    return -1;
  }
  if (a->id > b->id) {
    return 1;
  }
  return 0;
}

static inline uint64_t pdb_hash_u64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

static bool pdb_id_map_init(pdb_id_map_t *map, size_t initial_cap) {
  size_t cap = 64U;
  while (cap < initial_cap) {
    if (cap > SIZE_MAX / 2U) {
      return false;
    }
    cap *= 2U;
  }
  map->keys = calloc(cap, sizeof(uint64_t));
  map->part_ids = calloc(cap, sizeof(uint32_t));
  map->states = calloc(cap, sizeof(uint8_t));
  if (!map->keys || !map->part_ids || !map->states) {
    free(map->keys);
    free(map->part_ids);
    free(map->states);
    memset(map, 0, sizeof(*map));
    return false;
  }
  map->cap = cap;
  map->count = 0U;
  map->tombstones = 0U;
  return true;
}

static void pdb_id_map_free(pdb_id_map_t *map) {
  free(map->keys);
  free(map->part_ids);
  free(map->states);
  memset(map, 0, sizeof(*map));
}

static bool pdb_id_map_rehash(pdb_id_map_t *map, size_t new_cap) {
  pdb_id_map_t next;
  if (!pdb_id_map_init(&next, new_cap)) {
    return false;
  }
  size_t mask = next.cap - 1U;
  for (size_t i = 0U; i < map->cap; i++) {
    if (map->states[i] == PDB_MAP_OCCUPIED) {
      size_t idx = (size_t)pdb_hash_u64(map->keys[i]) & mask;
      while (next.states[idx] == PDB_MAP_OCCUPIED) {
        idx = (idx + 1U) & mask;
      }
      next.keys[idx] = map->keys[i];
      next.part_ids[idx] = map->part_ids[i];
      next.states[idx] = PDB_MAP_OCCUPIED;
      next.count++;
    }
  }
  pdb_id_map_free(map);
  *map = next;
  return true;
}

static bool pdb_id_map_find(const pdb_id_map_t *map, uint64_t key, uint32_t *out_part_id) {
  if (map->cap == 0U) {
    return false;
  }
  size_t mask = map->cap - 1U;
  size_t idx = (size_t)pdb_hash_u64(key) & mask;
  for (size_t probe = 0U; probe < map->cap; probe++) {
    uint8_t st = map->states[idx];
    if (st == PDB_MAP_EMPTY) {
      return false;
    }
    if (st == PDB_MAP_OCCUPIED && map->keys[idx] == key) {
      if (out_part_id) {
        *out_part_id = map->part_ids[idx];
      }
      return true;
    }
    idx = (idx + 1U) & mask;
  }
  return false;
}

static nbase_status_t pdb_id_map_insert(pdb_id_map_t *map, uint64_t key, uint32_t part_id) {
  if ((map->count + map->tombstones + 1U) * 100U >= map->cap * 65U) {
    size_t target = (map->count * 2U >= map->cap) ? map->cap * 2U : map->cap;
    if (!pdb_id_map_rehash(map, target)) {
      return NBASE_ERR_NOMEM;
    }
  }
  size_t mask = map->cap - 1U;
  size_t idx = (size_t)pdb_hash_u64(key) & mask;
  size_t first_deleted = SIZE_MAX;

  for (size_t probe = 0U; probe < map->cap; probe++) {
    uint8_t st = map->states[idx];
    if (st == PDB_MAP_EMPTY) {
      size_t dst = (first_deleted != SIZE_MAX) ? first_deleted : idx;
      if (map->states[dst] == PDB_MAP_DELETED) {
        map->tombstones--;
      }
      map->keys[dst] = key;
      map->part_ids[dst] = part_id;
      map->states[dst] = PDB_MAP_OCCUPIED;
      map->count++;
      return NBASE_OK;
    }
    if (st == PDB_MAP_OCCUPIED && map->keys[idx] == key) {
      return NBASE_ERR_ALREADY_EXISTS;
    }
    if (st == PDB_MAP_DELETED && first_deleted == SIZE_MAX) {
      first_deleted = idx;
    }
    idx = (idx + 1U) & mask;
  }
  return NBASE_ERR_NOMEM;
}

static bool pdb_id_map_delete(pdb_id_map_t *map, uint64_t key) {
  if (map->cap == 0U) {
    return false;
  }
  size_t mask = map->cap - 1U;
  size_t idx = (size_t)pdb_hash_u64(key) & mask;
  for (size_t probe = 0U; probe < map->cap; probe++) {
    uint8_t st = map->states[idx];
    if (st == PDB_MAP_EMPTY) {
      return false;
    }
    if (st == PDB_MAP_OCCUPIED && map->keys[idx] == key) {
      map->states[idx] = PDB_MAP_DELETED;
      map->count--;
      map->tombstones++;
      return true;
    }
    idx = (idx + 1U) & mask;
  }
  return false;
}

static char *pdb_part_dir(const nbase_pdb_t *pdb, uint32_t part_id) {
  size_t len = strlen(pdb->dir_path) + 32U;
  char *out = malloc(len);
  if (!out) {
    return NULL;
  }
  snprintf(out, len, "%s/p_%u", pdb->dir_path, part_id);
  return out;
}

static char *pdb_manifest_path(const nbase_pdb_t *pdb) {
  size_t len = strlen(pdb->dir_path) + 32U;
  char *out = malloc(len);
  if (!out) {
    return NULL;
  }
  snprintf(out, len, "%s/manifest.nbp", pdb->dir_path);
  return out;
}

static pdb_part_t *pdb_find_part(const nbase_pdb_t *pdb, uint32_t part_id) {
  for (uint32_t i = 0U; i < pdb->part_count; i++) {
    if (pdb->parts[i]->part_id == part_id) {
      return pdb->parts[i];
    }
  }
  return NULL;
}

static void pdb_lru_unlink(nbase_pdb_t *pdb, pdb_part_t *part) {
  if (part->lru_prev != NULL) {
    part->lru_prev->lru_next = part->lru_next;
  } else if (pdb->lru_head == part) {
    pdb->lru_head = part->lru_next;
  }
  if (part->lru_next != NULL) {
    part->lru_next->lru_prev = part->lru_prev;
  } else if (pdb->lru_tail == part) {
    pdb->lru_tail = part->lru_prev;
  }
  part->lru_prev = NULL;
  part->lru_next = NULL;
}

static void pdb_lru_push_front(nbase_pdb_t *pdb, pdb_part_t *part) {
  part->lru_prev = NULL;
  part->lru_next = pdb->lru_head;
  if (pdb->lru_head != NULL) {
    pdb->lru_head->lru_prev = part;
  }
  pdb->lru_head = part;
  if (pdb->lru_tail == NULL) {
    pdb->lru_tail = part;
  }
}

static void pdb_lru_touch(nbase_pdb_t *pdb, pdb_part_t *part) {
  if (pdb->lru_head == part) {
    return;
  }
  pdb_lru_unlink(pdb, part);
  pdb_lru_push_front(pdb, part);
}

static nbase_status_t pdb_save_part(const nbase_pdb_t *pdb, pdb_part_t *part) {
  if (part->db == NULL || !part->dirty) {
    return NBASE_OK;
  }
  char *part_dir = pdb_part_dir(pdb, part->part_id);
  if (!part_dir) {
    return NBASE_ERR_NOMEM;
  }
  nbase_status_t st = nbase_save(part->db, part_dir);
  free(part_dir);
  if (st == NBASE_OK) {
    part->dirty = false;
  }
  return st;
}

static nbase_status_t pdb_evict_one(nbase_pdb_t *pdb, uint32_t pinned_part_id) {
  pdb_part_t *victim = pdb->lru_tail;
  while (victim != NULL && victim->part_id == pinned_part_id) {
    victim = victim->lru_prev;
  }
  if (victim == NULL) {
    return NBASE_ERR_INVALID_ARG;
  }

  /* Persist dirty partition before closing so evicted vectors are never lost. */
  nbase_status_t st = pdb_save_part(pdb, victim);
  if (st != NBASE_OK) {
    return st;
  }

  nbase_close(victim->db);
  victim->db = NULL;
  pdb_lru_unlink(pdb, victim);
  pdb->loaded_count--;
  pdb->eviction_count++;
  return NBASE_OK;
}

static nbase_config_t pdb_make_part_config(const nbase_pdb_t *pdb, uint32_t part_id) {
  nbase_config_t cfg = nbase_config_default(pdb->cfg.dim);
  cfg.metric = pdb->cfg.metric;
  cfg.initial_capacity = (pdb->cfg.partition_capacity < 64U) ? pdb->cfg.partition_capacity : 64U;
  cfg.enable_hnsw = pdb->cfg.enable_hnsw;
  cfg.hnsw_m = pdb->cfg.hnsw_m;
  cfg.hnsw_ef_construction = pdb->cfg.hnsw_ef_construction;
  cfg.hnsw_ef_search = pdb->cfg.hnsw_ef_search;
  cfg.seed = pdb->cfg.seed + (uint64_t)part_id + 1ULL;
  return cfg;
}

static nbase_status_t pdb_load_partition(nbase_pdb_t *pdb, pdb_part_t *part) {
  if (part->db != NULL) {
    pdb->cache_hits++;
    pdb_lru_touch(pdb, part);
    return NBASE_OK;
  }

  pdb->cache_misses++;
  while (pdb->loaded_count >= pdb->cfg.max_active_partitions) {
    nbase_status_t st = pdb_evict_one(pdb, part->part_id);
    if (st != NBASE_OK) {
      return st;
    }
  }

  char *part_dir = pdb_part_dir(pdb, part->part_id);
  if (!part_dir) {
    return NBASE_ERR_NOMEM;
  }

  nbase_db_t *db = NULL;
  nbase_status_t st = nbase_open(part_dir, &db);
  free(part_dir);

  if (st == NBASE_ERR_NOT_FOUND) {
    nbase_config_t pcfg = pdb_make_part_config(pdb, part->part_id);
    st = nbase_create(&pcfg, &db);
    if (st == NBASE_OK) {
      part->dirty = true;
    }
  }
  if (st != NBASE_OK) {
    return st;
  }

  part->db = db;
  pdb_lru_push_front(pdb, part);
  pdb->loaded_count++;
  return NBASE_OK;
}

static nbase_status_t pdb_append_part_slot(nbase_pdb_t *pdb, pdb_part_t *part) {
  if (pdb->part_count == pdb->part_cap) {
    uint32_t new_cap = (pdb->part_cap == 0U) ? 8U : pdb->part_cap * 2U;
    pdb_part_t **grown = realloc(pdb->parts, (size_t)new_cap * sizeof(pdb_part_t *));
    if (!grown) {
      return NBASE_ERR_NOMEM;
    }
    pdb->parts = grown;
    pdb->part_cap = new_cap;
  }
  pdb->parts[pdb->part_count++] = part;
  return NBASE_OK;
}

static nbase_status_t pdb_create_partition(nbase_pdb_t *pdb, pdb_part_t **out_part) {
  while (pdb->loaded_count >= pdb->cfg.max_active_partitions) {
    nbase_status_t st = pdb_evict_one(pdb, UINT32_MAX);
    if (st != NBASE_OK) {
      return st;
    }
  }

  pdb_part_t *part = calloc(1, sizeof(*part));
  if (!part) {
    return NBASE_ERR_NOMEM;
  }
  part->part_id = pdb->next_partition_id;

  nbase_config_t pcfg = pdb_make_part_config(pdb, part->part_id);
  nbase_status_t st = nbase_create(&pcfg, &part->db);
  if (st != NBASE_OK) {
    free(part);
    return st;
  }

  st = pdb_append_part_slot(pdb, part);
  if (st != NBASE_OK) {
    nbase_close(part->db);
    free(part);
    return st;
  }

  pdb->next_partition_id++;
  part->dirty = true;
  pdb_lru_push_front(pdb, part);
  pdb->loaded_count++;
  pdb->active_partition_id = part->part_id;
  pdb->manifest_dirty = true;

  if (out_part) {
    *out_part = part;
  }
  return NBASE_OK;
}

static nbase_status_t pdb_save_manifest(nbase_pdb_t *pdb) {
  char *mpath = pdb_manifest_path(pdb);
  if (!mpath) {
    return NBASE_ERR_NOMEM;
  }

  size_t part_entry_bytes = sizeof(uint32_t) + sizeof(uint64_t);
  size_t id_entry_bytes = sizeof(uint64_t) + sizeof(uint32_t);
  size_t payload_bytes = (size_t)pdb->part_count * part_entry_bytes + (size_t)pdb->id_map.count * id_entry_bytes;

  uint8_t *buf = NULL;
  if (payload_bytes > 0U) {
    buf = malloc(payload_bytes);
    if (!buf) {
      free(mpath);
      return NBASE_ERR_NOMEM;
    }
    size_t pos = 0U;
    for (uint32_t i = 0U; i < pdb->part_count; i++) {
      const pdb_part_t *p = pdb->parts[i];
      memcpy(buf + pos, &p->part_id, sizeof(uint32_t));
      pos += sizeof(uint32_t);
      memcpy(buf + pos, &p->active_vectors, sizeof(uint64_t));
      pos += sizeof(uint64_t);
    }
    for (size_t i = 0U; i < pdb->id_map.cap; i++) {
      if (pdb->id_map.states[i] == PDB_MAP_OCCUPIED) {
        memcpy(buf + pos, &pdb->id_map.keys[i], sizeof(uint64_t));
        pos += sizeof(uint64_t);
        memcpy(buf + pos, &pdb->id_map.part_ids[i], sizeof(uint32_t));
        pos += sizeof(uint32_t);
      }
    }
  }

  nbp_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBP_MAGIC, 8);
  hdr.version = NBASE_NBP_VERSION;
  hdr.dim = pdb->cfg.dim;
  hdr.metric = (uint32_t)pdb->cfg.metric;
  hdr.partition_capacity = pdb->cfg.partition_capacity;
  hdr.max_active_partitions = pdb->cfg.max_active_partitions;
  hdr.enable_hnsw = pdb->cfg.enable_hnsw ? 1U : 0U;
  hdr.hnsw_m = pdb->cfg.hnsw_m;
  hdr.hnsw_ef_construction = pdb->cfg.hnsw_ef_construction;
  hdr.hnsw_ef_search = pdb->cfg.hnsw_ef_search;
  hdr.next_partition_id = pdb->next_partition_id;
  hdr.active_partition_id = pdb->active_partition_id;
  hdr.partition_count = pdb->part_count;
  hdr.seed = pdb->cfg.seed;
  hdr.total_vectors = pdb->total_vectors;
  hdr.payload_bytes = (uint64_t)payload_bytes;
  hdr.payload_crc32 = nbase_crc32(buf, payload_bytes);
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nbp_header_t, header_crc32));

  nbase_io_chunk_t chunk = {buf, payload_bytes};
  nbase_status_t st = nbase_write_file_atomic(mpath, &hdr, sizeof(hdr), &chunk, 1);
  free(buf);
  free(mpath);
  if (st == NBASE_OK) {
    pdb->manifest_dirty = false;
  }
  return st;
}

static nbase_status_t pdb_save_all_unlocked(nbase_pdb_t *pdb) {
  nbase_status_t st = nbase_ensure_dir(pdb->dir_path);
  if (st != NBASE_OK) {
    return st;
  }
  for (uint32_t i = 0U; i < pdb->part_count; i++) {
    st = pdb_save_part(pdb, pdb->parts[i]);
    if (st != NBASE_OK) {
      return st;
    }
  }
  return pdb_save_manifest(pdb);
}

static void pdb_destroy_unlocked(nbase_pdb_t *pdb) {
  for (uint32_t i = 0U; i < pdb->part_count; i++) {
    if (pdb->parts[i]) {
      nbase_close(pdb->parts[i]->db);
      free(pdb->parts[i]);
    }
  }
  free(pdb->parts);
  pdb_id_map_free(&pdb->id_map);
  free(pdb->dir_path);
  pthread_mutex_destroy(&pdb->mu);
  free(pdb);
}

nbase_pdb_config_t nbase_pdb_config_default(uint32_t dim) {
  nbase_pdb_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.dim = dim;
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.partition_capacity = 100000U;
  cfg.max_active_partitions = 3U;
  cfg.enable_hnsw = true;
  cfg.hnsw_m = 16U;
  cfg.hnsw_ef_construction = 200U;
  cfg.hnsw_ef_search = 50U;
  cfg.seed = 0ULL;
  return cfg;
}

nbase_pdb_search_opts_t nbase_pdb_search_opts_default(void) {
  nbase_pdb_search_opts_t opts;
  memset(&opts, 0, sizeof(opts));
  opts.search = nbase_search_opts_default();
  opts.partition_ids = NULL;
  opts.partition_id_count = 0U;
  opts.search_all_partitions = false;
  return opts;
}

static nbase_status_t pdb_load_manifest(nbase_pdb_t *pdb, const char *mpath) {
  int fd = open(mpath, O_RDONLY);
  if (fd < 0) {
    return (errno == ENOENT) ? NBASE_ERR_NOT_FOUND : NBASE_ERR_IO;
  }

  struct stat st_buf;
  if (fstat(fd, &st_buf) != 0) {
    close(fd);
    return NBASE_ERR_IO;
  }
  if (st_buf.st_size < (off_t)sizeof(nbp_header_t)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;

  nbp_header_t hdr;
  nbase_status_t st = nbase_read_exact(fd, &hdr, sizeof(hdr));
  if (st != NBASE_OK) {
    close(fd);
    return st;
  }

  if (memcmp(hdr.magic, NBASE_NBP_MAGIC, 8) != 0 || hdr.version != NBASE_NBP_VERSION) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.header_crc32 != nbase_crc32(&hdr, offsetof(nbp_header_t, header_crc32))) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.dim == 0U || hdr.dim > NBASE_MAX_DIM || !nbase_metric_is_valid((nbase_metric_t)hdr.metric) || hdr.partition_capacity == 0U || hdr.max_active_partitions == 0U || hdr.enable_hnsw > 1U || hdr.partition_count == 0U) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.enable_hnsw == 1U && (hdr.hnsw_m < 2U || hdr.hnsw_ef_construction == 0U || hdr.hnsw_ef_search == 0U)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint64_t expected_payload = (uint64_t)hdr.partition_count * (sizeof(uint32_t) + sizeof(uint64_t)) + hdr.total_vectors * (sizeof(uint64_t) + sizeof(uint32_t));
  if (hdr.payload_bytes != expected_payload || file_size != sizeof(nbp_header_t) + expected_payload || expected_payload > (uint64_t)SIZE_MAX) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint8_t *buf = malloc((size_t)expected_payload);
  if (!buf) {
    close(fd);
    return NBASE_ERR_NOMEM;
  }
  st = nbase_read_exact(fd, buf, (size_t)expected_payload);
  close(fd);
  if (st != NBASE_OK) {
    free(buf);
    return st;
  }
  if (nbase_crc32(buf, (size_t)expected_payload) != hdr.payload_crc32) {
    free(buf);
    return NBASE_ERR_CORRUPT;
  }

  pdb->cfg.dim = hdr.dim;
  pdb->cfg.metric = (nbase_metric_t)hdr.metric;
  pdb->cfg.partition_capacity = hdr.partition_capacity;
  pdb->cfg.max_active_partitions = hdr.max_active_partitions;
  pdb->cfg.enable_hnsw = (hdr.enable_hnsw != 0U);
  pdb->cfg.hnsw_m = hdr.hnsw_m;
  pdb->cfg.hnsw_ef_construction = hdr.hnsw_ef_construction;
  pdb->cfg.hnsw_ef_search = hdr.hnsw_ef_search;
  pdb->cfg.seed = hdr.seed;
  pdb->next_partition_id = hdr.next_partition_id;
  pdb->active_partition_id = hdr.active_partition_id;
  pdb->total_vectors = hdr.total_vectors;

  size_t pos = 0U;
  uint64_t sum_vecs = 0ULL;
  for (uint32_t i = 0U; i < hdr.partition_count; i++) {
    uint32_t pid = 0U;
    uint64_t count = 0ULL;
    memcpy(&pid, buf + pos, sizeof(uint32_t));
    pos += sizeof(uint32_t);
    memcpy(&count, buf + pos, sizeof(uint64_t));
    pos += sizeof(uint64_t);

    if (pid >= pdb->next_partition_id || pdb_find_part(pdb, pid) != NULL) {
      free(buf);
      return NBASE_ERR_CORRUPT;
    }
    pdb_part_t *part = calloc(1, sizeof(*part));
    if (!part) {
      free(buf);
      return NBASE_ERR_NOMEM;
    }
    part->part_id = pid;
    part->active_vectors = count;
    st = pdb_append_part_slot(pdb, part);
    if (st != NBASE_OK) {
      free(part);
      free(buf);
      return st;
    }
    sum_vecs += count;
  }

  if (sum_vecs != hdr.total_vectors || pdb_find_part(pdb, pdb->active_partition_id) == NULL) {
    free(buf);
    return NBASE_ERR_CORRUPT;
  }

  for (uint64_t i = 0ULL; i < hdr.total_vectors; i++) {
    uint64_t vid = 0ULL;
    uint32_t pid = 0U;
    memcpy(&vid, buf + pos, sizeof(uint64_t));
    pos += sizeof(uint64_t);
    memcpy(&pid, buf + pos, sizeof(uint32_t));
    pos += sizeof(uint32_t);

    if (pdb_find_part(pdb, pid) == NULL) {
      free(buf);
      return NBASE_ERR_CORRUPT;
    }
    st = pdb_id_map_insert(&pdb->id_map, vid, pid);
    if (st == NBASE_ERR_ALREADY_EXISTS) {
      free(buf);
      return NBASE_ERR_CORRUPT;
    }
    if (st != NBASE_OK) {
      free(buf);
      return st;
    }
  }

  free(buf);
  return NBASE_OK;
}

nbase_status_t nbase_pdb_open(const char *dir_path, const nbase_pdb_config_t *cfg, nbase_pdb_t **out_pdb) {
  if (!dir_path || dir_path[0] == '\0' || !out_pdb) {
    return NBASE_ERR_INVALID_ARG;
  }

  nbase_status_t st = nbase_ensure_dir(dir_path);
  if (st != NBASE_OK) {
    return st;
  }

  nbase_pdb_t *pdb = calloc(1, sizeof(*pdb));
  if (!pdb) {
    return NBASE_ERR_NOMEM;
  }
  size_t dlen = strlen(dir_path);
  pdb->dir_path = malloc(dlen + 1U);
  if (!pdb->dir_path) {
    free(pdb);
    return NBASE_ERR_NOMEM;
  }
  memcpy(pdb->dir_path, dir_path, dlen + 1U);

  if (pthread_mutex_init(&pdb->mu, NULL) != 0) {
    free(pdb->dir_path);
    free(pdb);
    return NBASE_ERR_NOMEM;
  }

  if (!pdb_id_map_init(&pdb->id_map, 64U)) {
    pthread_mutex_destroy(&pdb->mu);
    free(pdb->dir_path);
    free(pdb);
    return NBASE_ERR_NOMEM;
  }

  char *mpath = pdb_manifest_path(pdb);
  if (!mpath) {
    pdb_destroy_unlocked(pdb);
    return NBASE_ERR_NOMEM;
  }

  st = pdb_load_manifest(pdb, mpath);
  free(mpath);

  if (st == NBASE_OK) {
    if (cfg != NULL && cfg->max_active_partitions > 0U) {
      pdb->cfg.max_active_partitions = cfg->max_active_partitions;
    }
    pdb_part_t *active = pdb_find_part(pdb, pdb->active_partition_id);
    if (active != NULL) {
      st = pdb_load_partition(pdb, active);
      if (st != NBASE_OK) {
        pdb_destroy_unlocked(pdb);
        return st;
      }
    }
    *out_pdb = pdb;
    return NBASE_OK;
  }

  if (st != NBASE_ERR_NOT_FOUND) {
    pdb_destroy_unlocked(pdb);
    return st;
  }

  if (!cfg || cfg->dim == 0U || cfg->dim > NBASE_MAX_DIM || !nbase_metric_is_valid(cfg->metric) || cfg->partition_capacity == 0U || cfg->max_active_partitions == 0U) {
    pdb_destroy_unlocked(pdb);
    return NBASE_ERR_INVALID_ARG;
  }
  if (cfg->enable_hnsw && (cfg->hnsw_m < 2U || cfg->hnsw_ef_construction == 0U || cfg->hnsw_ef_search == 0U)) {
    pdb_destroy_unlocked(pdb);
    return NBASE_ERR_INVALID_ARG;
  }

  pdb->cfg = *cfg;
  if (pdb->cfg.seed == 0ULL) {
    pdb->cfg.seed = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)pdb;
    if (pdb->cfg.seed == 0ULL) {
      pdb->cfg.seed = 1ULL;
    }
  }

  st = pdb_create_partition(pdb, NULL);
  if (st == NBASE_OK) {
    st = pdb_save_all_unlocked(pdb);
  }
  if (st != NBASE_OK) {
    pdb_destroy_unlocked(pdb);
    return st;
  }

  *out_pdb = pdb;
  return NBASE_OK;
}

nbase_status_t nbase_pdb_save(nbase_pdb_t *pdb) {
  if (!pdb) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);
  nbase_status_t st = pdb_save_all_unlocked(pdb);
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_close(nbase_pdb_t *pdb) {
  if (!pdb) {
    return NBASE_OK;
  }
  pthread_mutex_lock(&pdb->mu);
  nbase_status_t st = pdb_save_all_unlocked(pdb);
  pthread_mutex_unlock(&pdb->mu);
  pdb_destroy_unlocked(pdb);
  return st;
}

static nbase_status_t pdb_add_unlocked(nbase_pdb_t *pdb, uint64_t id, const float *vec, uint32_t dim) {
  if (dim != pdb->cfg.dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (!nbase_vec_is_finite(vec, dim)) {
    return NBASE_ERR_INVALID_VECTOR;
  }
  if (pdb_id_map_find(&pdb->id_map, id, NULL)) {
    return NBASE_ERR_ALREADY_EXISTS;
  }

  pdb_part_t *active = pdb_find_part(pdb, pdb->active_partition_id);
  nbase_status_t st = NBASE_OK;
  if (!active || active->active_vectors >= pdb->cfg.partition_capacity) {
    st = pdb_create_partition(pdb, &active);
  } else {
    st = pdb_load_partition(pdb, active);
  }
  if (st != NBASE_OK) {
    return st;
  }

  st = nbase_add(active->db, id, vec, dim);
  if (st != NBASE_OK) {
    return st;
  }

  st = pdb_id_map_insert(&pdb->id_map, id, active->part_id);
  if (st != NBASE_OK) {
    (void)nbase_delete(active->db, id);
    return st;
  }

  active->active_vectors++;
  pdb->total_vectors++;
  active->dirty = true;
  pdb->manifest_dirty = true;
  return NBASE_OK;
}

nbase_status_t nbase_pdb_add(nbase_pdb_t *pdb, uint64_t id, const float *vec, uint32_t dim) {
  if (!pdb || !vec) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);
  nbase_status_t st = pdb_add_unlocked(pdb, id, vec, dim);
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

static nbase_status_t pdb_add_batch_unlocked(
    nbase_pdb_t *pdb,
    const uint64_t *ids,
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    uint32_t num_threads) {
  nbase_status_t st = NBASE_OK;
  uint32_t off = 0U;
  while (off < count) {
    pdb_part_t *active = pdb_find_part(pdb, pdb->active_partition_id);
    if (!active || active->active_vectors >= pdb->cfg.partition_capacity) {
      st = pdb_create_partition(pdb, &active);
    } else {
      st = pdb_load_partition(pdb, active);
    }
    if (st != NBASE_OK) {
      break;
    }

    uint64_t avail64 = (uint64_t)pdb->cfg.partition_capacity - active->active_vectors;
    uint32_t rem = count - off;
    uint32_t chunk = ((uint64_t)rem < avail64) ? rem : (uint32_t)avail64;

    st = nbase_add_batch(
        active->db,
        ids + off,
        vectors + (size_t)off * (size_t)dim,
        chunk,
        dim,
        num_threads);
    if (st != NBASE_OK) {
      break;
    }

    uint32_t inserted = 0U;
    for (uint32_t j = 0U; j < chunk; j++) {
      st = pdb_id_map_insert(&pdb->id_map, ids[off + j], active->part_id);
      if (st != NBASE_OK) {
        break;
      }
      inserted++;
    }
    if (st != NBASE_OK) {
      for (uint32_t j = 0U; j < inserted; j++) {
        (void)pdb_id_map_delete(&pdb->id_map, ids[off + j]);
      }
      for (uint32_t j = 0U; j < chunk; j++) {
        (void)nbase_delete(active->db, ids[off + j]);
      }
      break;
    }

    active->active_vectors += chunk;
    pdb->total_vectors += chunk;
    active->dirty = true;
    pdb->manifest_dirty = true;
    off += chunk;
  }
  return st;
}

nbase_status_t nbase_pdb_add_batch(
    nbase_pdb_t *pdb,
    const uint64_t *ids,
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    uint32_t num_threads) {
  if (!pdb || (count > 0U && (!ids || !vectors))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  pthread_mutex_lock(&pdb->mu);
  if (dim != pdb->cfg.dim) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_DIM_MISMATCH;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)dim, dim)) {
      pthread_mutex_unlock(&pdb->mu);
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  uint64_t *sorted_ids = malloc((size_t)count * sizeof(uint64_t));
  if (!sorted_ids) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_NOMEM;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (pdb_id_map_find(&pdb->id_map, ids[i], NULL)) {
      free(sorted_ids);
      pthread_mutex_unlock(&pdb->mu);
      return NBASE_ERR_ALREADY_EXISTS;
    }
    sorted_ids[i] = ids[i];
  }
  qsort(sorted_ids, count, sizeof(uint64_t), cmp_pdb_u64_asc);
  for (uint32_t i = 1U; i < count; i++) {
    if (sorted_ids[i] == sorted_ids[i - 1U]) {
      free(sorted_ids);
      pthread_mutex_unlock(&pdb->mu);
      return NBASE_ERR_ALREADY_EXISTS;
    }
  }
  free(sorted_ids);

  nbase_status_t st = pdb_add_batch_unlocked(pdb, ids, vectors, count, dim, num_threads);
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_upsert(nbase_pdb_t *pdb, uint64_t id, const float *vec, uint32_t dim) {
  if (!pdb || !vec) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);

  if (dim != pdb->cfg.dim) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (!nbase_vec_is_finite(vec, dim)) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_INVALID_VECTOR;
  }

  nbase_status_t st;
  uint32_t part_id = 0U;
  if (!pdb_id_map_find(&pdb->id_map, id, &part_id)) {
    st = pdb_add_unlocked(pdb, id, vec, dim);
  } else {
    pdb_part_t *part = pdb_find_part(pdb, part_id);
    if (!part) {
      st = NBASE_ERR_CORRUPT;
    } else {
      st = pdb_load_partition(pdb, part);
      if (st == NBASE_OK) {
        st = nbase_upsert(part->db, id, vec, dim);
        if (st == NBASE_OK) {
          part->dirty = true;
        }
      }
    }
  }

  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_upsert_batch(
    nbase_pdb_t *pdb,
    const uint64_t *ids,
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    uint32_t num_threads) {
  if (!pdb || (count > 0U && (!ids || !vectors))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  pthread_mutex_lock(&pdb->mu);
  if (dim != pdb->cfg.dim) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_DIM_MISMATCH;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)dim, dim)) {
      pthread_mutex_unlock(&pdb->mu);
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  uint64_t *sorted_ids = malloc((size_t)count * sizeof(uint64_t));
  pdb_part_ref_t *existing_refs = malloc((size_t)count * sizeof(pdb_part_ref_t));
  uint64_t *new_ids = malloc((size_t)count * sizeof(uint64_t));
  float *new_vecs = malloc((size_t)count * (size_t)dim * sizeof(float));
  if (!sorted_ids || !existing_refs || !new_ids || !new_vecs) {
    free(sorted_ids);
    free(existing_refs);
    free(new_ids);
    free(new_vecs);
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_NOMEM;
  }

  memcpy(sorted_ids, ids, (size_t)count * sizeof(uint64_t));
  qsort(sorted_ids, count, sizeof(uint64_t), cmp_pdb_u64_asc);
  for (uint32_t i = 1U; i < count; i++) {
    if (sorted_ids[i] == sorted_ids[i - 1U]) {
      free(sorted_ids);
      free(existing_refs);
      free(new_ids);
      free(new_vecs);
      pthread_mutex_unlock(&pdb->mu);
      return NBASE_ERR_INVALID_ARG;
    }
  }
  free(sorted_ids);

  uint32_t existing_cnt = 0U;
  uint32_t new_cnt = 0U;
  for (uint32_t i = 0U; i < count; i++) {
    uint32_t pid = 0U;
    if (pdb_id_map_find(&pdb->id_map, ids[i], &pid)) {
      existing_refs[existing_cnt].part_id = pid;
      existing_refs[existing_cnt].orig_idx = i;
      existing_cnt++;
    } else {
      new_ids[new_cnt] = ids[i];
      memcpy(new_vecs + (size_t)new_cnt * (size_t)dim,
             vectors + (size_t)i * (size_t)dim,
             (size_t)dim * sizeof(float));
      new_cnt++;
    }
  }

  nbase_status_t st = NBASE_OK;
  if (existing_cnt > 0U) {
    qsort(existing_refs, existing_cnt, sizeof(pdb_part_ref_t), cmp_pdb_part_ref_asc);
    uint64_t *batch_ids = malloc((size_t)existing_cnt * sizeof(uint64_t));
    float *batch_vecs = malloc((size_t)existing_cnt * (size_t)dim * sizeof(float));
    if (!batch_ids || !batch_vecs) {
      st = NBASE_ERR_NOMEM;
    } else {
      uint32_t start = 0U;
      while (start < existing_cnt && st == NBASE_OK) {
        uint32_t pid = existing_refs[start].part_id;
        uint32_t end = start;
        while (end < existing_cnt && existing_refs[end].part_id == pid) {
          uint32_t oi = existing_refs[end].orig_idx;
          uint32_t bi = end - start;
          batch_ids[bi] = ids[oi];
          memcpy(batch_vecs + (size_t)bi * (size_t)dim,
                 vectors + (size_t)oi * (size_t)dim,
                 (size_t)dim * sizeof(float));
          end++;
        }
        pdb_part_t *part = pdb_find_part(pdb, pid);
        if (!part) {
          st = NBASE_ERR_CORRUPT;
          break;
        }
        st = pdb_load_partition(pdb, part);
        if (st != NBASE_OK) {
          break;
        }
        st = nbase_upsert_batch(part->db, batch_ids, batch_vecs, end - start, dim, num_threads);
        if (st == NBASE_OK) {
          part->dirty = true;
        }
        start = end;
      }
    }
    free(batch_ids);
    free(batch_vecs);
  }

  if (st == NBASE_OK && new_cnt > 0U) {
    st = pdb_add_batch_unlocked(pdb, new_ids, new_vecs, new_cnt, dim, num_threads);
  }

  free(existing_refs);
  free(new_ids);
  free(new_vecs);
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_delete(nbase_pdb_t *pdb, uint64_t id) {
  if (!pdb) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);

  uint32_t part_id = 0U;
  if (!pdb_id_map_find(&pdb->id_map, id, &part_id)) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_NOT_FOUND;
  }

  pdb_part_t *part = pdb_find_part(pdb, part_id);
  nbase_status_t st;
  if (!part) {
    st = NBASE_ERR_CORRUPT;
  } else {
    st = pdb_load_partition(pdb, part);
    if (st == NBASE_OK) {
      st = nbase_delete(part->db, id);
      if (st == NBASE_OK) {
        (void)pdb_id_map_delete(&pdb->id_map, id);
        part->active_vectors--;
        pdb->total_vectors--;
        part->dirty = true;
        pdb->manifest_dirty = true;
      }
    }
  }

  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_delete_batch(
    nbase_pdb_t *pdb,
    const uint64_t *ids,
    uint32_t count,
    uint32_t *out_deleted) {
  if (!pdb || (count > 0U && !ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (out_deleted) {
    *out_deleted = 0U;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  pthread_mutex_lock(&pdb->mu);
  nbase_status_t st = NBASE_OK;
  uint32_t deleted = 0U;

  for (uint32_t i = 0U; i < count; i++) {
    uint32_t part_id = 0U;
    if (!pdb_id_map_find(&pdb->id_map, ids[i], &part_id)) {
      continue;
    }
    pdb_part_t *part = pdb_find_part(pdb, part_id);
    if (!part) {
      st = NBASE_ERR_CORRUPT;
      break;
    }
    st = pdb_load_partition(pdb, part);
    if (st != NBASE_OK) {
      break;
    }
    st = nbase_delete(part->db, ids[i]);
    if (st != NBASE_OK) {
      break;
    }
    (void)pdb_id_map_delete(&pdb->id_map, ids[i]);
    part->active_vectors--;
    pdb->total_vectors--;
    part->dirty = true;
    pdb->manifest_dirty = true;
    deleted++;
  }

  pthread_mutex_unlock(&pdb->mu);
  if (out_deleted) {
    *out_deleted = deleted;
  }
  return st;
}

nbase_status_t nbase_pdb_delete_by_filter(
    nbase_pdb_t *pdb,
    const nbase_meta_filter_t *filter,
    uint32_t *out_deleted) {
  if (!pdb || !filter || filter->count == 0U || !filter->conds) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (out_deleted) {
    *out_deleted = 0U;
  }

  pthread_mutex_lock(&pdb->mu);
  nbase_status_t st = NBASE_OK;
  uint32_t deleted = 0U;
  uint64_t max_part_vecs = 0ULL;
  for (uint32_t i = 0U; i < pdb->part_count; i++) {
    if (pdb->parts[i]->active_vectors > max_part_vecs) {
      max_part_vecs = pdb->parts[i]->active_vectors;
    }
  }
  if (max_part_vecs == 0ULL) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_OK;
  }

  uint32_t buf_limit = (max_part_vecs > (uint64_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)max_part_vecs;
  uint64_t *id_buf = malloc((size_t)buf_limit * sizeof(uint64_t));
  if (!id_buf) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_NOMEM;
  }

  for (uint32_t i = 0U; i < pdb->part_count && st == NBASE_OK; i++) {
    pdb_part_t *p = pdb->parts[i];
    if (p->active_vectors == 0ULL) {
      continue;
    }
    st = pdb_load_partition(pdb, p);
    if (st != NBASE_OK) {
      break;
    }
    uint32_t part_lim = (p->active_vectors > (uint64_t)buf_limit)
                            ? buf_limit
                            : (uint32_t)p->active_vectors;
    uint32_t got = 0U;
    st = nbase_query_meta(p->db, filter, part_lim, id_buf, &got);
    if (st != NBASE_OK) {
      break;
    }
    for (uint32_t j = 0U; j < got; j++) {
      st = nbase_delete(p->db, id_buf[j]);
      if (st != NBASE_OK) {
        break;
      }
      (void)pdb_id_map_delete(&pdb->id_map, id_buf[j]);
      p->active_vectors--;
      pdb->total_vectors--;
      p->dirty = true;
      pdb->manifest_dirty = true;
      deleted++;
    }
  }

  free(id_buf);
  pthread_mutex_unlock(&pdb->mu);
  if (out_deleted) {
    *out_deleted = deleted;
  }
  return st;
}

nbase_status_t nbase_pdb_get(nbase_pdb_t *pdb, uint64_t id, float *out_vec, uint32_t dim) {
  if (!pdb || !out_vec) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);

  nbase_status_t st;
  uint32_t part_id = 0U;
  if (dim != pdb->cfg.dim) {
    st = NBASE_ERR_DIM_MISMATCH;
  } else if (!pdb_id_map_find(&pdb->id_map, id, &part_id)) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    pdb_part_t *part = pdb_find_part(pdb, part_id);
    if (!part) {
      st = NBASE_ERR_CORRUPT;
    } else {
      st = pdb_load_partition(pdb, part);
      if (st == NBASE_OK) {
        st = nbase_get(part->db, id, out_vec, dim);
      }
    }
  }

  pthread_mutex_unlock(&pdb->mu);
  return st;
}

static nbase_status_t pdb_get_batch_unlocked(
    nbase_pdb_t *pdb,
    const uint64_t *ids,
    uint32_t count,
    uint32_t dim,
    float *out_vectors,
    bool *out_found) {
  if (count == 0U) {
    return NBASE_OK;
  }
  if (dim != pdb->cfg.dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }

  pdb_part_ref_t *refs = malloc((size_t)count * sizeof(pdb_part_ref_t));
  if (!refs) {
    return NBASE_ERR_NOMEM;
  }

  size_t vec_bytes = (size_t)dim * sizeof(float);
  uint32_t ref_cnt = 0U;
  for (uint32_t i = 0U; i < count; i++) {
    uint32_t pid = 0U;
    if (pdb_id_map_find(&pdb->id_map, ids[i], &pid)) {
      refs[ref_cnt].part_id = pid;
      refs[ref_cnt].orig_idx = i;
      ref_cnt++;
    } else {
      memset(out_vectors + (size_t)i * (size_t)dim, 0, vec_bytes);
      out_found[i] = false;
    }
  }

  nbase_status_t st = NBASE_OK;
  if (ref_cnt > 0U) {
    qsort(refs, ref_cnt, sizeof(pdb_part_ref_t), cmp_pdb_part_ref_asc);
    uint64_t *batch_ids = malloc((size_t)ref_cnt * sizeof(uint64_t));
    float *batch_vecs = malloc((size_t)ref_cnt * vec_bytes);
    bool *batch_found = malloc((size_t)ref_cnt * sizeof(bool));
    if (!batch_ids || !batch_vecs || !batch_found) {
      st = NBASE_ERR_NOMEM;
    } else {
      uint32_t start = 0U;
      while (start < ref_cnt && st == NBASE_OK) {
        uint32_t pid = refs[start].part_id;
        uint32_t end = start;
        while (end < ref_cnt && refs[end].part_id == pid) {
          batch_ids[end - start] = ids[refs[end].orig_idx];
          end++;
        }
        uint32_t sub_cnt = end - start;
        pdb_part_t *part = pdb_find_part(pdb, pid);
        if (!part) {
          st = NBASE_ERR_CORRUPT;
          break;
        }
        st = pdb_load_partition(pdb, part);
        if (st != NBASE_OK) {
          break;
        }
        st = nbase_get_batch(part->db, batch_ids, sub_cnt, dim, batch_vecs, batch_found);
        if (st != NBASE_OK) {
          break;
        }
        for (uint32_t j = 0U; j < sub_cnt; j++) {
          uint32_t oi = refs[start + j].orig_idx;
          out_found[oi] = batch_found[j];
          memcpy(out_vectors + (size_t)oi * (size_t)dim,
                 batch_vecs + (size_t)j * (size_t)dim,
                 vec_bytes);
        }
        start = end;
      }
    }
    free(batch_ids);
    free(batch_vecs);
    free(batch_found);
  }

  free(refs);
  return st;
}

nbase_status_t nbase_pdb_get_batch(
    nbase_pdb_t *pdb,
    const uint64_t *ids,
    uint32_t count,
    uint32_t dim,
    float *out_vectors,
    bool *out_found) {
  if (!pdb || (count > 0U && (!ids || !out_vectors || !out_found))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }
  pthread_mutex_lock(&pdb->mu);
  nbase_status_t st = pdb_get_batch_unlocked(pdb, ids, count, dim, out_vectors, out_found);
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

bool nbase_pdb_contains(nbase_pdb_t *pdb, uint64_t id) {
  if (!pdb) {
    return false;
  }
  pthread_mutex_lock(&pdb->mu);
  bool found = pdb_id_map_find(&pdb->id_map, id, NULL);
  pthread_mutex_unlock(&pdb->mu);
  return found;
}

nbase_status_t nbase_pdb_list_ids(
    nbase_pdb_t *pdb,
    uint64_t offset,
    uint32_t limit,
    uint64_t *out_ids,
    uint32_t *out_count,
    uint64_t *out_total) {
  if (!pdb || !out_count || (limit > 0U && !out_ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  if (out_total) {
    *out_total = 0ULL;
  }

  pthread_mutex_lock(&pdb->mu);
  uint64_t total = pdb->total_vectors;
  if (out_total) {
    *out_total = total;
  }
  if (limit == 0U || total == 0ULL || offset >= total || pdb->id_map.count == 0U) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_OK;
  }

  uint64_t *all = malloc(pdb->id_map.count * sizeof(uint64_t));
  if (!all) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_NOMEM;
  }

  size_t pos = 0U;
  for (size_t i = 0U; i < pdb->id_map.cap && pos < pdb->id_map.count; i++) {
    if (pdb->id_map.states[i] == PDB_MAP_OCCUPIED) {
      all[pos++] = pdb->id_map.keys[i];
    }
  }
  pthread_mutex_unlock(&pdb->mu);

  if (pos > 1U) {
    qsort(all, pos, sizeof(uint64_t), cmp_pdb_u64_asc);
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

nbase_status_t nbase_pdb_set_meta(nbase_pdb_t *pdb, uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
  if (!pdb) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);

  nbase_status_t st;
  uint32_t part_id = 0U;
  if (!pdb_id_map_find(&pdb->id_map, id, &part_id)) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    pdb_part_t *part = pdb_find_part(pdb, part_id);
    if (!part) {
      st = NBASE_ERR_CORRUPT;
    } else {
      st = pdb_load_partition(pdb, part);
      if (st == NBASE_OK) {
        st = nbase_set_meta(part->db, id, fields, count);
        if (st == NBASE_OK) {
          part->dirty = true;
        }
      }
    }
  }

  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_update_meta(nbase_pdb_t *pdb, uint64_t id, const nbase_meta_field_t *fields, uint32_t count) {
  if (!pdb) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);

  nbase_status_t st;
  uint32_t part_id = 0U;
  if (!pdb_id_map_find(&pdb->id_map, id, &part_id)) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    pdb_part_t *part = pdb_find_part(pdb, part_id);
    if (!part) {
      st = NBASE_ERR_CORRUPT;
    } else {
      st = pdb_load_partition(pdb, part);
      if (st == NBASE_OK) {
        st = nbase_update_meta(part->db, id, fields, count);
        if (st == NBASE_OK) {
          part->dirty = true;
        }
      }
    }
  }

  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_get_meta(nbase_pdb_t *pdb, uint64_t id, const char *key, nbase_meta_value_t *out_val, char *str_buf, size_t str_buf_len) {
  if (!pdb || !key || key[0] == '\0' || !out_val) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);

  nbase_status_t st;
  uint32_t part_id = 0U;
  if (!pdb_id_map_find(&pdb->id_map, id, &part_id)) {
    st = NBASE_ERR_NOT_FOUND;
  } else {
    pdb_part_t *part = pdb_find_part(pdb, part_id);
    if (!part) {
      st = NBASE_ERR_CORRUPT;
    } else {
      st = pdb_load_partition(pdb, part);
      if (st == NBASE_OK) {
        st = nbase_get_meta(part->db, id, key, out_val, str_buf, str_buf_len);
      }
    }
  }

  pthread_mutex_unlock(&pdb->mu);
  return st;
}

static nbase_status_t pdb_find_and_load_part(
    nbase_pdb_t *pdb,
    uint64_t id,
    pdb_part_t **out_part) {
  uint32_t part_id = 0U;
  if (!pdb_id_map_find(&pdb->id_map, id, &part_id)) {
    return NBASE_ERR_NOT_FOUND;
  }
  pdb_part_t *part = pdb_find_part(pdb, part_id);
  if (!part) {
    return NBASE_ERR_CORRUPT;
  }
  nbase_status_t st = pdb_load_partition(pdb, part);
  if (st == NBASE_OK) {
    *out_part = part;
  }
  return st;
}

nbase_status_t nbase_pdb_delete_meta(nbase_pdb_t *pdb, uint64_t id) {
  if (!pdb) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);
  pdb_part_t *part = NULL;
  nbase_status_t st = pdb_find_and_load_part(pdb, id, &part);
  if (st == NBASE_OK) {
    st = nbase_delete_meta(part->db, id);
    if (st == NBASE_OK) {
      part->dirty = true;
    }
  }
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_foreach_meta(
    nbase_pdb_t *pdb,
    uint64_t id,
    nbase_meta_iter_fn cb,
    void *user_ctx) {
  if (!pdb || !cb) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);
  pdb_part_t *part = NULL;
  nbase_status_t st = pdb_find_and_load_part(pdb, id, &part);
  if (st == NBASE_OK) {
    st = nbase_foreach_meta(part->db, id, cb, user_ctx);
  }
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_query_meta(
    nbase_pdb_t *pdb,
    const nbase_meta_filter_t *filter,
    uint32_t limit,
    uint64_t *out_ids,
    uint32_t *out_count) {
  if (!pdb || !out_count || (limit > 0U && !out_ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  if (limit == 0U) {
    return NBASE_OK;
  }

  pthread_mutex_lock(&pdb->mu);
  if (pdb->total_vectors == 0ULL || pdb->part_count == 0U) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_OK;
  }

  pdb_part_t **parts_copy = malloc((size_t)pdb->part_count * sizeof(pdb_part_t *));
  uint64_t *part_buf = malloc((size_t)limit * sizeof(uint64_t));
  if (!parts_copy || !part_buf) {
    free(parts_copy);
    free(part_buf);
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_NOMEM;
  }
  uint32_t pcnt = pdb->part_count;
  memcpy(parts_copy, pdb->parts, (size_t)pcnt * sizeof(pdb_part_t *));

  uint64_t *merged = NULL;
  size_t merged_len = 0U;
  size_t merged_cap = 0U;
  nbase_status_t st = NBASE_OK;

  for (uint32_t i = 0U; i < pcnt; i++) {
    pdb_part_t *p = parts_copy[i];
    if (p->active_vectors == 0ULL) {
      continue;
    }
    st = pdb_load_partition(pdb, p);
    if (st != NBASE_OK) {
      break;
    }
    uint32_t got = 0U;
    st = nbase_query_meta(p->db, filter, limit, part_buf, &got);
    if (st != NBASE_OK) {
      break;
    }
    if (got > 0U) {
      if (merged_len + got > merged_cap) {
        size_t new_cap = (merged_cap == 0U) ? (size_t)limit * 2U : merged_cap * 2U;
        while (new_cap < merged_len + got) {
          new_cap *= 2U;
        }
        uint64_t *grown = realloc(merged, new_cap * sizeof(uint64_t));
        if (!grown) {
          st = NBASE_ERR_NOMEM;
          break;
        }
        merged = grown;
        merged_cap = new_cap;
      }
      memcpy(merged + merged_len, part_buf, (size_t)got * sizeof(uint64_t));
      merged_len += got;
    }
  }

  free(parts_copy);
  free(part_buf);
  pthread_mutex_unlock(&pdb->mu);

  if (st != NBASE_OK) {
    free(merged);
    return st;
  }
  if (merged_len > 1U) {
    qsort(merged, merged_len, sizeof(uint64_t), cmp_pdb_u64_asc);
  }
  uint32_t ret = (merged_len < (size_t)limit) ? (uint32_t)merged_len : limit;
  if (ret > 0U) {
    memcpy(out_ids, merged, (size_t)ret * sizeof(uint64_t));
  }
  free(merged);
  *out_count = ret;
  return NBASE_OK;
}

nbase_status_t nbase_pdb_set_text(nbase_pdb_t *pdb, uint64_t id, const char *text) {
  if (!pdb || !text) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);
  pdb_part_t *part = NULL;
  nbase_status_t st = pdb_find_and_load_part(pdb, id, &part);
  if (st == NBASE_OK) {
    st = nbase_set_text(part->db, id, text);
    if (st == NBASE_OK) {
      part->dirty = true;
    }
  }
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_get_text(
    nbase_pdb_t *pdb,
    uint64_t id,
    char *out_buf,
    size_t buf_len,
    size_t *out_len) {
  if (!pdb || (!out_buf && !out_len)) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);
  pdb_part_t *part = NULL;
  nbase_status_t st = pdb_find_and_load_part(pdb, id, &part);
  if (st == NBASE_OK) {
    st = nbase_get_text(part->db, id, out_buf, buf_len, out_len);
  }
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_delete_text(nbase_pdb_t *pdb, uint64_t id) {
  if (!pdb) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);
  pdb_part_t *part = NULL;
  nbase_status_t st = pdb_find_and_load_part(pdb, id, &part);
  if (st == NBASE_OK) {
    st = nbase_delete_text(part->db, id);
    if (st == NBASE_OK) {
      part->dirty = true;
    }
  }
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

static nbase_status_t pdb_collect_targets(
    const nbase_pdb_t *pdb,
    const nbase_pdb_search_opts_t *eff,
    pdb_part_t ***out_targets,
    uint32_t *out_count) {
  uint32_t max_targets = (eff->partition_id_count > 0U) ? eff->partition_id_count : pdb->part_count;
  pdb_part_t **targets = calloc((max_targets > 0U) ? max_targets : 1U, sizeof(pdb_part_t *));
  if (!targets) {
    return NBASE_ERR_NOMEM;
  }
  uint32_t target_count = 0U;
  if (eff->partition_id_count > 0U) {
    for (uint32_t i = 0U; i < eff->partition_id_count; i++) {
      pdb_part_t *p = pdb_find_part(pdb, eff->partition_ids[i]);
      if (!p) {
        continue;
      }
      bool seen = false;
      for (uint32_t j = 0U; j < target_count; j++) {
        if (targets[j] == p) {
          seen = true;
          break;
        }
      }
      if (!seen) {
        targets[target_count++] = p;
      }
    }
  } else {
    for (uint32_t i = 0U; i < pdb->part_count; i++) {
      pdb_part_t *p = pdb->parts[i];
      if (eff->search_all_partitions || p->db != NULL) {
        targets[target_count++] = p;
      }
    }
  }
  *out_targets = targets;
  *out_count = target_count;
  return NBASE_OK;
}

static nbase_status_t pdb_search_common_unlocked(
    nbase_pdb_t *pdb,
    const float *query,
    uint32_t dim,
    uint32_t k,
    bool use_radius,
    float max_distance,
    const nbase_pdb_search_opts_t *eff,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  *out_count = 0U;
  if (dim != pdb->cfg.dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (!nbase_vec_is_finite(query, dim)) {
    return NBASE_ERR_INVALID_VECTOR;
  }
  if (k == 0U) {
    return NBASE_OK;
  }

  pdb_part_t **targets = NULL;
  uint32_t target_count = 0U;
  nbase_status_t st = pdb_collect_targets(pdb, eff, &targets, &target_count);
  if (st != NBASE_OK) {
    return st;
  }
  nbase_result_t *part_buf = malloc((size_t)k * sizeof(nbase_result_t));
  if (!part_buf) {
    free(targets);
    return NBASE_ERR_NOMEM;
  }

  nbase_result_t *merged = NULL;
  size_t merged_len = 0U;
  size_t merged_cap = 0U;

  for (uint32_t i = 0U; i < target_count; i++) {
    pdb_part_t *p = targets[i];
    st = pdb_load_partition(pdb, p);
    if (st != NBASE_OK) {
      break;
    }

    uint32_t part_n = 0U;
    st = use_radius
             ? nbase_search_radius(
                   p->db, query, dim, max_distance, k, &eff->search, part_buf, &part_n)
             : nbase_search(p->db, query, dim, k, &eff->search, part_buf, &part_n);
    if (st != NBASE_OK) {
      break;
    }

    if (part_n > 0U) {
      if (merged_len + part_n > merged_cap) {
        size_t new_cap = (merged_cap == 0U) ? (size_t)k * 2U : merged_cap * 2U;
        while (new_cap < merged_len + part_n) {
          new_cap *= 2U;
        }
        nbase_result_t *grown = realloc(merged, new_cap * sizeof(nbase_result_t));
        if (!grown) {
          st = NBASE_ERR_NOMEM;
          break;
        }
        merged = grown;
        merged_cap = new_cap;
      }
      memcpy(merged + merged_len, part_buf, (size_t)part_n * sizeof(nbase_result_t));
      merged_len += part_n;
    }
  }

  free(targets);
  free(part_buf);
  if (st != NBASE_OK) {
    free(merged);
    return st;
  }
  if (merged_len > 0U) {
    qsort(merged, merged_len, sizeof(nbase_result_t), nbase_cmp_result_asc);
  }
  uint32_t ret = (merged_len < (size_t)k) ? (uint32_t)merged_len : k;
  if (ret > 0U) {
    memcpy(out_results, merged, (size_t)ret * sizeof(nbase_result_t));
  }
  free(merged);
  *out_count = ret;
  return NBASE_OK;
}

static nbase_status_t pdb_search_unlocked(
    nbase_pdb_t *pdb,
    const float *query,
    uint32_t dim,
    uint32_t k,
    const nbase_pdb_search_opts_t *eff,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  return pdb_search_common_unlocked(pdb, query, dim, k, false, 0.0f, eff, out_results, out_count);
}

nbase_status_t nbase_pdb_search(
    nbase_pdb_t *pdb,
    const float *query,
    uint32_t dim,
    uint32_t k,
    const nbase_pdb_search_opts_t *opts,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!pdb || !query || !out_count || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;

  nbase_pdb_search_opts_t eff = opts ? *opts : nbase_pdb_search_opts_default();
  if (eff.partition_id_count > 0U && !eff.partition_ids) {
    return NBASE_ERR_INVALID_ARG;
  }

  pthread_mutex_lock(&pdb->mu);
  nbase_status_t st = pdb_search_unlocked(pdb, query, dim, k, &eff, out_results, out_count);
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_search_radius(
    nbase_pdb_t *pdb,
    const float *query,
    uint32_t dim,
    float max_distance,
    uint32_t max_results,
    const nbase_pdb_search_opts_t *opts,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!pdb || !query || !out_count || (max_results > 0U && !out_results) || !isfinite(max_distance)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;

  nbase_pdb_search_opts_t eff = opts ? *opts : nbase_pdb_search_opts_default();
  if (eff.partition_id_count > 0U && !eff.partition_ids) {
    return NBASE_ERR_INVALID_ARG;
  }

  pthread_mutex_lock(&pdb->mu);
  nbase_status_t st = pdb_search_common_unlocked(
      pdb, query, dim, max_results, true, max_distance, &eff, out_results, out_count);
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_search_batch(
    nbase_pdb_t *pdb,
    const float *queries,
    uint32_t query_count,
    uint32_t dim,
    uint32_t k,
    const nbase_pdb_search_opts_t *opts,
    nbase_result_t *out_results,
    uint32_t *out_counts) {
  if (!pdb || !out_counts || (query_count > 0U && !queries) ||
      (query_count > 0U && k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }

  nbase_pdb_search_opts_t eff = opts ? *opts : nbase_pdb_search_opts_default();
  if (eff.partition_id_count > 0U && !eff.partition_ids) {
    return NBASE_ERR_INVALID_ARG;
  }

  pthread_mutex_lock(&pdb->mu);
  if (dim != pdb->cfg.dim) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (query_count == 0U) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_OK;
  }

  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < query_count; i++) {
    const float *q = queries + (size_t)i * (size_t)dim;
    nbase_result_t *dst = (k > 0U) ? (out_results + (size_t)i * (size_t)k) : NULL;
    st = pdb_search_unlocked(pdb, q, dim, k, &eff, dst, &out_counts[i]);
    if (st != NBASE_OK) {
      break;
    }
  }

  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_search_text(
    nbase_pdb_t *pdb,
    const char *query,
    uint32_t k,
    const nbase_pdb_search_opts_t *opts,
    nbase_scored_id_t *out_results,
    uint32_t *out_count) {
  if (!pdb || !query || !out_count || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;

  nbase_pdb_search_opts_t eff;
  if (opts != NULL) {
    eff = *opts;
  } else {
    eff = nbase_pdb_search_opts_default();
    eff.search_all_partitions = true;
  }
  if (eff.partition_id_count > 0U && !eff.partition_ids) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (k == 0U) {
    return NBASE_OK;
  }

  pthread_mutex_lock(&pdb->mu);
  pdb_part_t **targets = NULL;
  uint32_t target_count = 0U;
  nbase_status_t st = pdb_collect_targets(pdb, &eff, &targets, &target_count);
  nbase_scored_id_t *part_buf = (st == NBASE_OK) ? malloc((size_t)k * sizeof(nbase_scored_id_t)) : NULL;
  if (st == NBASE_OK && !part_buf) {
    st = NBASE_ERR_NOMEM;
  }

  nbase_scored_id_t *merged = NULL;
  size_t merged_len = 0U;
  size_t merged_cap = 0U;
  for (uint32_t i = 0U; st == NBASE_OK && i < target_count; i++) {
    pdb_part_t *p = targets[i];
    st = pdb_load_partition(pdb, p);
    if (st != NBASE_OK) {
      break;
    }
    uint32_t part_n = 0U;
    st = nbase_search_text(p->db, query, k, eff.search.meta_filter, part_buf, &part_n);
    if (st != NBASE_OK) {
      break;
    }
    if (part_n > 0U) {
      if (merged_len + part_n > merged_cap) {
        size_t new_cap = (merged_cap == 0U) ? (size_t)k * 2U : merged_cap * 2U;
        while (new_cap < merged_len + part_n) {
          new_cap *= 2U;
        }
        nbase_scored_id_t *grown = realloc(merged, new_cap * sizeof(nbase_scored_id_t));
        if (!grown) {
          st = NBASE_ERR_NOMEM;
          break;
        }
        merged = grown;
        merged_cap = new_cap;
      }
      memcpy(merged + merged_len, part_buf, (size_t)part_n * sizeof(nbase_scored_id_t));
      merged_len += part_n;
    }
  }
  free(targets);
  free(part_buf);
  pthread_mutex_unlock(&pdb->mu);

  if (st != NBASE_OK) {
    free(merged);
    return st;
  }
  if (merged_len > 1U) {
    qsort(merged, merged_len, sizeof(nbase_scored_id_t), cmp_pdb_scored_desc);
  }
  uint32_t ret = (merged_len < (size_t)k) ? (uint32_t)merged_len : k;
  if (ret > 0U) {
    memcpy(out_results, merged, (size_t)ret * sizeof(nbase_scored_id_t));
  }
  free(merged);
  *out_count = ret;
  return NBASE_OK;
}

nbase_status_t nbase_pdb_search_hybrid(
    nbase_pdb_t *pdb,
    const float *query_vec,
    uint32_t dim,
    const char *query_text,
    uint32_t k,
    float alpha,
    float rrf_k,
    const nbase_pdb_search_opts_t *opts,
    nbase_scored_id_t *out_results,
    uint32_t *out_count) {
  bool has_vec = (query_vec != NULL);
  bool has_text = (query_text != NULL && query_text[0] != '\0');
  if (!pdb || !out_count || (k > 0U && !out_results) || (!has_vec && !has_text) ||
      isnan(alpha) || alpha < 0.0f || alpha > 1.0f || !isfinite(rrf_k) || rrf_k <= 0.0f) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;

  nbase_pdb_search_opts_t eff;
  if (opts != NULL) {
    eff = *opts;
  } else {
    eff = nbase_pdb_search_opts_default();
    eff.search_all_partitions = true;
  }
  if (eff.partition_id_count > 0U && !eff.partition_ids) {
    return NBASE_ERR_INVALID_ARG;
  }

  pthread_mutex_lock(&pdb->mu);
  if (has_vec) {
    if (dim != pdb->cfg.dim) {
      pthread_mutex_unlock(&pdb->mu);
      return NBASE_ERR_DIM_MISMATCH;
    }
    if (!nbase_vec_is_finite(query_vec, dim)) {
      pthread_mutex_unlock(&pdb->mu);
      return NBASE_ERR_INVALID_VECTOR;
    }
  }
  if (k == 0U) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_OK;
  }

  pdb_part_t **targets = NULL;
  uint32_t target_count = 0U;
  nbase_status_t st = pdb_collect_targets(pdb, &eff, &targets, &target_count);
  nbase_scored_id_t *part_buf = (st == NBASE_OK) ? malloc((size_t)k * sizeof(nbase_scored_id_t)) : NULL;
  if (st == NBASE_OK && !part_buf) {
    st = NBASE_ERR_NOMEM;
  }

  nbase_scored_id_t *merged = NULL;
  size_t merged_len = 0U;
  size_t merged_cap = 0U;
  for (uint32_t i = 0U; st == NBASE_OK && i < target_count; i++) {
    pdb_part_t *p = targets[i];
    st = pdb_load_partition(pdb, p);
    if (st != NBASE_OK) {
      break;
    }
    uint32_t part_n = 0U;
    st = nbase_search_hybrid(
        p->db, query_vec, dim, query_text, k, alpha, rrf_k, &eff.search, part_buf, &part_n);
    if (st != NBASE_OK) {
      break;
    }
    if (part_n > 0U) {
      if (merged_len + part_n > merged_cap) {
        size_t new_cap = (merged_cap == 0U) ? (size_t)k * 2U : merged_cap * 2U;
        while (new_cap < merged_len + part_n) {
          new_cap *= 2U;
        }
        nbase_scored_id_t *grown = realloc(merged, new_cap * sizeof(nbase_scored_id_t));
        if (!grown) {
          st = NBASE_ERR_NOMEM;
          break;
        }
        merged = grown;
        merged_cap = new_cap;
      }
      memcpy(merged + merged_len, part_buf, (size_t)part_n * sizeof(nbase_scored_id_t));
      merged_len += part_n;
    }
  }
  free(targets);
  free(part_buf);
  pthread_mutex_unlock(&pdb->mu);

  if (st != NBASE_OK) {
    free(merged);
    return st;
  }
  if (merged_len > 1U) {
    qsort(merged, merged_len, sizeof(nbase_scored_id_t), cmp_pdb_scored_desc);
  }
  uint32_t ret = (merged_len < (size_t)k) ? (uint32_t)merged_len : k;
  if (ret > 0U) {
    memcpy(out_results, merged, (size_t)ret * sizeof(nbase_scored_id_t));
  }
  free(merged);
  *out_count = ret;
  return NBASE_OK;
}

nbase_status_t nbase_pdb_find_similar(
    nbase_pdb_t *pdb,
    uint64_t id,
    uint32_t k,
    const nbase_pdb_search_opts_t *opts,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!pdb || !out_count || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;

  nbase_pdb_search_opts_t eff;
  if (opts != NULL) {
    eff = *opts;
  } else {
    eff = nbase_pdb_search_opts_default();
    eff.search_all_partitions = true;
  }
  if (eff.partition_id_count > 0U && !eff.partition_ids) {
    return NBASE_ERR_INVALID_ARG;
  }

  pthread_mutex_lock(&pdb->mu);
  pdb_part_t *src_part = NULL;
  nbase_status_t st = pdb_find_and_load_part(pdb, id, &src_part);
  if (st != NBASE_OK) {
    pthread_mutex_unlock(&pdb->mu);
    return st;
  }
  if (k == 0U || pdb->total_vectors <= 1ULL) {
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_OK;
  }

  uint32_t dim = pdb->cfg.dim;
  float *qvec = malloc((size_t)dim * sizeof(float));
  uint32_t fetch_k = (k < UINT32_MAX) ? (k + 1U) : k;
  if ((uint64_t)fetch_k > pdb->total_vectors) {
    fetch_k = (uint32_t)pdb->total_vectors;
  }
  nbase_result_t *tmp = malloc((size_t)fetch_k * sizeof(nbase_result_t));
  if (!qvec || !tmp) {
    free(qvec);
    free(tmp);
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_NOMEM;
  }

  st = nbase_get(src_part->db, id, qvec, dim);
  uint32_t got = 0U;
  if (st == NBASE_OK) {
    st = pdb_search_unlocked(pdb, qvec, dim, fetch_k, &eff, tmp, &got);
  }
  pthread_mutex_unlock(&pdb->mu);

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
  free(qvec);
  free(tmp);
  return st;
}

static int cmp_pdb_edge_canonical(const void *pa, const void *pb) {
  const nbase_edge_t *a = (const nbase_edge_t *)pa;
  const nbase_edge_t *b = (const nbase_edge_t *)pb;
  if (a->src_id < b->src_id) {
    return -1;
  }
  if (a->src_id > b->src_id) {
    return 1;
  }
  if (a->dst_id < b->dst_id) {
    return -1;
  }
  if (a->dst_id > b->dst_id) {
    return 1;
  }
  if (a->weight < b->weight) {
    return -1;
  }
  if (a->weight > b->weight) {
    return 1;
  }
  return 0;
}

nbase_status_t nbase_pdb_extract_relationships(
    nbase_pdb_t *pdb,
    uint32_t k_neighbors,
    float max_distance,
    const nbase_pdb_search_opts_t *opts,
    nbase_edge_t **out_edges,
    uint64_t *out_edge_count) {
  if (!pdb || !out_edges || !out_edge_count || k_neighbors == 0U || isnan(max_distance)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_edges = NULL;
  *out_edge_count = 0ULL;

  nbase_pdb_search_opts_t eff;
  if (opts != NULL) {
    eff = *opts;
  } else {
    eff = nbase_pdb_search_opts_default();
    eff.search_all_partitions = true;
  }
  if (eff.partition_id_count > 0U && !eff.partition_ids) {
    return NBASE_ERR_INVALID_ARG;
  }

  pthread_mutex_lock(&pdb->mu);
  pdb_part_t **targets = NULL;
  uint32_t target_count = 0U;
  nbase_status_t st = pdb_collect_targets(pdb, &eff, &targets, &target_count);
  if (st != NBASE_OK) {
    pthread_mutex_unlock(&pdb->mu);
    return st;
  }

  uint64_t total_est = 0ULL;
  for (uint32_t t = 0U; t < target_count; t++) {
    total_est += targets[t]->active_vectors;
  }
  if (total_est < 2ULL) {
    free(targets);
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_OK;
  }

  uint32_t dim = pdb->cfg.dim;
  if (total_est > (uint64_t)(SIZE_MAX / ((size_t)dim * sizeof(float)))) {
    free(targets);
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_NOMEM;
  }

  uint64_t *all_ids = malloc((size_t)total_est * sizeof(uint64_t));
  float *all_vecs = malloc((size_t)total_est * (size_t)dim * sizeof(float));
  if (!all_ids || !all_vecs) {
    free(all_ids);
    free(all_vecs);
    free(targets);
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_NOMEM;
  }

  size_t n_vecs = 0U;
  for (uint32_t t = 0U; t < target_count && st == NBASE_OK; t++) {
    pdb_part_t *p = targets[t];
    if (p->active_vectors == 0ULL) {
      continue;
    }
    st = pdb_load_partition(pdb, p);
    if (st != NBASE_OK) {
      break;
    }
    nbase_db_rdlock(p->db);
    const nbase_store_t *store = nbase_db_store(p->db);
    for (uint32_t s = 0U; s < store->slot_count && n_vecs < (size_t)total_est; s++) {
      if (store->slot_flags[s] == NBASE_SLOT_ACTIVE) {
        all_ids[n_vecs] = store->ids[s];
        memcpy(
            all_vecs + n_vecs * (size_t)dim,
            nbase_store_get_vector(store, s),
            (size_t)dim * sizeof(float));
        n_vecs++;
      }
    }
    nbase_db_unlock(p->db);
  }
  free(targets);

  if (st != NBASE_OK || n_vecs < 2U) {
    free(all_ids);
    free(all_vecs);
    pthread_mutex_unlock(&pdb->mu);
    return st;
  }

  uint32_t max_nbrs = (uint32_t)(n_vecs - 1U);
  uint32_t eff_k = (k_neighbors < max_nbrs) ? k_neighbors : max_nbrs;
  size_t raw_cap = n_vecs * (size_t)eff_k;
  nbase_edge_t *edges = malloc(raw_cap * sizeof(nbase_edge_t));
  nbase_result_t *knn = malloc(((size_t)eff_k + 1U) * sizeof(nbase_result_t));
  if (!edges || !knn) {
    free(edges);
    free(knn);
    free(all_ids);
    free(all_vecs);
    pthread_mutex_unlock(&pdb->mu);
    return NBASE_ERR_NOMEM;
  }

  size_t edge_len = 0U;
  for (size_t i = 0U; i < n_vecs && st == NBASE_OK; i++) {
    uint32_t got = 0U;
    st = pdb_search_unlocked(pdb, all_vecs + i * (size_t)dim, dim, eff_k + 1U, &eff, knn, &got);
    if (st != NBASE_OK) {
      break;
    }
    uint32_t added = 0U;
    for (uint32_t r = 0U; r < got && added < eff_k && edge_len < raw_cap; r++) {
      if (knn[r].id == all_ids[i]) {
        continue;
      }
      if (knn[r].distance <= max_distance) {
        uint64_t a = (all_ids[i] < knn[r].id) ? all_ids[i] : knn[r].id;
        uint64_t b = (all_ids[i] < knn[r].id) ? knn[r].id : all_ids[i];
        edges[edge_len].src_id = a;
        edges[edge_len].dst_id = b;
        edges[edge_len].weight = knn[r].distance;
        edge_len++;
      }
      added++;
    }
  }

  free(knn);
  free(all_ids);
  free(all_vecs);
  pthread_mutex_unlock(&pdb->mu);

  if (st != NBASE_OK) {
    free(edges);
    return st;
  }
  if (edge_len == 0U) {
    free(edges);
    return NBASE_OK;
  }

  qsort(edges, edge_len, sizeof(nbase_edge_t), cmp_pdb_edge_canonical);
  size_t unique_len = 0U;
  for (size_t i = 0U; i < edge_len; i++) {
    if (unique_len == 0U ||
        edges[i].src_id != edges[unique_len - 1U].src_id ||
        edges[i].dst_id != edges[unique_len - 1U].dst_id) {
      edges[unique_len++] = edges[i];
    }
  }

  nbase_edge_t *shrunk = realloc(edges, unique_len * sizeof(nbase_edge_t));
  *out_edges = shrunk ? shrunk : edges;
  *out_edge_count = (uint64_t)unique_len;
  return NBASE_OK;
}

nbase_status_t nbase_pdb_rerank_mmr(
    nbase_pdb_t *pdb,
    const nbase_result_t *candidates,
    uint32_t candidate_count,
    uint32_t k,
    float lambda,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!pdb || !out_count || (candidate_count > 0U && !candidates) ||
      (k > 0U && !out_results) || isnan(lambda) || lambda < 0.0f || lambda > 1.0f) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  if (candidate_count == 0U || k == 0U) {
    return NBASE_OK;
  }
  for (uint32_t i = 0U; i < candidate_count; i++) {
    if (!isfinite(candidates[i].distance)) {
      return NBASE_ERR_INVALID_ARG;
    }
  }

  uint32_t dim = pdb->cfg.dim;
  nbase_metric_t metric = pdb->cfg.metric;
  uint64_t *cand_ids = malloc((size_t)candidate_count * sizeof(uint64_t));
  float *cand_vecs = malloc((size_t)candidate_count * (size_t)dim * sizeof(float));
  bool *cand_found = malloc((size_t)candidate_count * sizeof(bool));
  float *rel = malloc((size_t)candidate_count * sizeof(float));
  float *min_div = malloc((size_t)candidate_count * sizeof(float));
  bool *used = calloc((size_t)candidate_count, sizeof(bool));
  if (!cand_ids || !cand_vecs || !cand_found || !rel || !min_div || !used) {
    free(cand_ids);
    free(cand_vecs);
    free(cand_found);
    free(rel);
    free(min_div);
    free(used);
    return NBASE_ERR_NOMEM;
  }

  for (uint32_t i = 0U; i < candidate_count; i++) {
    cand_ids[i] = candidates[i].id;
  }

  pthread_mutex_lock(&pdb->mu);
  nbase_status_t st = pdb_get_batch_unlocked(pdb, cand_ids, candidate_count, dim, cand_vecs, cand_found);
  pthread_mutex_unlock(&pdb->mu);
  free(cand_ids);

  if (st != NBASE_OK) {
    free(cand_vecs);
    free(cand_found);
    free(rel);
    free(min_div);
    free(used);
    return st;
  }

  float d_min = candidates[0].distance;
  float d_max = candidates[0].distance;
  for (uint32_t i = 0U; i < candidate_count; i++) {
    if (!cand_found[i]) {
      free(cand_vecs);
      free(cand_found);
      free(rel);
      free(min_div);
      free(used);
      return NBASE_ERR_NOT_FOUND;
    }
    if (candidates[i].distance < d_min) {
      d_min = candidates[i].distance;
    }
    if (candidates[i].distance > d_max) {
      d_max = candidates[i].distance;
    }
  }
  free(cand_found);

  uint32_t first = 0U;
  float rel_span = d_max - d_min;
  for (uint32_t i = 0U; i < candidate_count; i++) {
    rel[i] = (rel_span > 0.0f) ? ((d_max - candidates[i].distance) / rel_span) : 1.0f;
    if (candidates[i].distance < candidates[first].distance ||
        (candidates[i].distance == candidates[first].distance && candidates[i].id < candidates[first].id)) {
      first = i;
    }
  }

  uint32_t eff_k = (k < candidate_count) ? k : candidate_count;
  nbase_dist_fn dist_fn = nbase_get_distance_fn(metric);

  out_results[0] = candidates[first];
  used[first] = true;

  float pair_min = 0.0f;
  float pair_max = 0.0f;
  bool first_pair_set = false;
  const float *first_vec = cand_vecs + (size_t)first * (size_t)dim;
  for (uint32_t i = 0U; i < candidate_count; i++) {
    if (used[i]) {
      continue;
    }
    float pd = dist_fn(cand_vecs + (size_t)i * (size_t)dim, first_vec, dim);
    min_div[i] = pd;
    if (!first_pair_set) {
      pair_min = (pd < 0.0f) ? pd : 0.0f;
      pair_max = pd;
      first_pair_set = true;
    } else {
      if (pd < pair_min) {
        pair_min = pd;
      }
      if (pd > pair_max) {
        pair_max = pd;
      }
    }
  }

  float pair_span = pair_max - pair_min;
  for (uint32_t step = 1U; step < eff_k; step++) {
    uint32_t best_idx = UINT32_MAX;
    float best_score = -FLT_MAX;

    for (uint32_t i = 0U; i < candidate_count; i++) {
      if (used[i]) {
        continue;
      }
      float div_norm = (pair_span > 0.0f) ? ((min_div[i] - pair_min) / pair_span) : 0.0f;
      float score = lambda * rel[i] + (1.0f - lambda) * div_norm;
      if (best_idx == UINT32_MAX || score > best_score ||
          (score == best_score && candidates[i].distance < candidates[best_idx].distance)) {
        best_score = score;
        best_idx = i;
      }
    }

    if (best_idx == UINT32_MAX) {
      break;
    }

    out_results[step] = candidates[best_idx];
    used[best_idx] = true;

    const float *picked_vec = cand_vecs + (size_t)best_idx * (size_t)dim;
    for (uint32_t i = 0U; i < candidate_count; i++) {
      if (used[i]) {
        continue;
      }
      float pd = dist_fn(cand_vecs + (size_t)i * (size_t)dim, picked_vec, dim);
      if (pd < min_div[i]) {
        min_div[i] = pd;
      }
    }
  }

  *out_count = eff_k;
  free(cand_vecs);
  free(rel);
  free(min_div);
  free(used);
  return NBASE_OK;
}

nbase_status_t nbase_pdb_rerank_weighted(
    nbase_pdb_t *pdb,
    const nbase_result_t *candidates,
    uint32_t candidate_count,
    const nbase_rerank_weight_t *weights,
    uint32_t weight_count,
    uint32_t k,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!pdb || !out_count || (candidate_count > 0U && !candidates) ||
      (k > 0U && !out_results) || (weight_count > 0U && !weights)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  for (uint32_t i = 0U; i < candidate_count; i++) {
    if (!isfinite(candidates[i].distance)) {
      return NBASE_ERR_INVALID_ARG;
    }
  }
  for (uint32_t w = 0U; w < weight_count; w++) {
    if (!weights[w].key || weights[w].key[0] == '\0' || !isfinite(weights[w].weight)) {
      return NBASE_ERR_INVALID_ARG;
    }
  }
  if (candidate_count == 0U || k == 0U) {
    return NBASE_OK;
  }

  pdb_weighted_cand_t *items = malloc((size_t)candidate_count * sizeof(pdb_weighted_cand_t));
  pdb_part_ref_t *refs = malloc((size_t)candidate_count * sizeof(pdb_part_ref_t));
  if (!items || !refs) {
    free(items);
    free(refs);
    return NBASE_ERR_NOMEM;
  }

  pthread_mutex_lock(&pdb->mu);
  for (uint32_t i = 0U; i < candidate_count; i++) {
    uint32_t pid = 0U;
    if (!pdb_id_map_find(&pdb->id_map, candidates[i].id, &pid)) {
      pthread_mutex_unlock(&pdb->mu);
      free(items);
      free(refs);
      return NBASE_ERR_NOT_FOUND;
    }
    refs[i].part_id = pid;
    refs[i].orig_idx = i;
  }

  qsort(refs, candidate_count, sizeof(pdb_part_ref_t), cmp_pdb_part_ref_asc);

  nbase_status_t st = NBASE_OK;
  uint32_t start = 0U;
  while (start < candidate_count && st == NBASE_OK) {
    uint32_t pid = refs[start].part_id;
    pdb_part_t *part = pdb_find_part(pdb, pid);
    if (!part) {
      st = NBASE_ERR_CORRUPT;
      break;
    }
    st = pdb_load_partition(pdb, part);
    if (st != NBASE_OK) {
      break;
    }

    uint32_t end = start;
    while (end < candidate_count && refs[end].part_id == pid) {
      uint32_t oi = refs[end].orig_idx;
      double w_dist = (double)candidates[oi].distance;
      for (uint32_t w = 0U; w < weight_count; w++) {
        nbase_meta_value_t val;
        if (nbase_get_meta(part->db, candidates[oi].id, weights[w].key, &val, NULL, 0U) == NBASE_OK) {
          if (val.type == NBASE_META_INT64) {
            w_dist -= (double)val.as.i64 * (double)weights[w].weight;
          } else if (val.type == NBASE_META_DOUBLE) {
            w_dist -= val.as.f64 * (double)weights[w].weight;
          }
        }
      }
      items[oi].id = candidates[oi].id;
      items[oi].weighted_dist = (float)w_dist;
      items[oi].orig_dist = candidates[oi].distance;
      end++;
    }
    start = end;
  }
  pthread_mutex_unlock(&pdb->mu);
  free(refs);

  if (st != NBASE_OK) {
    free(items);
    return st;
  }

  qsort(items, candidate_count, sizeof(pdb_weighted_cand_t), cmp_pdb_weighted_asc);
  uint32_t eff_k = (k < candidate_count) ? k : candidate_count;
  for (uint32_t i = 0U; i < eff_k; i++) {
    out_results[i].id = items[i].id;
    out_results[i].distance = items[i].weighted_dist;
  }
  *out_count = eff_k;
  free(items);
  return NBASE_OK;
}

nbase_status_t nbase_pdb_compact(nbase_pdb_t *pdb) {
  if (!pdb) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);
  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < pdb->part_count; i++) {
    st = pdb_load_partition(pdb, pdb->parts[i]);
    if (st != NBASE_OK) {
      break;
    }
    st = nbase_compact(pdb->parts[i]->db);
    if (st != NBASE_OK) {
      break;
    }
    pdb->parts[i]->dirty = true;
  }
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_rebuild_hnsw_mt(nbase_pdb_t *pdb, uint32_t num_threads) {
  if (!pdb) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);
  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < pdb->part_count; i++) {
    st = pdb_load_partition(pdb, pdb->parts[i]);
    if (st != NBASE_OK) {
      break;
    }
    st = nbase_rebuild_hnsw_mt(pdb->parts[i]->db, num_threads);
    if (st != NBASE_OK) {
      break;
    }
    pdb->parts[i]->dirty = true;
  }
  pthread_mutex_unlock(&pdb->mu);
  return st;
}

nbase_status_t nbase_pdb_stats(nbase_pdb_t *pdb, nbase_pdb_stats_t *out_stats) {
  if (!pdb || !out_stats) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_mutex_lock(&pdb->mu);
  memset(out_stats, 0, sizeof(*out_stats));
  out_stats->dim = pdb->cfg.dim;
  out_stats->metric = pdb->cfg.metric;
  out_stats->total_vectors = pdb->total_vectors;
  out_stats->total_partitions = pdb->part_count;
  out_stats->loaded_partitions = pdb->loaded_count;
  out_stats->max_active_partitions = pdb->cfg.max_active_partitions;
  out_stats->active_partition_id = pdb->active_partition_id;
  out_stats->eviction_count = pdb->eviction_count;
  out_stats->cache_hits = pdb->cache_hits;
  out_stats->cache_misses = pdb->cache_misses;

  uint64_t mem = (uint64_t)sizeof(*pdb) + (uint64_t)pdb->part_cap * (uint64_t)sizeof(pdb_part_t *);
  mem += (uint64_t)pdb->id_map.cap *
         ((uint64_t)sizeof(uint64_t) + (uint64_t)sizeof(uint32_t) + (uint64_t)sizeof(uint8_t));
  for (uint32_t i = 0U; i < pdb->part_count; i++) {
    mem += (uint64_t)sizeof(pdb_part_t);
    if (pdb->parts[i]->db != NULL) {
      nbase_stats_t ps;
      if (nbase_stats(pdb->parts[i]->db, &ps) == NBASE_OK) {
        mem += ps.memory_bytes;
      }
    }
  }
  out_stats->memory_bytes = mem;

  pthread_mutex_unlock(&pdb->mu);
  return NBASE_OK;
}
