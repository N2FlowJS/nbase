#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAP_EMPTY 0U
#define MAP_OCCUPIED 1U
#define MAP_DELETED 2U

static inline uint64_t hash_u64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

static bool id_map_init(nbase_id_map_t *map, size_t initial_cap) {
  size_t cap = 16;
  while (cap < initial_cap) {
    if (cap > SIZE_MAX / 2) {
      return false;
    }
    cap *= 2;
  }
  map->keys = calloc(cap, sizeof(uint64_t));
  map->slots = calloc(cap, sizeof(uint32_t));
  map->states = calloc(cap, sizeof(uint8_t));
  if (!map->keys || !map->slots || !map->states) {
    free(map->keys);
    free(map->slots);
    free(map->states);
    memset(map, 0, sizeof(*map));
    return false;
  }
  map->cap = cap;
  map->count = 0;
  map->tombstones = 0;
  return true;
}

static void id_map_free(nbase_id_map_t *map) {
  free(map->keys);
  free(map->slots);
  free(map->states);
  memset(map, 0, sizeof(*map));
}

static bool id_map_rehash(nbase_id_map_t *map, size_t new_cap) {
  nbase_id_map_t next;
  if (!id_map_init(&next, new_cap)) {
    return false;
  }
  size_t mask = next.cap - 1;
  for (size_t i = 0; i < map->cap; i++) {
    if (map->states[i] == MAP_OCCUPIED) {
      size_t idx = (size_t)hash_u64(map->keys[i]) & mask;
      while (next.states[idx] == MAP_OCCUPIED) {
        idx = (idx + 1) & mask;
      }
      next.keys[idx] = map->keys[i];
      next.slots[idx] = map->slots[i];
      next.states[idx] = MAP_OCCUPIED;
      next.count++;
    }
  }
  id_map_free(map);
  *map = next;
  return true;
}

static bool id_map_find(const nbase_id_map_t *map, uint64_t key, uint32_t *out_slot) {
  if (map->cap == 0) {
    return false;
  }
  size_t mask = map->cap - 1;
  size_t idx = (size_t)hash_u64(key) & mask;
  for (size_t probe = 0; probe < map->cap; probe++) {
    uint8_t st = map->states[idx];
    if (st == MAP_EMPTY) {
      return false;
    }
    if (st == MAP_OCCUPIED && map->keys[idx] == key) {
      if (out_slot) {
        *out_slot = map->slots[idx];
      }
      return true;
    }
    idx = (idx + 1) & mask;
  }
  return false;
}

static nbase_status_t id_map_insert(nbase_id_map_t *map, uint64_t key, uint32_t slot) {
  if ((map->count + map->tombstones + 1) * 100 >= map->cap * 65) {
    size_t target = (map->count * 2 >= map->cap) ? map->cap * 2 : map->cap;
    if (!id_map_rehash(map, target)) {
      return NBASE_ERR_NOMEM;
    }
  }

  size_t mask = map->cap - 1;
  size_t idx = (size_t)hash_u64(key) & mask;
  size_t first_deleted = SIZE_MAX;

  for (size_t probe = 0; probe < map->cap; probe++) {
    uint8_t st = map->states[idx];
    if (st == MAP_EMPTY) {
      size_t dst = (first_deleted != SIZE_MAX) ? first_deleted : idx;
      if (map->states[dst] == MAP_DELETED) {
        map->tombstones--;
      }
      map->keys[dst] = key;
      map->slots[dst] = slot;
      map->states[dst] = MAP_OCCUPIED;
      map->count++;
      return NBASE_OK;
    }
    if (st == MAP_OCCUPIED && map->keys[idx] == key) {
      return NBASE_ERR_ALREADY_EXISTS;
    }
    if (st == MAP_DELETED && first_deleted == SIZE_MAX) {
      first_deleted = idx;
    }
    idx = (idx + 1) & mask;
  }

  return NBASE_ERR_NOMEM;
}

static bool id_map_delete(nbase_id_map_t *map, uint64_t key, uint32_t *out_slot) {
  if (map->cap == 0) {
    return false;
  }
  size_t mask = map->cap - 1;
  size_t idx = (size_t)hash_u64(key) & mask;
  for (size_t probe = 0; probe < map->cap; probe++) {
    uint8_t st = map->states[idx];
    if (st == MAP_EMPTY) {
      return false;
    }
    if (st == MAP_OCCUPIED && map->keys[idx] == key) {
      if (out_slot) {
        *out_slot = map->slots[idx];
      }
      map->states[idx] = MAP_DELETED;
      map->count--;
      map->tombstones++;
      return true;
    }
    idx = (idx + 1) & mask;
  }
  return false;
}

nbase_status_t nbase_write_all_fd(int fd, const void *buf, size_t len) {
  const uint8_t *p = (const uint8_t *)buf;
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t n = write(fd, p, remaining);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return NBASE_ERR_IO;
    }
    if (n == 0) {
      return NBASE_ERR_IO;
    }
    p += (size_t)n;
    remaining -= (size_t)n;
  }
  return NBASE_OK;
}

nbase_status_t nbase_read_exact(int fd, void *buf, size_t len) {
  uint8_t *p = (uint8_t *)buf;
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t n = read(fd, p, remaining);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return NBASE_ERR_IO;
    }
    if (n == 0) {
      return NBASE_ERR_CORRUPT;
    }
    p += (size_t)n;
    remaining -= (size_t)n;
  }
  return NBASE_OK;
}

nbase_status_t nbase_ensure_dir(const char *dir_path) {
  if (!dir_path || dir_path[0] == '\0') {
    return NBASE_ERR_INVALID_ARG;
  }
  size_t len = strlen(dir_path);
  char *buf = malloc(len + 1);
  if (!buf) {
    return NBASE_ERR_NOMEM;
  }
  memcpy(buf, dir_path, len + 1);

  for (size_t i = 1; i < len; i++) {
    if (buf[i] == '/') {
      buf[i] = '\0';
      if (mkdir(buf, 0755) != 0 && errno != EEXIST) {
        free(buf);
        return NBASE_ERR_IO;
      }
      buf[i] = '/';
    }
  }
  if (mkdir(buf, 0755) != 0) {
    if (errno != EEXIST) {
      free(buf);
      return NBASE_ERR_IO;
    }
    struct stat st;
    if (stat(buf, &st) != 0 || !S_ISDIR(st.st_mode)) {
      free(buf);
      return NBASE_ERR_IO;
    }
  }
  free(buf);
  return NBASE_OK;
}

nbase_status_t nbase_write_file_atomic(const char *file_path, const void *header, size_t header_len, const nbase_io_chunk_t *chunks, size_t chunk_count) {
  if (!file_path || !header) {
    return NBASE_ERR_INVALID_ARG;
  }
  size_t path_len = strlen(file_path);
  char *tmp_path = malloc(path_len + 5);
  if (!tmp_path) {
    return NBASE_ERR_NOMEM;
  }
  snprintf(tmp_path, path_len + 5, "%s.tmp", file_path);

  int fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    free(tmp_path);
    return NBASE_ERR_IO;
  }

  nbase_status_t st = nbase_write_all_fd(fd, header, header_len);
  for (size_t i = 0; st == NBASE_OK && i < chunk_count; i++) {
    if (chunks[i].len > 0 && chunks[i].ptr != NULL) {
      st = nbase_write_all_fd(fd, chunks[i].ptr, chunks[i].len);
    }
  }
  if (st == NBASE_OK && fsync(fd) != 0) {
    st = NBASE_ERR_IO;
  }
  if (close(fd) != 0 && st == NBASE_OK) {
    st = NBASE_ERR_IO;
  }
  if (st == NBASE_OK && rename(tmp_path, file_path) != 0) {
    st = NBASE_ERR_IO;
  }
  if (st != NBASE_OK) {
    unlink(tmp_path);
    free(tmp_path);
    return st;
  }

  /* Best-effort parent directory fsync for crash durability. */
  const char *slash = strrchr(file_path, '/');
  if (slash != NULL) {
    size_t dir_len = (size_t)(slash - file_path);
    char *dir_path = malloc(dir_len + 2);
    if (dir_path) {
      if (dir_len == 0) {
        strcpy(dir_path, "/");
      } else {
        memcpy(dir_path, file_path, dir_len);
        dir_path[dir_len] = '\0';
      }
      int dir_fd = open(dir_path, O_RDONLY);
      if (dir_fd >= 0) {
        (void)fsync(dir_fd);
        close(dir_fd);
      }
      free(dir_path);
    }
  }

  free(tmp_path);
  return NBASE_OK;
}

static nbase_status_t store_ensure_capacity(nbase_store_t *store, uint32_t needed) {
  if (needed <= store->capacity) {
    return NBASE_OK;
  }
  uint32_t new_cap = store->capacity == 0 ? 64U : store->capacity;
  while (new_cap < needed) {
    if (new_cap > UINT32_MAX / 2U) {
      new_cap = needed;
      break;
    }
    new_cap *= 2U;
  }

  size_t vec_floats = (size_t)new_cap * (size_t)store->dim;
  if (vec_floats > SIZE_MAX / sizeof(float)) {
    return NBASE_ERR_NOMEM;
  }
  float *new_vecs = realloc(store->vectors, vec_floats * sizeof(float));
  if (!new_vecs) {
    return NBASE_ERR_NOMEM;
  }
  store->vectors = new_vecs;

  uint64_t *new_ids = realloc(store->ids, (size_t)new_cap * sizeof(uint64_t));
  if (!new_ids) {
    return NBASE_ERR_NOMEM;
  }
  store->ids = new_ids;

  uint8_t *new_flags = realloc(store->slot_flags, (size_t)new_cap * sizeof(uint8_t));
  if (!new_flags) {
    return NBASE_ERR_NOMEM;
  }
  store->slot_flags = new_flags;
  store->capacity = new_cap;
  return NBASE_OK;
}

nbase_status_t nbase_store_create(uint32_t dim, nbase_metric_t metric, uint32_t initial_capacity, nbase_store_t **out_store) {
  if (!out_store || dim == 0U || dim > NBASE_MAX_DIM || !nbase_metric_is_valid(metric)) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_store_t *store = calloc(1, sizeof(*store));
  if (!store) {
    return NBASE_ERR_NOMEM;
  }
  store->dim = dim;
  store->metric = metric;

  uint32_t cap = initial_capacity > 0U ? initial_capacity : 64U;
  if (!id_map_init(&store->id_map, (size_t)cap * 2U)) {
    free(store);
    return NBASE_ERR_NOMEM;
  }
  nbase_status_t st = store_ensure_capacity(store, cap);
  if (st != NBASE_OK) {
    nbase_store_destroy(store);
    return st;
  }
  *out_store = store;
  return NBASE_OK;
}

void nbase_store_destroy(nbase_store_t *store) {
  if (!store) {
    return;
  }
  id_map_free(&store->id_map);
  if (store->mmap_backed) {
    if (store->mmap_base != NULL && store->mmap_base != MAP_FAILED && store->mmap_size > 0) {
      munmap(store->mmap_base, store->mmap_size);
    }
  } else {
    free(store->vectors);
    free(store->ids);
    free(store->slot_flags);
  }
  free(store);
}

nbase_status_t nbase_store_add(nbase_store_t *store, uint64_t id, const float *vec, uint32_t *out_slot) {
  if (!store || !vec) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (store->mmap_backed) {
    return NBASE_ERR_UNSUPPORTED;
  }
  if (!nbase_vec_is_finite(vec, store->dim)) {
    return NBASE_ERR_INVALID_VECTOR;
  }
  if (id_map_find(&store->id_map, id, NULL)) {
    return NBASE_ERR_ALREADY_EXISTS;
  }
  if (store->slot_count == NBASE_INVALID_SLOT) {
    return NBASE_ERR_NOMEM;
  }

  nbase_status_t st = store_ensure_capacity(store, store->slot_count + 1U);
  if (st != NBASE_OK) {
    return st;
  }

  uint32_t slot = store->slot_count;
  st = id_map_insert(&store->id_map, id, slot);
  if (st != NBASE_OK) {
    return st;
  }

  memcpy(store->vectors + (size_t)slot * (size_t)store->dim, vec, (size_t)store->dim * sizeof(float));
  store->ids[slot] = id;
  store->slot_flags[slot] = NBASE_SLOT_ACTIVE;
  store->slot_count++;
  store->active_count++;
  if (out_slot) {
    *out_slot = slot;
  }
  return NBASE_OK;
}

nbase_status_t nbase_store_update_inplace(nbase_store_t *store, uint32_t slot, const float *vec) {
  if (!store || !vec || slot >= store->slot_count) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (store->mmap_backed) {
    return NBASE_ERR_UNSUPPORTED;
  }
  if (!nbase_vec_is_finite(vec, store->dim)) {
    return NBASE_ERR_INVALID_VECTOR;
  }
  if (store->slot_flags[slot] != NBASE_SLOT_ACTIVE) {
    return NBASE_ERR_NOT_FOUND;
  }
  memcpy(store->vectors + (size_t)slot * (size_t)store->dim, vec, (size_t)store->dim * sizeof(float));
  return NBASE_OK;
}

nbase_status_t nbase_store_delete(nbase_store_t *store, uint64_t id, uint32_t *out_slot) {
  if (!store) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (store->mmap_backed) {
    return NBASE_ERR_UNSUPPORTED;
  }
  uint32_t slot = NBASE_INVALID_SLOT;
  if (!id_map_delete(&store->id_map, id, &slot)) {
    return NBASE_ERR_NOT_FOUND;
  }
  store->slot_flags[slot] = NBASE_SLOT_TOMBSTONE;
  store->active_count--;
  store->tombstone_count++;
  if (out_slot) {
    *out_slot = slot;
  }
  return NBASE_OK;
}

void nbase_store_rollback_slot(nbase_store_t *store, uint32_t slot) {
  if (!store || store->mmap_backed || slot >= store->slot_count) {
    return;
  }
  if (store->slot_flags[slot] != NBASE_SLOT_ACTIVE) {
    return;
  }
  (void)id_map_delete(&store->id_map, store->ids[slot], NULL);
  store->slot_flags[slot] = NBASE_SLOT_TOMBSTONE;
  store->active_count--;
  store->tombstone_count++;
}

nbase_status_t nbase_store_compact(nbase_store_t *store) {
  if (!store) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (store->mmap_backed) {
    return NBASE_ERR_UNSUPPORTED;
  }
  if (store->tombstone_count == 0U) {
    return NBASE_OK;
  }

  uint32_t active = (uint32_t)store->active_count;
  uint32_t new_cap = (active > 64U) ? active : 64U;
  size_t vec_floats = (size_t)new_cap * (size_t)store->dim;
  if (vec_floats > SIZE_MAX / sizeof(float)) {
    return NBASE_ERR_NOMEM;
  }

  float *new_vecs = malloc(vec_floats * sizeof(float));
  uint64_t *new_ids = malloc((size_t)new_cap * sizeof(uint64_t));
  uint8_t *new_flags = malloc((size_t)new_cap * sizeof(uint8_t));
  nbase_id_map_t new_map;
  memset(&new_map, 0, sizeof(new_map));

  if (!new_vecs || !new_ids || !new_flags || !id_map_init(&new_map, (size_t)new_cap * 2U)) {
    free(new_vecs);
    free(new_ids);
    free(new_flags);
    id_map_free(&new_map);
    return NBASE_ERR_NOMEM;
  }

  uint32_t dst = 0U;
  for (uint32_t src = 0U; src < store->slot_count; src++) {
    if (store->slot_flags[src] != NBASE_SLOT_ACTIVE) {
      continue;
    }
    memcpy(new_vecs + (size_t)dst * (size_t)store->dim, store->vectors + (size_t)src * (size_t)store->dim, (size_t)store->dim * sizeof(float));
    new_ids[dst] = store->ids[src];
    new_flags[dst] = NBASE_SLOT_ACTIVE;
    nbase_status_t st = id_map_insert(&new_map, new_ids[dst], dst);
    if (st != NBASE_OK) {
      free(new_vecs);
      free(new_ids);
      free(new_flags);
      id_map_free(&new_map);
      return st;
    }
    dst++;
  }

  free(store->vectors);
  free(store->ids);
  free(store->slot_flags);
  id_map_free(&store->id_map);

  store->vectors = new_vecs;
  store->ids = new_ids;
  store->slot_flags = new_flags;
  store->id_map = new_map;
  store->capacity = new_cap;
  store->slot_count = dst;
  store->tombstone_count = 0ULL;
  return NBASE_OK;
}

bool nbase_store_find_slot(const nbase_store_t *store, uint64_t id, uint32_t *out_slot) {
  if (!store) {
    return false;
  }
  return id_map_find(&store->id_map, id, out_slot);
}

const float *nbase_store_get_vector(const nbase_store_t *store, uint32_t slot) {
  if (!store || slot >= store->slot_count || store->slot_flags[slot] != NBASE_SLOT_ACTIVE) {
    return NULL;
  }
  return store->vectors + (size_t)slot * (size_t)store->dim;
}

static bool exact_heap_consider(
    nbase_heap_t *heap,
    uint32_t k,
    const uint64_t *ids,
    uint32_t slot,
    float dist) {
  if (heap->len < k) {
    return nbase_heap_push(heap, (nbase_heap_item_t){slot, dist});
  }
  nbase_heap_item_t top = heap->data[0];
  if (dist < top.dist || (dist == top.dist && ids[slot] < ids[top.slot])) {
    (void)nbase_heap_pop(heap);
    return nbase_heap_push(heap, (nbase_heap_item_t){slot, dist});
  }
  return true;
}

static nbase_status_t store_exact_search_internal(
    const nbase_store_t *store,
    const float *query,
    uint32_t k,
    bool use_radius,
    float max_distance,
    nbase_metric_t metric,
    nbase_filter_fn filter,
    void *filter_ctx,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!store || !query || !out_count || (k > 0U && !out_results) ||
      (use_radius && !isfinite(max_distance))) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0;
  if (k == 0U || store->active_count == 0U) {
    return NBASE_OK;
  }

  nbase_heap_t heap;
  nbase_heap_init(&heap, true);

  if (filter == NULL) {
    float batch_dists[256] = {0.0f};
    uint32_t slot = 0U;
    while (slot < store->slot_count) {
      while (slot < store->slot_count && store->slot_flags[slot] != NBASE_SLOT_ACTIVE) {
        slot++;
      }
      if (slot >= store->slot_count) {
        break;
      }
      uint32_t start = slot;
      while (slot < store->slot_count &&
             store->slot_flags[slot] == NBASE_SLOT_ACTIVE &&
             (slot - start) < 256U) {
        slot++;
      }
      uint32_t run_len = slot - start;
      nbase_status_t st = nbase_distance_batch(
          metric,
          query,
          store->vectors + (size_t)start * (size_t)store->dim,
          run_len,
          store->dim,
          batch_dists);
      if (st != NBASE_OK) {
        nbase_heap_free(&heap);
        return st;
      }
      for (uint32_t bi = 0U; bi < run_len; bi++) {
        float dist = batch_dists[bi];
        if (use_radius && dist > max_distance) {
          continue;
        }
        if (!exact_heap_consider(&heap, k, store->ids, start + bi, dist)) {
          nbase_heap_free(&heap);
          return NBASE_ERR_NOMEM;
        }
      }
    }
  } else {
    nbase_dist_fn dist_fn = nbase_get_distance_fn(metric);
    for (uint32_t slot = 0U; slot < store->slot_count; slot++) {
      if (store->slot_flags[slot] != NBASE_SLOT_ACTIVE) {
        continue;
      }
      uint64_t id = store->ids[slot];
      if (!filter(id, filter_ctx)) {
        continue;
      }
      float dist = dist_fn(query, store->vectors + (size_t)slot * (size_t)store->dim, store->dim);
      if (use_radius && dist > max_distance) {
        continue;
      }
      if (!exact_heap_consider(&heap, k, store->ids, slot, dist)) {
        nbase_heap_free(&heap);
        return NBASE_ERR_NOMEM;
      }
    }
  }

  uint32_t n = (uint32_t)heap.len;
  for (uint32_t i = 0; i < n; i++) {
    out_results[i].id = store->ids[heap.data[i].slot];
    out_results[i].distance = heap.data[i].dist;
  }
  nbase_heap_free(&heap);

  qsort(out_results, n, sizeof(nbase_result_t), nbase_cmp_result_asc);
  *out_count = n;
  return NBASE_OK;
}

nbase_status_t nbase_store_exact_search(
    const nbase_store_t *store,
    const float *query,
    uint32_t k,
    nbase_metric_t metric,
    nbase_filter_fn filter,
    void *filter_ctx,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  return store_exact_search_internal(
      store, query, k, false, 0.0f, metric, filter, filter_ctx, out_results, out_count);
}

nbase_status_t nbase_store_exact_search_radius(
    const nbase_store_t *store,
    const float *query,
    uint32_t k,
    float max_distance,
    nbase_metric_t metric,
    nbase_filter_fn filter,
    void *filter_ctx,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  return store_exact_search_internal(
      store, query, k, true, max_distance, metric, filter, filter_ctx, out_results, out_count);
}

nbase_status_t nbase_store_save(const nbase_store_t *store, const char *file_path) {
  if (!store || !file_path) {
    return NBASE_ERR_INVALID_ARG;
  }

  uint64_t vec_bytes = (uint64_t)store->slot_count * (uint64_t)store->dim * sizeof(float);
  uint64_t ids_rel = (vec_bytes + 7ULL) & ~7ULL;
  uint64_t pad_bytes = ids_rel - vec_bytes;
  uint64_t ids_bytes = (uint64_t)store->slot_count * sizeof(uint64_t);
  uint64_t flags_rel = ids_rel + ids_bytes;
  uint64_t flags_bytes = (uint64_t)store->slot_count * sizeof(uint8_t);
  uint64_t payload_bytes = flags_rel + flags_bytes;

  static const uint8_t zeros[8] = {0};

  uint32_t payload_crc = 0U;
  payload_crc = nbase_crc32_update(payload_crc, store->vectors, (size_t)vec_bytes);
  payload_crc = nbase_crc32_update(payload_crc, zeros, (size_t)pad_bytes);
  payload_crc = nbase_crc32_update(payload_crc, store->ids, (size_t)ids_bytes);
  payload_crc = nbase_crc32_update(payload_crc, store->slot_flags, (size_t)flags_bytes);

  nbv_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBV_MAGIC, 8);
  hdr.version = NBASE_NBV_VERSION;
  hdr.dim = store->dim;
  hdr.metric = (uint32_t)store->metric;
  hdr.flags = 0U;
  hdr.slot_count = store->slot_count;
  hdr.active_count = store->active_count;
  hdr.vectors_offset = sizeof(nbv_header_t);
  hdr.ids_offset = sizeof(nbv_header_t) + ids_rel;
  hdr.slot_flags_offset = sizeof(nbv_header_t) + flags_rel;
  hdr.payload_bytes = payload_bytes;
  hdr.payload_crc32 = payload_crc;
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nbv_header_t, header_crc32));

  nbase_io_chunk_t chunks[4] = {
      {store->vectors, (size_t)vec_bytes},
      {zeros, (size_t)pad_bytes},
      {store->ids, (size_t)ids_bytes},
      {store->slot_flags, (size_t)flags_bytes},
  };

  return nbase_write_file_atomic(file_path, &hdr, sizeof(hdr), chunks, 4);
}

nbase_status_t nbase_store_load(const char *file_path, bool mmap_mode, nbase_store_t **out_store) {
  if (!file_path || !out_store) {
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
  if (st_buf.st_size < (off_t)sizeof(nbv_header_t)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;

  nbv_header_t hdr;
  nbase_status_t status = nbase_read_exact(fd, &hdr, sizeof(hdr));
  if (status != NBASE_OK) {
    close(fd);
    return status;
  }

  if (memcmp(hdr.magic, NBASE_NBV_MAGIC, 8) != 0 || hdr.version != NBASE_NBV_VERSION) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint32_t expected_hdr_crc = nbase_crc32(&hdr, offsetof(nbv_header_t, header_crc32));
  if (hdr.header_crc32 != expected_hdr_crc) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.dim == 0U || hdr.dim > NBASE_MAX_DIM || !nbase_metric_is_valid((nbase_metric_t)hdr.metric)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.slot_count >= NBASE_INVALID_SLOT || hdr.active_count > hdr.slot_count) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint64_t vec_bytes = hdr.slot_count * (uint64_t)hdr.dim * sizeof(float);
  uint64_t ids_rel = (vec_bytes + 7ULL) & ~7ULL;
  uint64_t pad_bytes = ids_rel - vec_bytes;
  uint64_t ids_bytes = hdr.slot_count * sizeof(uint64_t);
  uint64_t flags_rel = ids_rel + ids_bytes;
  uint64_t flags_bytes = hdr.slot_count * sizeof(uint8_t);
  uint64_t expected_payload = flags_rel + flags_bytes;

  if (hdr.vectors_offset != sizeof(nbv_header_t) || hdr.ids_offset != sizeof(nbv_header_t) + ids_rel || hdr.slot_flags_offset != sizeof(nbv_header_t) + flags_rel || hdr.payload_bytes != expected_payload || file_size != sizeof(nbv_header_t) + expected_payload) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  nbase_store_t *store = calloc(1, sizeof(*store));
  if (!store) {
    close(fd);
    return NBASE_ERR_NOMEM;
  }
  store->dim = hdr.dim;
  store->metric = (nbase_metric_t)hdr.metric;
  store->slot_count = (uint32_t)hdr.slot_count;
  store->mmap_backed = mmap_mode;

  if (mmap_mode) {
    if (file_size > (uint64_t)SIZE_MAX) {
      close(fd);
      free(store);
      return NBASE_ERR_NOMEM;
    }
    void *mapped = mmap(NULL, (size_t)file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
      free(store);
      return NBASE_ERR_IO;
    }
    store->mmap_base = mapped;
    store->mmap_size = (size_t)file_size;

    const uint8_t *base = (const uint8_t *)mapped;
    uint32_t actual_crc = nbase_crc32(base + sizeof(nbv_header_t), (size_t)expected_payload);
    if (actual_crc != hdr.payload_crc32) {
      nbase_store_destroy(store);
      return NBASE_ERR_CORRUPT;
    }
    if (store->slot_count > 0U) {
      store->vectors = (float *)(uintptr_t)(base + hdr.vectors_offset);
      store->ids = (uint64_t *)(uintptr_t)(base + hdr.ids_offset);
      store->slot_flags = (uint8_t *)(uintptr_t)(base + hdr.slot_flags_offset);
    }
    store->capacity = store->slot_count;
  } else {
    uint32_t cap = store->slot_count > 0U ? store->slot_count : 64U;
    status = store_ensure_capacity(store, cap);
    if (status != NBASE_OK) {
      close(fd);
      nbase_store_destroy(store);
      return status;
    }

    uint8_t pad_buf[8] = {0};
    if (vec_bytes > 0U) {
      status = nbase_read_exact(fd, store->vectors, (size_t)vec_bytes);
    }
    if (status == NBASE_OK && pad_bytes > 0U) {
      status = nbase_read_exact(fd, pad_buf, (size_t)pad_bytes);
    }
    if (status == NBASE_OK && ids_bytes > 0U) {
      status = nbase_read_exact(fd, store->ids, (size_t)ids_bytes);
    }
    if (status == NBASE_OK && flags_bytes > 0U) {
      status = nbase_read_exact(fd, store->slot_flags, (size_t)flags_bytes);
    }
    close(fd);
    if (status != NBASE_OK) {
      nbase_store_destroy(store);
      return status;
    }

    uint32_t actual_crc = 0U;
    actual_crc = nbase_crc32_update(actual_crc, store->vectors, (size_t)vec_bytes);
    actual_crc = nbase_crc32_update(actual_crc, pad_buf, (size_t)pad_bytes);
    actual_crc = nbase_crc32_update(actual_crc, store->ids, (size_t)ids_bytes);
    actual_crc = nbase_crc32_update(actual_crc, store->slot_flags, (size_t)flags_bytes);
    if (actual_crc != hdr.payload_crc32) {
      nbase_store_destroy(store);
      return NBASE_ERR_CORRUPT;
    }
  }

  size_t map_cap = (size_t)store->slot_count * 2U;
  if (map_cap < 16U) {
    map_cap = 16U;
  }
  if (!id_map_init(&store->id_map, map_cap)) {
    nbase_store_destroy(store);
    return NBASE_ERR_NOMEM;
  }

  uint64_t active = 0;
  uint64_t tombstones = 0;
  for (uint32_t slot = 0; slot < store->slot_count; slot++) {
    uint8_t flag = store->slot_flags[slot];
    if (flag == NBASE_SLOT_ACTIVE) {
      if (!nbase_vec_is_finite(store->vectors + (size_t)slot * (size_t)store->dim, store->dim)) {
        nbase_store_destroy(store);
        return NBASE_ERR_CORRUPT;
      }
      nbase_status_t ins = id_map_insert(&store->id_map, store->ids[slot], slot);
      if (ins == NBASE_ERR_ALREADY_EXISTS) {
        nbase_store_destroy(store);
        return NBASE_ERR_CORRUPT;
      }
      if (ins != NBASE_OK) {
        nbase_store_destroy(store);
        return ins;
      }
      active++;
    } else if (flag == NBASE_SLOT_TOMBSTONE) {
      tombstones++;
    } else {
      nbase_store_destroy(store);
      return NBASE_ERR_CORRUPT;
    }
  }

  if (active != hdr.active_count) {
    nbase_store_destroy(store);
    return NBASE_ERR_CORRUPT;
  }
  store->active_count = active;
  store->tombstone_count = tombstones;

  *out_store = store;
  return NBASE_OK;
}

uint64_t nbase_store_memory_bytes(const nbase_store_t *store) {
  if (!store) {
    return 0ULL;
  }
  uint64_t total = (uint64_t)sizeof(*store);
  if (store->mmap_backed) {
    total += (uint64_t)store->mmap_size;
  } else {
    total += (uint64_t)store->capacity * (uint64_t)store->dim * (uint64_t)sizeof(float);
    total += (uint64_t)store->capacity * (uint64_t)sizeof(uint64_t);
    total += (uint64_t)store->capacity * (uint64_t)sizeof(uint8_t);
  }
  total += (uint64_t)store->id_map.cap *
           ((uint64_t)sizeof(uint64_t) + (uint64_t)sizeof(uint32_t) + (uint64_t)sizeof(uint8_t));
  return total;
}
