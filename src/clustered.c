#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct cl_cluster {
  float *centroid;
  uint64_t *member_ids;
  uint32_t count;
  uint32_t cap;
} cl_cluster_t;

struct nbase_clustered {
  nbase_clustered_config_t cfg;
  nbase_store_t *store;
  cl_cluster_t *clusters;
  uint32_t cluster_count;
  uint32_t cluster_cap;
  uint32_t *slot_cluster;
  uint32_t slot_cap;
  pthread_rwlock_t rwlock;
  bool rwlock_init;
};

typedef struct cl_probe {
  uint32_t cluster_idx;
  float dist;
} cl_probe_t;

typedef struct cl_maxheap {
  nbase_result_t *data;
  uint32_t len;
  uint32_t cap;
} cl_maxheap_t;

static int cmp_cl_u64_asc(const void *pa, const void *pb) {
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

static int cmp_cl_probe_asc(const void *pa, const void *pb) {
  const cl_probe_t *a = (const cl_probe_t *)pa;
  const cl_probe_t *b = (const cl_probe_t *)pb;
  if (a->dist < b->dist) {
    return -1;
  }
  if (a->dist > b->dist) {
    return 1;
  }
  if (a->cluster_idx < b->cluster_idx) {
    return -1;
  }
  if (a->cluster_idx > b->cluster_idx) {
    return 1;
  }
  return 0;
}

static bool cl_res_higher(nbase_result_t a, nbase_result_t b) {
  return a.distance > b.distance || (a.distance == b.distance && a.id > b.id);
}

static void cl_heap_push_bounded(cl_maxheap_t *h, uint64_t id, float dist) {
  nbase_result_t item;
  item.id = id;
  item.distance = dist;
  if (h->len < h->cap) {
    uint32_t idx = h->len++;
    while (idx > 0U) {
      uint32_t parent = (idx - 1U) / 2U;
      if (!cl_res_higher(item, h->data[parent])) {
        break;
      }
      h->data[idx] = h->data[parent];
      idx = parent;
    }
    h->data[idx] = item;
  } else if (h->len > 0U && cl_res_higher(h->data[0], item)) {
    uint32_t idx = 0U;
    while (true) {
      uint32_t left = 2U * idx + 1U;
      if (left >= h->len) {
        break;
      }
      uint32_t right = left + 1U;
      uint32_t best = left;
      if (right < h->len && cl_res_higher(h->data[right], h->data[left])) {
        best = right;
      }
      if (!cl_res_higher(h->data[best], item)) {
        break;
      }
      h->data[idx] = h->data[best];
      idx = best;
    }
    h->data[idx] = item;
  }
}

static void free_cluster_array(cl_cluster_t *clusters, uint32_t count) {
  if (!clusters) {
    return;
  }
  for (uint32_t i = 0U; i < count; i++) {
    free(clusters[i].centroid);
    free(clusters[i].member_ids);
  }
  free(clusters);
}

static bool ensure_slot_cluster_cap(nbase_clustered_t *cl, uint32_t min_slots) {
  if (cl->slot_cap >= min_slots) {
    return true;
  }
  uint32_t new_cap = (cl->slot_cap == 0U) ? 64U : cl->slot_cap * 2U;
  while (new_cap < min_slots) {
    if (new_cap > UINT32_MAX / 2U) {
      new_cap = min_slots;
      break;
    }
    new_cap *= 2U;
  }
  uint32_t *next = realloc(cl->slot_cluster, (size_t)new_cap * sizeof(uint32_t));
  if (!next) {
    return false;
  }
  for (uint32_t i = cl->slot_cap; i < new_cap; i++) {
    next[i] = UINT32_MAX;
  }
  cl->slot_cluster = next;
  cl->slot_cap = new_cap;
  return true;
}

static bool ensure_cluster_member_cap(cl_cluster_t *c, uint32_t min_cap) {
  if (c->cap >= min_cap) {
    return true;
  }
  uint32_t new_cap = (c->cap == 0U) ? 16U : c->cap * 2U;
  while (new_cap < min_cap) {
    new_cap *= 2U;
  }
  uint64_t *next = realloc(c->member_ids, (size_t)new_cap * sizeof(uint64_t));
  if (!next) {
    return false;
  }
  c->member_ids = next;
  c->cap = new_cap;
  return true;
}

static nbase_status_t append_empty_cluster(nbase_clustered_t *cl, uint32_t *out_idx) {
  if (cl->cluster_count == cl->cluster_cap) {
    uint32_t new_cap = (cl->cluster_cap == 0U) ? 16U : cl->cluster_cap * 2U;
    cl_cluster_t *next = realloc(cl->clusters, (size_t)new_cap * sizeof(cl_cluster_t));
    if (!next) {
      return NBASE_ERR_NOMEM;
    }
    cl->clusters = next;
    cl->cluster_cap = new_cap;
  }

  float *cent = calloc((size_t)cl->cfg.dim, sizeof(float));
  if (!cent) {
    return NBASE_ERR_NOMEM;
  }
  uint32_t idx = cl->cluster_count++;
  cl->clusters[idx].centroid = cent;
  cl->clusters[idx].member_ids = NULL;
  cl->clusters[idx].count = 0U;
  cl->clusters[idx].cap = 0U;
  *out_idx = idx;
  return NBASE_OK;
}

nbase_clustered_config_t nbase_clustered_config_default(uint32_t dim) {
  nbase_clustered_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.dim = dim;
  cfg.metric = NBASE_METRIC_EUCLIDEAN;
  cfg.target_cluster_size = 128U;
  cfg.max_clusters = 256U;
  cfg.distance_threshold = 1.0f;
  cfg.kmeans_max_iters = 20U;
  cfg.seed = 0x434C555354455231ULL;
  return cfg;
}

nbase_status_t nbase_clustered_create(const nbase_clustered_config_t *cfg, nbase_clustered_t **out_cl) {
  if (!cfg || !out_cl) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_cl = NULL;
  if (cfg->dim == 0U || cfg->dim > NBASE_MAX_DIM || !nbase_metric_is_valid(cfg->metric) ||
      cfg->target_cluster_size == 0U || cfg->max_clusters == 0U ||
      !isfinite(cfg->distance_threshold) || cfg->distance_threshold < 0.0f) {
    return NBASE_ERR_INVALID_ARG;
  }

  nbase_clustered_t *cl = calloc(1, sizeof(*cl));
  if (!cl) {
    return NBASE_ERR_NOMEM;
  }
  cl->cfg = *cfg;
  if (cl->cfg.kmeans_max_iters == 0U) {
    cl->cfg.kmeans_max_iters = 20U;
  }
  if (cl->cfg.seed == 0ULL) {
    cl->cfg.seed = 0x434C555354455231ULL;
  }

  if (pthread_rwlock_init(&cl->rwlock, NULL) != 0) {
    free(cl);
    return NBASE_ERR_NOMEM;
  }
  cl->rwlock_init = true;

  nbase_status_t st = nbase_store_create(cl->cfg.dim, cl->cfg.metric, 64U, &cl->store);
  if (st != NBASE_OK) {
    pthread_rwlock_destroy(&cl->rwlock);
    free(cl);
    return st;
  }

  if (!ensure_slot_cluster_cap(cl, 64U)) {
    nbase_clustered_destroy(cl);
    return NBASE_ERR_NOMEM;
  }

  *out_cl = cl;
  return NBASE_OK;
}

void nbase_clustered_destroy(nbase_clustered_t *cl) {
  if (!cl) {
    return;
  }
  free_cluster_array(cl->clusters, cl->cluster_count);
  free(cl->slot_cluster);
  nbase_store_destroy(cl->store);
  if (cl->rwlock_init) {
    pthread_rwlock_destroy(&cl->rwlock);
  }
  free(cl);
}

static nbase_status_t cl_add_unlocked(
    nbase_clustered_t *cl,
    uint64_t id,
    const float *vec,
    uint32_t dim) {
  if (dim != cl->cfg.dim) {
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (!nbase_vec_is_finite(vec, dim)) {
    return NBASE_ERR_INVALID_VECTOR;
  }
  if (nbase_store_find_slot(cl->store, id, NULL)) {
    return NBASE_ERR_ALREADY_EXISTS;
  }

  nbase_dist_fn dfn = nbase_get_distance_fn(cl->cfg.metric);
  uint32_t best_nonfull = UINT32_MAX;
  float best_nonfull_dist = FLT_MAX;
  uint32_t best_any = UINT32_MAX;
  float best_any_dist = FLT_MAX;
  uint32_t empty_idx = UINT32_MAX;

  for (uint32_t c = 0U; c < cl->cluster_count; c++) {
    const cl_cluster_t *clst = &cl->clusters[c];
    if (clst->count == 0U) {
      if (empty_idx == UINT32_MAX) {
        empty_idx = c;
      }
      continue;
    }
    float d = dfn(vec, clst->centroid, dim);
    if (d < best_any_dist) {
      best_any_dist = d;
      best_any = c;
    }
    if (clst->count < cl->cfg.target_cluster_size && d < best_nonfull_dist) {
      best_nonfull_dist = d;
      best_nonfull = c;
    }
  }

  uint32_t target_c = UINT32_MAX;
  bool create_new = false;
  if (best_nonfull != UINT32_MAX && best_nonfull_dist <= cl->cfg.distance_threshold) {
    target_c = best_nonfull;
  } else if (empty_idx != UINT32_MAX) {
    target_c = empty_idx;
  } else if (cl->cluster_count < cl->cfg.max_clusters) {
    create_new = true;
  } else if (best_nonfull != UINT32_MAX) {
    target_c = best_nonfull;
  } else {
    target_c = best_any;
  }

  uint32_t slot = NBASE_INVALID_SLOT;
  nbase_status_t st = nbase_store_add(cl->store, id, vec, &slot);
  if (st != NBASE_OK) {
    return st;
  }

  if (!ensure_slot_cluster_cap(cl, slot + 1U)) {
    nbase_store_rollback_slot(cl->store, slot);
    return NBASE_ERR_NOMEM;
  }

  if (create_new) {
    st = append_empty_cluster(cl, &target_c);
    if (st != NBASE_OK) {
      nbase_store_rollback_slot(cl->store, slot);
      return st;
    }
  }

  cl_cluster_t *dst = &cl->clusters[target_c];
  if (!ensure_cluster_member_cap(dst, dst->count + 1U)) {
    nbase_store_rollback_slot(cl->store, slot);
    return NBASE_ERR_NOMEM;
  }

  uint32_t old_n = dst->count;
  if (old_n == 0U) {
    memcpy(dst->centroid, vec, (size_t)dim * sizeof(float));
  } else {
    float inv_new = 1.0f / (float)(old_n + 1U);
    float f_old = (float)old_n;
    for (uint32_t d = 0U; d < dim; d++) {
      dst->centroid[d] = (dst->centroid[d] * f_old + vec[d]) * inv_new;
    }
  }

  dst->member_ids[dst->count++] = id;
  cl->slot_cluster[slot] = target_c;
  return NBASE_OK;
}

static nbase_status_t cl_delete_unlocked(nbase_clustered_t *cl, uint64_t id) {
  uint32_t slot = NBASE_INVALID_SLOT;
  if (!nbase_store_find_slot(cl->store, id, &slot)) {
    return NBASE_ERR_NOT_FOUND;
  }

  uint32_t cidx = (slot < cl->slot_cap) ? cl->slot_cluster[slot] : UINT32_MAX;
  if (cidx < cl->cluster_count) {
    cl_cluster_t *c = &cl->clusters[cidx];
    for (uint32_t i = 0U; i < c->count; i++) {
      if (c->member_ids[i] == id) {
        c->member_ids[i] = c->member_ids[c->count - 1U];
        c->count--;
        break;
      }
    }

    uint32_t dim = cl->cfg.dim;
    memset(c->centroid, 0, (size_t)dim * sizeof(float));
    if (c->count > 0U) {
      for (uint32_t i = 0U; i < c->count; i++) {
        uint32_t mslot = NBASE_INVALID_SLOT;
        if (nbase_store_find_slot(cl->store, c->member_ids[i], &mslot)) {
          const float *mv = nbase_store_get_vector(cl->store, mslot);
          for (uint32_t d = 0U; d < dim; d++) {
            c->centroid[d] += mv[d];
          }
        }
      }
      float inv = 1.0f / (float)c->count;
      for (uint32_t d = 0U; d < dim; d++) {
        c->centroid[d] *= inv;
      }
    }
  }

  if (slot < cl->slot_cap) {
    cl->slot_cluster[slot] = UINT32_MAX;
  }
  return nbase_store_delete(cl->store, id, NULL);
}

nbase_status_t nbase_clustered_add(
    nbase_clustered_t *cl,
    uint64_t id,
    const float *vec,
    uint32_t dim) {
  if (!cl || !vec) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&cl->rwlock);
  nbase_status_t st = cl_add_unlocked(cl, id, vec, dim);
  pthread_rwlock_unlock(&cl->rwlock);
  return st;
}

nbase_status_t nbase_clustered_add_batch(
    nbase_clustered_t *cl,
    const uint64_t *ids,
    const float *vectors,
    uint32_t count,
    uint32_t dim) {
  if (!cl || (count > 0U && (!ids || !vectors))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  pthread_rwlock_wrlock(&cl->rwlock);
  if (dim != cl->cfg.dim) {
    pthread_rwlock_unlock(&cl->rwlock);
    return NBASE_ERR_DIM_MISMATCH;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)dim, dim)) {
      pthread_rwlock_unlock(&cl->rwlock);
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  uint64_t *sorted = malloc((size_t)count * sizeof(uint64_t));
  if (!sorted) {
    pthread_rwlock_unlock(&cl->rwlock);
    return NBASE_ERR_NOMEM;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (nbase_store_find_slot(cl->store, ids[i], NULL)) {
      free(sorted);
      pthread_rwlock_unlock(&cl->rwlock);
      return NBASE_ERR_ALREADY_EXISTS;
    }
    sorted[i] = ids[i];
  }
  qsort(sorted, count, sizeof(uint64_t), cmp_cl_u64_asc);
  for (uint32_t i = 1U; i < count; i++) {
    if (sorted[i] == sorted[i - 1U]) {
      free(sorted);
      pthread_rwlock_unlock(&cl->rwlock);
      return NBASE_ERR_ALREADY_EXISTS;
    }
  }
  free(sorted);

  nbase_status_t st = NBASE_OK;
  uint32_t added = 0U;
  for (uint32_t i = 0U; i < count; i++) {
    st = cl_add_unlocked(cl, ids[i], vectors + (size_t)i * (size_t)dim, dim);
    if (st != NBASE_OK) {
      break;
    }
    added++;
  }
  if (st != NBASE_OK) {
    for (uint32_t j = 0U; j < added; j++) {
      (void)cl_delete_unlocked(cl, ids[j]);
    }
  }

  pthread_rwlock_unlock(&cl->rwlock);
  return st;
}

nbase_status_t nbase_clustered_delete(nbase_clustered_t *cl, uint64_t id) {
  if (!cl) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&cl->rwlock);
  nbase_status_t st = cl_delete_unlocked(cl, id);
  pthread_rwlock_unlock(&cl->rwlock);
  return st;
}

nbase_status_t nbase_clustered_delete_batch(
    nbase_clustered_t *cl,
    const uint64_t *ids,
    uint32_t count,
    uint32_t *out_deleted) {
  if (!cl || (count > 0U && !ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (out_deleted) {
    *out_deleted = 0U;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  pthread_rwlock_wrlock(&cl->rwlock);
  uint32_t deleted = 0U;
  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < count; i++) {
    nbase_status_t ds = cl_delete_unlocked(cl, ids[i]);
    if (ds == NBASE_ERR_NOT_FOUND) {
      continue;
    }
    if (ds != NBASE_OK) {
      st = ds;
      break;
    }
    deleted++;
  }
  pthread_rwlock_unlock(&cl->rwlock);
  if (out_deleted) {
    *out_deleted = deleted;
  }
  return st;
}

nbase_status_t nbase_clustered_upsert(
    nbase_clustered_t *cl,
    uint64_t id,
    const float *vec,
    uint32_t dim) {
  if (!cl || !vec) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&cl->rwlock);
  if (dim != cl->cfg.dim) {
    pthread_rwlock_unlock(&cl->rwlock);
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (!nbase_vec_is_finite(vec, dim)) {
    pthread_rwlock_unlock(&cl->rwlock);
    return NBASE_ERR_INVALID_VECTOR;
  }
  if (nbase_store_find_slot(cl->store, id, NULL)) {
    nbase_status_t st = cl_delete_unlocked(cl, id);
    if (st != NBASE_OK) {
      pthread_rwlock_unlock(&cl->rwlock);
      return st;
    }
  }
  nbase_status_t st = cl_add_unlocked(cl, id, vec, dim);
  pthread_rwlock_unlock(&cl->rwlock);
  return st;
}

nbase_status_t nbase_clustered_upsert_batch(
    nbase_clustered_t *cl,
    const uint64_t *ids,
    const float *vectors,
    uint32_t count,
    uint32_t dim) {
  if (!cl || (count > 0U && (!ids || !vectors))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  pthread_rwlock_wrlock(&cl->rwlock);
  if (dim != cl->cfg.dim) {
    pthread_rwlock_unlock(&cl->rwlock);
    return NBASE_ERR_DIM_MISMATCH;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)dim, dim)) {
      pthread_rwlock_unlock(&cl->rwlock);
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  uint64_t *sorted = malloc((size_t)count * sizeof(uint64_t));
  if (!sorted) {
    pthread_rwlock_unlock(&cl->rwlock);
    return NBASE_ERR_NOMEM;
  }
  memcpy(sorted, ids, (size_t)count * sizeof(uint64_t));
  qsort(sorted, count, sizeof(uint64_t), cmp_cl_u64_asc);
  for (uint32_t i = 1U; i < count; i++) {
    if (sorted[i] == sorted[i - 1U]) {
      free(sorted);
      pthread_rwlock_unlock(&cl->rwlock);
      return NBASE_ERR_ALREADY_EXISTS;
    }
  }
  free(sorted);

  nbase_status_t st = NBASE_OK;
  for (uint32_t i = 0U; i < count; i++) {
    if (nbase_store_find_slot(cl->store, ids[i], NULL)) {
      st = cl_delete_unlocked(cl, ids[i]);
      if (st != NBASE_OK) {
        break;
      }
    }
    st = cl_add_unlocked(cl, ids[i], vectors + (size_t)i * (size_t)dim, dim);
    if (st != NBASE_OK) {
      break;
    }
  }
  pthread_rwlock_unlock(&cl->rwlock);
  return st;
}

nbase_status_t nbase_clustered_get(
    const nbase_clustered_t *cl,
    uint64_t id,
    float *out_vec,
    uint32_t dim) {
  if (!cl || !out_vec) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_clustered_t *mcl = (nbase_clustered_t *)(uintptr_t)cl;
  pthread_rwlock_rdlock(&mcl->rwlock);
  nbase_status_t st = NBASE_OK;
  if (dim != cl->cfg.dim) {
    st = NBASE_ERR_DIM_MISMATCH;
  } else {
    uint32_t slot = NBASE_INVALID_SLOT;
    if (!nbase_store_find_slot(cl->store, id, &slot)) {
      st = NBASE_ERR_NOT_FOUND;
    } else {
      const float *v = nbase_store_get_vector(cl->store, slot);
      if (!v) {
        st = NBASE_ERR_NOT_FOUND;
      } else {
        memcpy(out_vec, v, (size_t)dim * sizeof(float));
      }
    }
  }
  pthread_rwlock_unlock(&mcl->rwlock);
  return st;
}

bool nbase_clustered_contains(const nbase_clustered_t *cl, uint64_t id) {
  if (!cl) {
    return false;
  }
  nbase_clustered_t *mcl = (nbase_clustered_t *)(uintptr_t)cl;
  pthread_rwlock_rdlock(&mcl->rwlock);
  bool found = nbase_store_find_slot(cl->store, id, NULL);
  pthread_rwlock_unlock(&mcl->rwlock);
  return found;
}

nbase_status_t nbase_clustered_list_ids(
    const nbase_clustered_t *cl,
    uint64_t offset,
    uint32_t limit,
    uint64_t *out_ids,
    uint32_t *out_count,
    uint64_t *out_total) {
  if (!cl || !out_count || (limit > 0U && !out_ids)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  if (out_total) {
    *out_total = 0ULL;
  }

  nbase_clustered_t *mcl = (nbase_clustered_t *)(uintptr_t)cl;
  pthread_rwlock_rdlock(&mcl->rwlock);
  uint64_t active = cl->store->active_count;
  if (out_total) {
    *out_total = active;
  }
  if (active == 0ULL || limit == 0U || offset >= active) {
    pthread_rwlock_unlock(&mcl->rwlock);
    return NBASE_OK;
  }
  if (active > (uint64_t)SIZE_MAX / sizeof(uint64_t)) {
    pthread_rwlock_unlock(&mcl->rwlock);
    return NBASE_ERR_NOMEM;
  }
  uint64_t *all = malloc((size_t)active * sizeof(uint64_t));
  if (!all) {
    pthread_rwlock_unlock(&mcl->rwlock);
    return NBASE_ERR_NOMEM;
  }
  size_t n = 0U;
  for (uint32_t s = 0U; s < cl->store->slot_count && n < (size_t)active; s++) {
    if (cl->store->slot_flags[s] == NBASE_SLOT_ACTIVE) {
      all[n++] = cl->store->ids[s];
    }
  }
  pthread_rwlock_unlock(&mcl->rwlock);

  qsort(all, n, sizeof(uint64_t), cmp_cl_u64_asc);
  size_t start = (size_t)offset;
  if (start < n) {
    size_t rem = n - start;
    size_t take = ((size_t)limit < rem) ? (size_t)limit : rem;
    memcpy(out_ids, all + start, take * sizeof(uint64_t));
    *out_count = (uint32_t)take;
  }
  free(all);
  return NBASE_OK;
}

nbase_status_t nbase_clustered_recluster(nbase_clustered_t *cl, uint32_t num_clusters) {
  if (!cl) {
    return NBASE_ERR_INVALID_ARG;
  }
  pthread_rwlock_wrlock(&cl->rwlock);
  if (cl->store->active_count == 0ULL) {
    free_cluster_array(cl->clusters, cl->cluster_count);
    cl->clusters = NULL;
    cl->cluster_count = 0U;
    cl->cluster_cap = 0U;
    pthread_rwlock_unlock(&cl->rwlock);
    return NBASE_OK;
  }
  if (cl->store->active_count > (uint64_t)UINT32_MAX) {
    pthread_rwlock_unlock(&cl->rwlock);
    return NBASE_ERR_INVALID_ARG;
  }

  uint32_t active = (uint32_t)cl->store->active_count;
  uint32_t dim = cl->cfg.dim;
  uint32_t k_eff = num_clusters;
  if (k_eff == 0U) {
    uint64_t auto_k = (cl->store->active_count + (uint64_t)cl->cfg.target_cluster_size - 1ULL) /
                      (uint64_t)cl->cfg.target_cluster_size;
    k_eff = (auto_k > (uint64_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)auto_k;
  }
  if (k_eff < 1U) {
    k_eff = 1U;
  }
  if (k_eff > active) {
    k_eff = active;
  }
  if (k_eff > cl->cfg.max_clusters) {
    k_eff = cl->cfg.max_clusters;
  }

  float *vecs = malloc((size_t)active * (size_t)dim * sizeof(float));
  uint64_t *ids = malloc((size_t)active * sizeof(uint64_t));
  uint32_t *slots = malloc((size_t)active * sizeof(uint32_t));
  float *centroids = malloc((size_t)k_eff * (size_t)dim * sizeof(float));
  uint32_t *assign = malloc((size_t)active * sizeof(uint32_t));
  cl_cluster_t *new_clusters = calloc(k_eff, sizeof(cl_cluster_t));
  if (!vecs || !ids || !slots || !centroids || !assign || !new_clusters) {
    free(vecs);
    free(ids);
    free(slots);
    free(centroids);
    free(assign);
    free(new_clusters);
    pthread_rwlock_unlock(&cl->rwlock);
    return NBASE_ERR_NOMEM;
  }

  uint32_t n = 0U;
  for (uint32_t s = 0U; s < cl->store->slot_count && n < active; s++) {
    if (cl->store->slot_flags[s] != NBASE_SLOT_ACTIVE) {
      continue;
    }
    ids[n] = cl->store->ids[s];
    slots[n] = s;
    memcpy(vecs + (size_t)n * (size_t)dim, nbase_store_get_vector(cl->store, s), (size_t)dim * sizeof(float));
    n++;
  }

  nbase_status_t st = nbase_kmeans_train(
      vecs, n, dim, k_eff, cl->cfg.kmeans_max_iters, cl->cfg.seed, centroids, assign);
  free(vecs);
  if (st != NBASE_OK) {
    free(ids);
    free(slots);
    free(centroids);
    free(assign);
    free(new_clusters);
    pthread_rwlock_unlock(&cl->rwlock);
    return st;
  }

  for (uint32_t c = 0U; c < k_eff; c++) {
    new_clusters[c].centroid = malloc((size_t)dim * sizeof(float));
    if (!new_clusters[c].centroid) {
      free_cluster_array(new_clusters, k_eff);
      free(ids);
      free(slots);
      free(centroids);
      free(assign);
      pthread_rwlock_unlock(&cl->rwlock);
      return NBASE_ERR_NOMEM;
    }
    memcpy(new_clusters[c].centroid, centroids + (size_t)c * (size_t)dim, (size_t)dim * sizeof(float));
  }
  free(centroids);

  for (uint32_t i = 0U; i < n; i++) {
    uint32_t cid = assign[i];
    if (!ensure_cluster_member_cap(&new_clusters[cid], new_clusters[cid].count + 1U)) {
      free_cluster_array(new_clusters, k_eff);
      free(ids);
      free(slots);
      free(assign);
      pthread_rwlock_unlock(&cl->rwlock);
      return NBASE_ERR_NOMEM;
    }
    new_clusters[cid].member_ids[new_clusters[cid].count++] = ids[i];
  }

  if (!ensure_slot_cluster_cap(cl, cl->store->slot_count)) {
    free_cluster_array(new_clusters, k_eff);
    free(ids);
    free(slots);
    free(assign);
    pthread_rwlock_unlock(&cl->rwlock);
    return NBASE_ERR_NOMEM;
  }
  for (uint32_t i = 0U; i < n; i++) {
    cl->slot_cluster[slots[i]] = assign[i];
  }

  free(ids);
  free(slots);
  free(assign);

  free_cluster_array(cl->clusters, cl->cluster_count);
  cl->clusters = new_clusters;
  cl->cluster_count = k_eff;
  cl->cluster_cap = k_eff;
  pthread_rwlock_unlock(&cl->rwlock);
  return NBASE_OK;
}

nbase_status_t nbase_clustered_search_filtered(
    const nbase_clustered_t *cl,
    const float *query,
    uint32_t dim,
    uint32_t k,
    uint32_t nprobe,
    nbase_filter_fn filter,
    void *filter_ctx,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!cl || !query || !out_count || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  nbase_clustered_t *mcl = (nbase_clustered_t *)(uintptr_t)cl;
  pthread_rwlock_rdlock(&mcl->rwlock);

  if (dim != cl->cfg.dim) {
    pthread_rwlock_unlock(&mcl->rwlock);
    return NBASE_ERR_DIM_MISMATCH;
  }
  if (!nbase_vec_is_finite(query, dim)) {
    pthread_rwlock_unlock(&mcl->rwlock);
    return NBASE_ERR_INVALID_VECTOR;
  }
  if (k == 0U || cl->store->active_count == 0ULL || cl->cluster_count == 0U) {
    pthread_rwlock_unlock(&mcl->rwlock);
    return NBASE_OK;
  }

  nbase_dist_fn dfn = nbase_get_distance_fn(cl->cfg.metric);
  cl_probe_t *probes = malloc((size_t)cl->cluster_count * sizeof(cl_probe_t));
  if (!probes) {
    pthread_rwlock_unlock(&mcl->rwlock);
    return NBASE_ERR_NOMEM;
  }
  uint32_t non_empty = 0U;
  for (uint32_t c = 0U; c < cl->cluster_count; c++) {
    if (cl->clusters[c].count == 0U) {
      continue;
    }
    probes[non_empty].cluster_idx = c;
    probes[non_empty].dist = dfn(query, cl->clusters[c].centroid, dim);
    non_empty++;
  }
  if (non_empty == 0U) {
    free(probes);
    pthread_rwlock_unlock(&mcl->rwlock);
    return NBASE_OK;
  }

  qsort(probes, non_empty, sizeof(cl_probe_t), cmp_cl_probe_asc);
  uint32_t to_probe = (nprobe == 0U || nprobe > non_empty) ? non_empty : nprobe;

  nbase_result_t *heap_buf = malloc((size_t)k * sizeof(nbase_result_t));
  if (!heap_buf) {
    free(probes);
    pthread_rwlock_unlock(&mcl->rwlock);
    return NBASE_ERR_NOMEM;
  }
  cl_maxheap_t heap = {heap_buf, 0U, k};

  for (uint32_t pi = 0U; pi < to_probe; pi++) {
    const cl_cluster_t *c = &cl->clusters[probes[pi].cluster_idx];
    for (uint32_t mi = 0U; mi < c->count; mi++) {
      uint64_t mid = c->member_ids[mi];
      if (filter && !filter(mid, filter_ctx)) {
        continue;
      }
      uint32_t mslot = NBASE_INVALID_SLOT;
      if (!nbase_store_find_slot(cl->store, mid, &mslot)) {
        continue;
      }
      const float *mv = nbase_store_get_vector(cl->store, mslot);
      float dist = dfn(query, mv, dim);
      cl_heap_push_bounded(&heap, mid, dist);
    }
  }
  free(probes);
  pthread_rwlock_unlock(&mcl->rwlock);

  if (heap.len > 1U) {
    qsort(heap.data, heap.len, sizeof(nbase_result_t), nbase_cmp_result_asc);
  }
  if (heap.len > 0U) {
    memcpy(out_results, heap.data, (size_t)heap.len * sizeof(nbase_result_t));
  }
  *out_count = heap.len;
  free(heap_buf);
  return NBASE_OK;
}

nbase_status_t nbase_clustered_search(
    const nbase_clustered_t *cl,
    const float *query,
    uint32_t dim,
    uint32_t k,
    uint32_t nprobe,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  return nbase_clustered_search_filtered(
      cl, query, dim, k, nprobe, NULL, NULL, out_results, out_count);
}

nbase_status_t nbase_clustered_stats(
    const nbase_clustered_t *cl,
    nbase_clustered_stats_t *out_stats) {
  if (!cl || !out_stats) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_clustered_t *mcl = (nbase_clustered_t *)(uintptr_t)cl;
  pthread_rwlock_rdlock(&mcl->rwlock);

  memset(out_stats, 0, sizeof(*out_stats));
  out_stats->dim = cl->cfg.dim;
  out_stats->metric = cl->cfg.metric;
  out_stats->target_cluster_size = cl->cfg.target_cluster_size;
  out_stats->max_clusters = cl->cfg.max_clusters;
  uint32_t non_empty = 0U;
  uint64_t mem = (uint64_t)sizeof(*cl) +
                 nbase_store_memory_bytes(cl->store) +
                 (uint64_t)cl->cluster_cap * (uint64_t)sizeof(cl_cluster_t) +
                 (uint64_t)cl->slot_cap * (uint64_t)sizeof(uint32_t);
  for (uint32_t c = 0U; c < cl->cluster_count; c++) {
    if (cl->clusters[c].count > 0U) {
      non_empty++;
    }
    if (cl->clusters[c].centroid != NULL) {
      mem += (uint64_t)cl->cfg.dim * (uint64_t)sizeof(float);
    }
    mem += (uint64_t)cl->clusters[c].cap * (uint64_t)sizeof(uint64_t);
  }
  out_stats->cluster_count = non_empty;
  out_stats->active_vectors = cl->store->active_count;
  out_stats->memory_bytes = mem;

  pthread_rwlock_unlock(&mcl->rwlock);
  return NBASE_OK;
}

nbase_status_t nbase_clustered_save(const nbase_clustered_t *cl, const char *file_path) {
  if (!cl || !file_path || file_path[0] == '\0') {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_clustered_t *mcl = (nbase_clustered_t *)(uintptr_t)cl;
  pthread_rwlock_rdlock(&mcl->rwlock);

  uint32_t dim = cl->cfg.dim;
  size_t vec_bytes = (size_t)dim * sizeof(float);
  size_t payload_bytes = 0U;
  uint32_t non_empty_clusters = 0U;

  for (uint32_t c = 0U; c < cl->cluster_count; c++) {
    if (cl->clusters[c].count == 0U) {
      continue;
    }
    non_empty_clusters++;
    size_t per_cluster = vec_bytes + sizeof(uint32_t) +
                         (size_t)cl->clusters[c].count * (sizeof(uint64_t) + vec_bytes);
    if (payload_bytes > SIZE_MAX - per_cluster) {
      pthread_rwlock_unlock(&mcl->rwlock);
      return NBASE_ERR_NOMEM;
    }
    payload_bytes += per_cluster;
  }

  uint8_t *buf = NULL;
  if (payload_bytes > 0U) {
    buf = malloc(payload_bytes);
    if (!buf) {
      pthread_rwlock_unlock(&mcl->rwlock);
      return NBASE_ERR_NOMEM;
    }
    size_t off = 0U;
    for (uint32_t c = 0U; c < cl->cluster_count; c++) {
      const cl_cluster_t *clst = &cl->clusters[c];
      if (clst->count == 0U) {
        continue;
      }
      memcpy(buf + off, clst->centroid, vec_bytes);
      off += vec_bytes;
      memcpy(buf + off, &clst->count, sizeof(uint32_t));
      off += sizeof(uint32_t);
      for (uint32_t mi = 0U; mi < clst->count; mi++) {
        uint64_t mid = clst->member_ids[mi];
        uint32_t mslot = NBASE_INVALID_SLOT;
        if (!nbase_store_find_slot(cl->store, mid, &mslot)) {
          free(buf);
          pthread_rwlock_unlock(&mcl->rwlock);
          return NBASE_ERR_CORRUPT;
        }
        const float *mv = nbase_store_get_vector(cl->store, mslot);
        memcpy(buf + off, &mid, sizeof(uint64_t));
        off += sizeof(uint64_t);
        memcpy(buf + off, mv, vec_bytes);
        off += vec_bytes;
      }
    }
  }

  nbcl_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBCL_MAGIC, 8);
  hdr.version = NBASE_NBCL_VERSION;
  hdr.dim = cl->cfg.dim;
  hdr.metric = (uint32_t)cl->cfg.metric;
  hdr.target_cluster_size = cl->cfg.target_cluster_size;
  hdr.max_clusters = cl->cfg.max_clusters;
  hdr.distance_threshold = cl->cfg.distance_threshold;
  hdr.kmeans_max_iters = cl->cfg.kmeans_max_iters;
  hdr.cluster_count = non_empty_clusters;
  hdr.seed = cl->cfg.seed;
  hdr.active_vectors = cl->store->active_count;
  hdr.payload_bytes = (uint64_t)payload_bytes;
  hdr.payload_crc32 = (payload_bytes > 0U) ? nbase_crc32(buf, payload_bytes) : 0U;
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nbcl_header_t, header_crc32));

  pthread_rwlock_unlock(&mcl->rwlock);

  nbase_io_chunk_t chunk = {buf, payload_bytes};
  nbase_status_t st = nbase_write_file_atomic(
      file_path, &hdr, sizeof(hdr), &chunk, (payload_bytes > 0U) ? 1U : 0U);
  free(buf);
  return st;
}

nbase_status_t nbase_clustered_load(const char *file_path, nbase_clustered_t **out_cl) {
  if (!file_path || file_path[0] == '\0' || !out_cl) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_cl = NULL;

  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    return (errno == ENOENT) ? NBASE_ERR_NOT_FOUND : NBASE_ERR_IO;
  }

  struct stat st_buf;
  if (fstat(fd, &st_buf) != 0) {
    close(fd);
    return NBASE_ERR_IO;
  }
  if (st_buf.st_size < (off_t)sizeof(nbcl_header_t)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;

  nbcl_header_t hdr;
  nbase_status_t st = nbase_read_exact(fd, &hdr, sizeof(hdr));
  if (st != NBASE_OK) {
    close(fd);
    return st;
  }

  if (memcmp(hdr.magic, NBASE_NBCL_MAGIC, 8) != 0 || hdr.version != NBASE_NBCL_VERSION) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.header_crc32 != nbase_crc32(&hdr, offsetof(nbcl_header_t, header_crc32))) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.dim == 0U || hdr.dim > NBASE_MAX_DIM || !nbase_metric_is_valid((nbase_metric_t)hdr.metric) ||
      hdr.target_cluster_size == 0U || hdr.max_clusters == 0U ||
      hdr.cluster_count > hdr.max_clusters ||
      !isfinite(hdr.distance_threshold) || hdr.distance_threshold < 0.0f ||
      hdr.payload_bytes > (uint64_t)SIZE_MAX ||
      file_size != (uint64_t)sizeof(nbcl_header_t) + hdr.payload_bytes) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint8_t *payload = NULL;
  if (hdr.payload_bytes > 0ULL) {
    payload = malloc((size_t)hdr.payload_bytes);
    if (!payload) {
      close(fd);
      return NBASE_ERR_NOMEM;
    }
    st = nbase_read_exact(fd, payload, (size_t)hdr.payload_bytes);
  }
  close(fd);
  if (st != NBASE_OK) {
    free(payload);
    return st;
  }

  uint32_t crc = (hdr.payload_bytes > 0ULL) ? nbase_crc32(payload, (size_t)hdr.payload_bytes) : 0U;
  if (crc != hdr.payload_crc32) {
    free(payload);
    return NBASE_ERR_CORRUPT;
  }

  nbase_clustered_config_t cfg;
  cfg.dim = hdr.dim;
  cfg.metric = (nbase_metric_t)hdr.metric;
  cfg.target_cluster_size = hdr.target_cluster_size;
  cfg.max_clusters = hdr.max_clusters;
  cfg.distance_threshold = hdr.distance_threshold;
  cfg.kmeans_max_iters = hdr.kmeans_max_iters;
  cfg.seed = hdr.seed;

  nbase_clustered_t *cl = NULL;
  st = nbase_clustered_create(&cfg, &cl);
  if (st != NBASE_OK) {
    free(payload);
    return st;
  }

  size_t vec_bytes = (size_t)hdr.dim * sizeof(float);
  size_t off = 0U;
  size_t total_len = (size_t)hdr.payload_bytes;
  uint64_t sum_members = 0ULL;

  for (uint32_t c = 0U; c < hdr.cluster_count; c++) {
    if (off + vec_bytes + sizeof(uint32_t) > total_len) {
      free(payload);
      nbase_clustered_destroy(cl);
      return NBASE_ERR_CORRUPT;
    }
    uint32_t cidx = 0U;
    st = append_empty_cluster(cl, &cidx);
    if (st != NBASE_OK) {
      free(payload);
      nbase_clustered_destroy(cl);
      return st;
    }
    memcpy(cl->clusters[cidx].centroid, payload + off, vec_bytes);
    off += vec_bytes;
    if (!nbase_vec_is_finite(cl->clusters[cidx].centroid, hdr.dim)) {
      free(payload);
      nbase_clustered_destroy(cl);
      return NBASE_ERR_CORRUPT;
    }

    uint32_t mcnt = 0U;
    memcpy(&mcnt, payload + off, sizeof(uint32_t));
    off += sizeof(uint32_t);
    size_t entry_sz = sizeof(uint64_t) + vec_bytes;
    if ((uint64_t)mcnt > (uint64_t)(total_len - off) / (uint64_t)entry_sz) {
      free(payload);
      nbase_clustered_destroy(cl);
      return NBASE_ERR_CORRUPT;
    }

    if (mcnt > 0U && !ensure_cluster_member_cap(&cl->clusters[cidx], mcnt)) {
      free(payload);
      nbase_clustered_destroy(cl);
      return NBASE_ERR_NOMEM;
    }

    for (uint32_t mi = 0U; mi < mcnt; mi++) {
      uint64_t mid = 0ULL;
      memcpy(&mid, payload + off, sizeof(uint64_t));
      off += sizeof(uint64_t);

      float *tmp_v = malloc(vec_bytes);
      if (!tmp_v) {
        free(payload);
        nbase_clustered_destroy(cl);
        return NBASE_ERR_NOMEM;
      }
      memcpy(tmp_v, payload + off, vec_bytes);
      off += vec_bytes;

      if (!nbase_vec_is_finite(tmp_v, hdr.dim)) {
        free(tmp_v);
        free(payload);
        nbase_clustered_destroy(cl);
        return NBASE_ERR_CORRUPT;
      }

      uint32_t slot = NBASE_INVALID_SLOT;
      st = nbase_store_add(cl->store, mid, tmp_v, &slot);
      free(tmp_v);
      if (st == NBASE_ERR_ALREADY_EXISTS) {
        free(payload);
        nbase_clustered_destroy(cl);
        return NBASE_ERR_CORRUPT;
      }
      if (st != NBASE_OK) {
        free(payload);
        nbase_clustered_destroy(cl);
        return st;
      }
      if (!ensure_slot_cluster_cap(cl, slot + 1U)) {
        free(payload);
        nbase_clustered_destroy(cl);
        return NBASE_ERR_NOMEM;
      }
      cl->clusters[cidx].member_ids[cl->clusters[cidx].count++] = mid;
      cl->slot_cluster[slot] = cidx;
    }
    sum_members += mcnt;
  }

  free(payload);
  if (off != total_len || sum_members != hdr.active_vectors) {
    nbase_clustered_destroy(cl);
    return NBASE_ERR_CORRUPT;
  }

  *out_cl = cl;
  return NBASE_OK;
}
