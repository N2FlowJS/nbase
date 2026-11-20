#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define NBASE_HNSW_MAX_LEVEL 32

typedef struct hnsw_visited {
  uint32_t *marks;
  uint32_t cap;
  uint32_t gen;
} hnsw_visited_t;

static inline const float *hnsw_slot_vec(const nbase_store_t *store, uint32_t slot) {
  return store->vectors + (size_t)slot * (size_t)store->dim;
}

static int cmp_heap_item_asc(const void *pa, const void *pb) {
  const nbase_heap_item_t *a = (const nbase_heap_item_t *)pa;
  const nbase_heap_item_t *b = (const nbase_heap_item_t *)pb;
  if (a->dist < b->dist) {
    return -1;
  }
  if (a->dist > b->dist) {
    return 1;
  }
  if (a->slot < b->slot) {
    return -1;
  }
  if (a->slot > b->slot) {
    return 1;
  }
  return 0;
}

static bool visited_init(hnsw_visited_t *v, uint32_t cap) {
  v->cap = cap > 0U ? cap : 1U;
  v->marks = calloc(v->cap, sizeof(uint32_t));
  if (!v->marks) {
    v->cap = 0U;
    return false;
  }
  v->gen = 0U;
  return true;
}

static void visited_free(hnsw_visited_t *v) {
  free(v->marks);
  v->marks = NULL;
  v->cap = 0U;
  v->gen = 0U;
}

static void visited_next_gen(hnsw_visited_t *v) {
  if (v->gen == UINT32_MAX) {
    memset(v->marks, 0, (size_t)v->cap * sizeof(uint32_t));
    v->gen = 1U;
  } else {
    v->gen++;
  }
}

static inline bool visited_test_and_set(hnsw_visited_t *v, uint32_t slot) {
  if (v->marks[slot] == v->gen) {
    return true;
  }
  v->marks[slot] = v->gen;
  return false;
}

static bool adj_contains(const nbase_adj_t *list, uint32_t target) {
  for (uint32_t i = 0U; i < list->len; i++) {
    if (list->neighbors[i] == target) {
      return true;
    }
  }
  return false;
}

static bool adj_push(nbase_adj_t *list, uint32_t target) {
  if (adj_contains(list, target)) {
    return true;
  }
  if (list->len == list->cap) {
    uint32_t new_cap = list->cap == 0U ? 8U : list->cap * 2U;
    uint32_t *next = realloc(list->neighbors, (size_t)new_cap * sizeof(uint32_t));
    if (!next) {
      return false;
    }
    list->neighbors = next;
    list->cap = new_cap;
  }
  list->neighbors[list->len++] = target;
  return true;
}

static int32_t random_level(nbase_hnsw_t *hnsw) {
  double u = nbase_rng_next_double(&hnsw->rng_state);
  double raw = floor(-log(u) * hnsw->ml);
  if (raw < 0.0) {
    return 0;
  }
  if (raw > (double)NBASE_HNSW_MAX_LEVEL) {
    return NBASE_HNSW_MAX_LEVEL;
  }
  return (int32_t)raw;
}

static nbase_status_t hnsw_ensure_node_capacity(nbase_hnsw_t *hnsw, uint32_t needed) {
  if (needed <= hnsw->node_capacity) {
    return NBASE_OK;
  }
  uint32_t new_cap = hnsw->node_capacity == 0U ? 64U : hnsw->node_capacity;
  while (new_cap < needed) {
    if (new_cap > UINT32_MAX / 2U) {
      new_cap = needed;
      break;
    }
    new_cap *= 2U;
  }

  int32_t *new_levels = realloc(hnsw->levels, (size_t)new_cap * sizeof(int32_t));
  if (!new_levels) {
    return NBASE_ERR_NOMEM;
  }
  hnsw->levels = new_levels;

  uint8_t *new_states = realloc(hnsw->states, (size_t)new_cap * sizeof(uint8_t));
  if (!new_states) {
    return NBASE_ERR_NOMEM;
  }
  hnsw->states = new_states;

  nbase_adj_t **new_adj = realloc(hnsw->adj, (size_t)new_cap * sizeof(nbase_adj_t *));
  if (!new_adj) {
    return NBASE_ERR_NOMEM;
  }
  hnsw->adj = new_adj;

  for (uint32_t i = hnsw->node_capacity; i < new_cap; i++) {
    hnsw->levels[i] = -1;
    hnsw->states[i] = NBASE_SLOT_EMPTY;
    hnsw->adj[i] = NULL;
  }
  hnsw->node_capacity = new_cap;
  return NBASE_OK;
}

static nbase_status_t search_layer(
    const nbase_hnsw_t *hnsw,
    const nbase_store_t *store,
    const float *query_vec,
    uint32_t ep_slot,
    uint32_t ef,
    int32_t level,
    nbase_dist_fn dist_fn,
    bool for_query,
    nbase_filter_fn filter,
    void *filter_ctx,
    hnsw_visited_t *vis,
    nbase_heap_item_t **out_items,
    size_t *out_count) {
  *out_items = NULL;
  *out_count = 0;

  visited_next_gen(vis);

  nbase_heap_t candidates;
  nbase_heap_t results;
  nbase_heap_init(&candidates, false);
  nbase_heap_init(&results, true);

  float ep_dist = dist_fn(query_vec, hnsw_slot_vec(store, ep_slot), store->dim);
  (void)visited_test_and_set(vis, ep_slot);

  if (!nbase_heap_push(&candidates, (nbase_heap_item_t){ep_slot, ep_dist})) {
    nbase_heap_free(&candidates);
    nbase_heap_free(&results);
    return NBASE_ERR_NOMEM;
  }

  bool ep_eligible = (hnsw->states[ep_slot] == NBASE_SLOT_ACTIVE);
  if (ep_eligible && for_query && filter != NULL) {
    ep_eligible = filter(store->ids[ep_slot], filter_ctx);
  }
  if (ep_eligible) {
    if (!nbase_heap_push(&results, (nbase_heap_item_t){ep_slot, ep_dist})) {
      nbase_heap_free(&candidates);
      nbase_heap_free(&results);
      return NBASE_ERR_NOMEM;
    }
  }

  while (candidates.len > 0) {
    nbase_heap_item_t curr = nbase_heap_pop(&candidates);
    float worst = (results.len > 0) ? results.data[0].dist : INFINITY;
    if (results.len >= ef && curr.dist > worst) {
      break;
    }

    if (level <= hnsw->levels[curr.slot] && hnsw->adj[curr.slot] != NULL) {
      const nbase_adj_t *list = &hnsw->adj[curr.slot][level];
      for (uint32_t i = 0U; i < list->len; i++) {
        uint32_t nb = list->neighbors[i];
        if (nb >= hnsw->node_count || visited_test_and_set(vis, nb)) {
          continue;
        }
        if (hnsw->states[nb] == NBASE_SLOT_EMPTY) {
          continue;
        }

        float d = dist_fn(query_vec, hnsw_slot_vec(store, nb), store->dim);
        worst = (results.len > 0) ? results.data[0].dist : INFINITY;
        if (results.len < ef || d < worst) {
          if (!nbase_heap_push(&candidates, (nbase_heap_item_t){nb, d})) {
            nbase_heap_free(&candidates);
            nbase_heap_free(&results);
            return NBASE_ERR_NOMEM;
          }

          bool nb_eligible = (hnsw->states[nb] == NBASE_SLOT_ACTIVE);
          if (nb_eligible && for_query && filter != NULL) {
            nb_eligible = filter(store->ids[nb], filter_ctx);
          }
          if (nb_eligible) {
            if (!nbase_heap_push(&results, (nbase_heap_item_t){nb, d})) {
              nbase_heap_free(&candidates);
              nbase_heap_free(&results);
              return NBASE_ERR_NOMEM;
            }
            if (results.len > ef) {
              (void)nbase_heap_pop(&results);
            }
          }
        }
      }
    }
  }

  nbase_heap_free(&candidates);

  if (results.len > 0) {
    qsort(results.data, results.len, sizeof(nbase_heap_item_t), cmp_heap_item_asc);
  }
  *out_items = results.data;
  *out_count = results.len;
  return NBASE_OK;
}

/* Algorithm 4 neighbor selection heuristic (no keepPrunedConnections refill). */
static void select_neighbors_heuristic(
    const nbase_store_t *store,
    nbase_dist_fn dist_fn,
    uint32_t self_slot,
    nbase_heap_item_t *candidates,
    size_t cand_count,
    uint32_t max_conn,
    uint32_t *out_neighbors,
    uint32_t *out_count) {
  *out_count = 0U;
  if (cand_count == 0U || max_conn == 0U) {
    return;
  }
  qsort(candidates, cand_count, sizeof(nbase_heap_item_t), cmp_heap_item_asc);

  for (size_t i = 0; i < cand_count && *out_count < max_conn; i++) {
    nbase_heap_item_t e = candidates[i];
    if (e.slot == self_slot) {
      continue;
    }
    const float *e_vec = hnsw_slot_vec(store, e.slot);
    bool keep = true;
    for (uint32_t j = 0U; j < *out_count; j++) {
      uint32_t r_slot = out_neighbors[j];
      if (r_slot == e.slot) {
        keep = false;
        break;
      }
      float dist_er = dist_fn(e_vec, hnsw_slot_vec(store, r_slot), store->dim);
      if (dist_er < e.dist) {
        keep = false;
        break;
      }
    }
    if (keep) {
      out_neighbors[(*out_count)++] = e.slot;
    }
  }
}

static nbase_status_t shrink_connections(
    nbase_hnsw_t *hnsw,
    const nbase_store_t *store,
    nbase_dist_fn dist_fn,
    uint32_t node_slot,
    int32_t level,
    uint32_t max_conn) {
  nbase_adj_t *list = &hnsw->adj[node_slot][level];
  if (list->len <= max_conn) {
    return NBASE_OK;
  }

  nbase_heap_item_t *cands = malloc((size_t)list->len * sizeof(nbase_heap_item_t));
  uint32_t *selected = malloc((size_t)max_conn * sizeof(uint32_t));
  if (!cands || !selected) {
    free(cands);
    free(selected);
    return NBASE_ERR_NOMEM;
  }

  const float *base_vec = hnsw_slot_vec(store, node_slot);
  size_t cand_count = 0;
  for (uint32_t i = 0U; i < list->len; i++) {
    uint32_t nb = list->neighbors[i];
    if (nb == node_slot || hnsw->states[nb] == NBASE_SLOT_EMPTY) {
      continue;
    }
    float d = dist_fn(base_vec, hnsw_slot_vec(store, nb), store->dim);
    cands[cand_count++] = (nbase_heap_item_t){nb, d};
  }

  uint32_t kept = 0U;
  select_neighbors_heuristic(store, dist_fn, node_slot, cands, cand_count, max_conn, selected, &kept);
  memcpy(list->neighbors, selected, (size_t)kept * sizeof(uint32_t));
  list->len = kept;

  free(cands);
  free(selected);
  return NBASE_OK;
}

nbase_status_t nbase_hnsw_create(uint32_t m, uint32_t ef_construction, uint32_t ef_search, uint64_t seed, nbase_hnsw_t **out_hnsw) {
  if (!out_hnsw || m < 2U || ef_construction == 0U || ef_search == 0U) {
    return NBASE_ERR_INVALID_ARG;
  }
  nbase_hnsw_t *hnsw = calloc(1, sizeof(*hnsw));
  if (!hnsw) {
    return NBASE_ERR_NOMEM;
  }
  hnsw->m = m;
  hnsw->m_max = m;
  hnsw->m_max0 = m * 2U;
  hnsw->ef_construction = ef_construction;
  hnsw->ef_search = ef_search;
  hnsw->ml = 1.0 / log((double)m);
  hnsw->max_level = -1;
  hnsw->entry_point = NBASE_INVALID_SLOT;
  hnsw->rng_state = seed != 0ULL ? seed : 0x4E42415345534545ULL;
  *out_hnsw = hnsw;
  return NBASE_OK;
}

void nbase_hnsw_destroy(nbase_hnsw_t *hnsw) {
  if (!hnsw) {
    return;
  }
  for (uint32_t i = 0U; i < hnsw->node_count; i++) {
    if (hnsw->adj[i] != NULL) {
      int32_t lvl = hnsw->levels[i];
      for (int32_t lc = 0; lc <= lvl; lc++) {
        free(hnsw->adj[i][lc].neighbors);
      }
      free(hnsw->adj[i]);
    }
  }
  free(hnsw->adj);
  free(hnsw->levels);
  free(hnsw->states);
  free(hnsw);
}

nbase_status_t nbase_hnsw_insert(nbase_hnsw_t *hnsw, const nbase_store_t *store, uint32_t slot) {
  if (!hnsw || !store || slot >= store->slot_count) {
    return NBASE_ERR_INVALID_ARG;
  }

  nbase_status_t st = hnsw_ensure_node_capacity(hnsw, slot + 1U);
  if (st != NBASE_OK) {
    return st;
  }

  int32_t level = random_level(hnsw);
  nbase_adj_t *node_adj = calloc((size_t)(level + 1), sizeof(nbase_adj_t));
  if (!node_adj) {
    return NBASE_ERR_NOMEM;
  }

  if (slot + 1U > hnsw->node_count) {
    hnsw->node_count = slot + 1U;
  }
  hnsw->levels[slot] = level;
  hnsw->states[slot] = NBASE_SLOT_ACTIVE;
  hnsw->adj[slot] = node_adj;
  hnsw->active_nodes++;

  if (hnsw->entry_point == NBASE_INVALID_SLOT) {
    hnsw->entry_point = slot;
    hnsw->max_level = level;
    return NBASE_OK;
  }

  nbase_dist_fn dist_fn = nbase_get_distance_fn(store->metric);
  const float *query_vec = hnsw_slot_vec(store, slot);
  uint32_t curr = hnsw->entry_point;
  float curr_dist = dist_fn(query_vec, hnsw_slot_vec(store, curr), store->dim);

  for (int32_t lc = hnsw->max_level; lc > level; lc--) {
    bool changed = true;
    while (changed) {
      changed = false;
      if (lc <= hnsw->levels[curr] && hnsw->adj[curr] != NULL) {
        const nbase_adj_t *list = &hnsw->adj[curr][lc];
        for (uint32_t i = 0U; i < list->len; i++) {
          uint32_t nb = list->neighbors[i];
          if (nb >= hnsw->node_count || hnsw->states[nb] == NBASE_SLOT_EMPTY) {
            continue;
          }
          float d = dist_fn(query_vec, hnsw_slot_vec(store, nb), store->dim);
          if (d < curr_dist) {
            curr_dist = d;
            curr = nb;
            changed = true;
          }
        }
      }
    }
  }

  hnsw_visited_t vis;
  if (!visited_init(&vis, hnsw->node_count)) {
    return NBASE_ERR_NOMEM;
  }

  uint32_t *selected = malloc((size_t)hnsw->m_max0 * sizeof(uint32_t));
  if (!selected) {
    visited_free(&vis);
    return NBASE_ERR_NOMEM;
  }

  int32_t start_lc = (level < hnsw->max_level) ? level : hnsw->max_level;
  for (int32_t lc = start_lc; lc >= 0; lc--) {
    nbase_heap_item_t *cands = NULL;
    size_t cand_count = 0;
    st = search_layer(hnsw, store, query_vec, curr, hnsw->ef_construction, lc, dist_fn, false, NULL, NULL, &vis, &cands, &cand_count);
    if (st != NBASE_OK) {
      free(cands);
      free(selected);
      visited_free(&vis);
      return st;
    }

    uint32_t max_m = (lc == 0) ? hnsw->m_max0 : hnsw->m_max;
    uint32_t sel_count = 0U;
    select_neighbors_heuristic(store, dist_fn, slot, cands, cand_count, max_m, selected, &sel_count);

    for (uint32_t i = 0U; i < sel_count; i++) {
      uint32_t nb = selected[i];
      if (!adj_push(&hnsw->adj[slot][lc], nb)) {
        free(cands);
        free(selected);
        visited_free(&vis);
        return NBASE_ERR_NOMEM;
      }
      if (lc <= hnsw->levels[nb] && hnsw->adj[nb] != NULL) {
        if (!adj_push(&hnsw->adj[nb][lc], slot)) {
          free(cands);
          free(selected);
          visited_free(&vis);
          return NBASE_ERR_NOMEM;
        }
        st = shrink_connections(hnsw, store, dist_fn, nb, lc, max_m);
        if (st != NBASE_OK) {
          free(cands);
          free(selected);
          visited_free(&vis);
          return st;
        }
      }
    }

    if (cand_count > 0) {
      curr = cands[0].slot;
    }
    free(cands);
  }

  free(selected);
  visited_free(&vis);

  if (level > hnsw->max_level) {
    hnsw->entry_point = slot;
    hnsw->max_level = level;
  }
  return NBASE_OK;
}

#define NBASE_HNSW_STRIPE_COUNT 256U

typedef struct hnsw_mt_shared {
  nbase_hnsw_t *hnsw;
  const nbase_store_t *store;
  const uint32_t *slots;
  uint32_t start_idx;
  uint32_t count;
  nbase_dist_fn dist_fn;
  pthread_mutex_t work_lock;
  uint32_t next_idx;
  pthread_mutex_t global_lock;
  pthread_mutex_t node_locks[NBASE_HNSW_STRIPE_COUNT];
  nbase_status_t status;
} hnsw_mt_shared_t;

static nbase_status_t copy_neighbors_locked(
    hnsw_mt_shared_t *ctx,
    uint32_t node_slot,
    int32_t level,
    uint32_t **scratch,
    uint32_t *scratch_cap,
    uint32_t *out_len) {
  *out_len = 0U;
  const nbase_hnsw_t *hnsw = ctx->hnsw;
  if (level > hnsw->levels[node_slot] || hnsw->adj[node_slot] == NULL) {
    return NBASE_OK;
  }
  pthread_mutex_t *lk = &ctx->node_locks[node_slot & (NBASE_HNSW_STRIPE_COUNT - 1U)];
  pthread_mutex_lock(lk);
  const nbase_adj_t *list = &hnsw->adj[node_slot][level];
  uint32_t len = list->len;
  if (len > *scratch_cap) {
    uint32_t new_cap = *scratch_cap == 0U ? 32U : *scratch_cap;
    while (new_cap < len) {
      new_cap *= 2U;
    }
    uint32_t *next = realloc(*scratch, (size_t)new_cap * sizeof(uint32_t));
    if (!next) {
      pthread_mutex_unlock(lk);
      return NBASE_ERR_NOMEM;
    }
    *scratch = next;
    *scratch_cap = new_cap;
  }
  if (len > 0U) {
    memcpy(*scratch, list->neighbors, (size_t)len * sizeof(uint32_t));
  }
  pthread_mutex_unlock(lk);
  *out_len = len;
  return NBASE_OK;
}

static nbase_status_t search_layer_mt(
    hnsw_mt_shared_t *ctx,
    const float *query_vec,
    uint32_t ep_slot,
    uint32_t ef,
    int32_t level,
    hnsw_visited_t *vis,
    uint32_t **nb_scratch,
    uint32_t *nb_scratch_cap,
    nbase_heap_item_t **out_items,
    size_t *out_count) {
  *out_items = NULL;
  *out_count = 0;

  const nbase_hnsw_t *hnsw = ctx->hnsw;
  const nbase_store_t *store = ctx->store;
  nbase_dist_fn dist_fn = ctx->dist_fn;

  visited_next_gen(vis);

  nbase_heap_t candidates;
  nbase_heap_t results;
  nbase_heap_init(&candidates, false);
  nbase_heap_init(&results, true);

  float ep_dist = dist_fn(query_vec, hnsw_slot_vec(store, ep_slot), store->dim);
  (void)visited_test_and_set(vis, ep_slot);

  if (!nbase_heap_push(&candidates, (nbase_heap_item_t){ep_slot, ep_dist})) {
    nbase_heap_free(&candidates);
    nbase_heap_free(&results);
    return NBASE_ERR_NOMEM;
  }

  if (hnsw->states[ep_slot] == NBASE_SLOT_ACTIVE) {
    if (!nbase_heap_push(&results, (nbase_heap_item_t){ep_slot, ep_dist})) {
      nbase_heap_free(&candidates);
      nbase_heap_free(&results);
      return NBASE_ERR_NOMEM;
    }
  }

  while (candidates.len > 0) {
    nbase_heap_item_t curr = nbase_heap_pop(&candidates);
    float worst = (results.len > 0) ? results.data[0].dist : INFINITY;
    if (results.len >= ef && curr.dist > worst) {
      break;
    }

    uint32_t nb_len = 0U;
    nbase_status_t st = copy_neighbors_locked(ctx, curr.slot, level, nb_scratch, nb_scratch_cap, &nb_len);
    if (st != NBASE_OK) {
      nbase_heap_free(&candidates);
      nbase_heap_free(&results);
      return st;
    }

    for (uint32_t i = 0U; i < nb_len; i++) {
      uint32_t nb = (*nb_scratch)[i];
      if (nb >= hnsw->node_count || visited_test_and_set(vis, nb)) {
        continue;
      }
      if (hnsw->states[nb] == NBASE_SLOT_EMPTY) {
        continue;
      }

      float d = dist_fn(query_vec, hnsw_slot_vec(store, nb), store->dim);
      worst = (results.len > 0) ? results.data[0].dist : INFINITY;
      if (results.len < ef || d < worst) {
        if (!nbase_heap_push(&candidates, (nbase_heap_item_t){nb, d})) {
          nbase_heap_free(&candidates);
          nbase_heap_free(&results);
          return NBASE_ERR_NOMEM;
        }
        if (hnsw->states[nb] == NBASE_SLOT_ACTIVE) {
          if (!nbase_heap_push(&results, (nbase_heap_item_t){nb, d})) {
            nbase_heap_free(&candidates);
            nbase_heap_free(&results);
            return NBASE_ERR_NOMEM;
          }
          if (results.len > ef) {
            (void)nbase_heap_pop(&results);
          }
        }
      }
    }
  }

  nbase_heap_free(&candidates);
  if (results.len > 0) {
    qsort(results.data, results.len, sizeof(nbase_heap_item_t), cmp_heap_item_asc);
  }
  *out_items = results.data;
  *out_count = results.len;
  return NBASE_OK;
}

static void *hnsw_mt_worker(void *arg) {
  hnsw_mt_shared_t *ctx = (hnsw_mt_shared_t *)arg;
  nbase_hnsw_t *hnsw = ctx->hnsw;
  const nbase_store_t *store = ctx->store;

  hnsw_visited_t vis;
  if (!visited_init(&vis, hnsw->node_count)) {
    pthread_mutex_lock(&ctx->work_lock);
    if (ctx->status == NBASE_OK) {
      ctx->status = NBASE_ERR_NOMEM;
    }
    pthread_mutex_unlock(&ctx->work_lock);
    return NULL;
  }

  uint32_t *selected = malloc((size_t)hnsw->m_max0 * sizeof(uint32_t));
  uint32_t *nb_scratch = NULL;
  uint32_t nb_scratch_cap = 0U;
  if (!selected) {
    visited_free(&vis);
    pthread_mutex_lock(&ctx->work_lock);
    if (ctx->status == NBASE_OK) {
      ctx->status = NBASE_ERR_NOMEM;
    }
    pthread_mutex_unlock(&ctx->work_lock);
    return NULL;
  }

  while (true) {
    pthread_mutex_lock(&ctx->work_lock);
    if (ctx->status != NBASE_OK || ctx->next_idx >= ctx->count) {
      pthread_mutex_unlock(&ctx->work_lock);
      break;
    }
    uint32_t idx = ctx->next_idx++;
    pthread_mutex_unlock(&ctx->work_lock);

    uint32_t slot = ctx->slots[idx];
    int32_t level = hnsw->levels[slot];
    const float *query_vec = hnsw_slot_vec(store, slot);

    pthread_mutex_lock(&ctx->global_lock);
    uint32_t curr = hnsw->entry_point;
    int32_t max_level = hnsw->max_level;
    pthread_mutex_unlock(&ctx->global_lock);

    float curr_dist = ctx->dist_fn(query_vec, hnsw_slot_vec(store, curr), store->dim);
    nbase_status_t st = NBASE_OK;

    for (int32_t lc = max_level; lc > level && st == NBASE_OK; lc--) {
      bool changed = true;
      while (changed) {
        changed = false;
        uint32_t nb_len = 0U;
        st = copy_neighbors_locked(ctx, curr, lc, &nb_scratch, &nb_scratch_cap, &nb_len);
        if (st != NBASE_OK) {
          break;
        }
        for (uint32_t i = 0U; i < nb_len; i++) {
          uint32_t nb = nb_scratch[i];
          if (nb >= hnsw->node_count || hnsw->states[nb] == NBASE_SLOT_EMPTY) {
            continue;
          }
          float d = ctx->dist_fn(query_vec, hnsw_slot_vec(store, nb), store->dim);
          if (d < curr_dist) {
            curr_dist = d;
            curr = nb;
            changed = true;
          }
        }
      }
    }

    int32_t start_lc = (level < max_level) ? level : max_level;
    for (int32_t lc = start_lc; lc >= 0 && st == NBASE_OK; lc--) {
      nbase_heap_item_t *cands = NULL;
      size_t cand_count = 0;
      st = search_layer_mt(ctx, query_vec, curr, hnsw->ef_construction, lc, &vis, &nb_scratch, &nb_scratch_cap, &cands, &cand_count);
      if (st != NBASE_OK) {
        free(cands);
        break;
      }

      uint32_t max_m = (lc == 0) ? hnsw->m_max0 : hnsw->m_max;
      uint32_t sel_count = 0U;
      select_neighbors_heuristic(store, ctx->dist_fn, slot, cands, cand_count, max_m, selected, &sel_count);

      pthread_mutex_t *self_lk = &ctx->node_locks[slot & (NBASE_HNSW_STRIPE_COUNT - 1U)];
      pthread_mutex_lock(self_lk);
      for (uint32_t i = 0U; i < sel_count; i++) {
        if (!adj_push(&hnsw->adj[slot][lc], selected[i])) {
          st = NBASE_ERR_NOMEM;
          break;
        }
      }
      if (st == NBASE_OK && hnsw->adj[slot][lc].len > max_m) {
        st = shrink_connections(hnsw, store, ctx->dist_fn, slot, lc, max_m);
      }
      pthread_mutex_unlock(self_lk);

      for (uint32_t i = 0U; i < sel_count && st == NBASE_OK; i++) {
        uint32_t nb = selected[i];
        if (lc <= hnsw->levels[nb] && hnsw->adj[nb] != NULL) {
          pthread_mutex_t *nb_lk = &ctx->node_locks[nb & (NBASE_HNSW_STRIPE_COUNT - 1U)];
          pthread_mutex_lock(nb_lk);
          if (!adj_push(&hnsw->adj[nb][lc], slot)) {
            st = NBASE_ERR_NOMEM;
          } else {
            st = shrink_connections(hnsw, store, ctx->dist_fn, nb, lc, max_m);
          }
          pthread_mutex_unlock(nb_lk);
        }
      }

      if (cand_count > 0) {
        curr = cands[0].slot;
      }
      free(cands);
    }

    if (st != NBASE_OK) {
      pthread_mutex_lock(&ctx->work_lock);
      if (ctx->status == NBASE_OK) {
        ctx->status = st;
      }
      pthread_mutex_unlock(&ctx->work_lock);
      break;
    }

    pthread_mutex_lock(&ctx->global_lock);
    if (level > hnsw->max_level) {
      hnsw->entry_point = slot;
      hnsw->max_level = level;
    }
    pthread_mutex_unlock(&ctx->global_lock);
  }

  free(nb_scratch);
  free(selected);
  visited_free(&vis);
  return NULL;
}

nbase_status_t nbase_hnsw_insert_batch_mt(
    nbase_hnsw_t *hnsw,
    const nbase_store_t *store,
    const uint32_t *slots,
    uint32_t count,
    uint32_t num_threads) {
  if (!hnsw || !store || (!slots && count > 0U)) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }

  uint32_t max_slot = 0U;
  for (uint32_t i = 0U; i < count; i++) {
    if (slots[i] >= store->slot_count) {
      return NBASE_ERR_INVALID_ARG;
    }
    if (slots[i] > max_slot) {
      max_slot = slots[i];
    }
  }

  if (count == 1U || num_threads <= 1U) {
    for (uint32_t i = 0U; i < count; i++) {
      nbase_status_t st = nbase_hnsw_insert(hnsw, store, slots[i]);
      if (st != NBASE_OK) {
        return st;
      }
    }
    return NBASE_OK;
  }

  nbase_status_t st = hnsw_ensure_node_capacity(hnsw, max_slot + 1U);
  if (st != NBASE_OK) {
    return st;
  }

  for (uint32_t i = 0U; i < count; i++) {
    uint32_t s = slots[i];
    if (hnsw->states[s] != NBASE_SLOT_EMPTY) {
      return NBASE_ERR_INVALID_ARG;
    }
  }

  int32_t *batch_levels = malloc((size_t)count * sizeof(int32_t));
  nbase_adj_t **batch_adj = calloc((size_t)count, sizeof(nbase_adj_t *));
  if (!batch_levels || !batch_adj) {
    free(batch_levels);
    free(batch_adj);
    return NBASE_ERR_NOMEM;
  }

  for (uint32_t i = 0U; i < count; i++) {
    batch_levels[i] = random_level(hnsw);
    batch_adj[i] = calloc((size_t)(batch_levels[i] + 1), sizeof(nbase_adj_t));
    if (!batch_adj[i]) {
      for (uint32_t j = 0U; j < i; j++) {
        free(batch_adj[j]);
      }
      free(batch_adj);
      free(batch_levels);
      return NBASE_ERR_NOMEM;
    }
  }

  if (max_slot + 1U > hnsw->node_count) {
    hnsw->node_count = max_slot + 1U;
  }
  for (uint32_t i = 0U; i < count; i++) {
    uint32_t s = slots[i];
    if (hnsw->states[s] != NBASE_SLOT_EMPTY) {
      for (uint32_t j = 0U; j < i; j++) {
        uint32_t sj = slots[j];
        hnsw->states[sj] = NBASE_SLOT_EMPTY;
        hnsw->levels[sj] = -1;
        hnsw->adj[sj] = NULL;
      }
      for (uint32_t j = 0U; j < count; j++) {
        free(batch_adj[j]);
      }
      free(batch_adj);
      free(batch_levels);
      return NBASE_ERR_INVALID_ARG;
    }
    hnsw->levels[s] = batch_levels[i];
    hnsw->states[s] = NBASE_SLOT_ACTIVE;
    hnsw->adj[s] = batch_adj[i];
  }
  hnsw->active_nodes += count;
  free(batch_adj);
  free(batch_levels);

  uint32_t start_idx = 0U;
  if (hnsw->entry_point == NBASE_INVALID_SLOT) {
    hnsw->entry_point = slots[0];
    hnsw->max_level = hnsw->levels[slots[0]];
    start_idx = 1U;
  }
  if (start_idx >= count) {
    return NBASE_OK;
  }

  uint32_t threads = num_threads > 64U ? 64U : num_threads;
  uint32_t remaining = count - start_idx;
  if (threads > remaining) {
    threads = remaining;
  }

  hnsw_mt_shared_t ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.hnsw = hnsw;
  ctx.store = store;
  ctx.slots = slots;
  ctx.start_idx = start_idx;
  ctx.count = count;
  ctx.dist_fn = nbase_get_distance_fn(store->metric);
  ctx.next_idx = start_idx;
  ctx.status = NBASE_OK;

  pthread_mutex_init(&ctx.work_lock, NULL);
  pthread_mutex_init(&ctx.global_lock, NULL);
  for (uint32_t i = 0U; i < NBASE_HNSW_STRIPE_COUNT; i++) {
    pthread_mutex_init(&ctx.node_locks[i], NULL);
  }

  pthread_t *tids = calloc((size_t)threads, sizeof(pthread_t));
  if (!tids) {
    for (uint32_t i = 0U; i < NBASE_HNSW_STRIPE_COUNT; i++) {
      pthread_mutex_destroy(&ctx.node_locks[i]);
    }
    pthread_mutex_destroy(&ctx.global_lock);
    pthread_mutex_destroy(&ctx.work_lock);
    return NBASE_ERR_NOMEM;
  }

  uint32_t spawned = 0U;
  for (uint32_t t = 0U; t < threads; t++) {
    if (pthread_create(&tids[t], NULL, hnsw_mt_worker, &ctx) == 0) {
      spawned++;
    } else {
      pthread_mutex_lock(&ctx.work_lock);
      if (ctx.status == NBASE_OK) {
        ctx.status = NBASE_ERR_NOMEM;
      }
      pthread_mutex_unlock(&ctx.work_lock);
      break;
    }
  }

  for (uint32_t t = 0U; t < spawned; t++) {
    pthread_join(tids[t], NULL);
  }
  free(tids);

  for (uint32_t i = 0U; i < NBASE_HNSW_STRIPE_COUNT; i++) {
    pthread_mutex_destroy(&ctx.node_locks[i]);
  }
  pthread_mutex_destroy(&ctx.global_lock);
  pthread_mutex_destroy(&ctx.work_lock);

  return ctx.status;
}

void nbase_hnsw_mark_deleted(nbase_hnsw_t *hnsw, uint32_t slot) {
  if (!hnsw || slot >= hnsw->node_count) {
    return;
  }
  if (hnsw->states[slot] != NBASE_SLOT_ACTIVE) {
    return;
  }
  hnsw->states[slot] = NBASE_SLOT_TOMBSTONE;
  hnsw->active_nodes--;
  hnsw->deleted_nodes++;
  if (hnsw->active_nodes == 0U) {
    hnsw->entry_point = NBASE_INVALID_SLOT;
    hnsw->max_level = -1;
  }
}

nbase_status_t nbase_hnsw_search(
    const nbase_hnsw_t *hnsw,
    const nbase_store_t *store,
    const float *query,
    uint32_t k,
    uint32_t ef_override,
    nbase_metric_t metric,
    nbase_filter_fn filter,
    void *filter_ctx,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!hnsw || !store || !query || !out_count || (k > 0U && !out_results)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_count = 0U;
  if (hnsw->entry_point == NBASE_INVALID_SLOT || hnsw->active_nodes == 0U || k == 0U) {
    return NBASE_OK;
  }

  uint32_t ef = (ef_override > 0U) ? ef_override : hnsw->ef_search;
  if (ef < k) {
    ef = k;
  }

  nbase_dist_fn dist_fn = nbase_get_distance_fn(metric);
  uint32_t curr = hnsw->entry_point;
  float curr_dist = dist_fn(query, hnsw_slot_vec(store, curr), store->dim);

  for (int32_t lc = hnsw->max_level; lc > 0; lc--) {
    bool changed = true;
    while (changed) {
      changed = false;
      if (lc <= hnsw->levels[curr] && hnsw->adj[curr] != NULL) {
        const nbase_adj_t *list = &hnsw->adj[curr][lc];
        for (uint32_t i = 0U; i < list->len; i++) {
          uint32_t nb = list->neighbors[i];
          if (nb >= hnsw->node_count || hnsw->states[nb] == NBASE_SLOT_EMPTY) {
            continue;
          }
          float d = dist_fn(query, hnsw_slot_vec(store, nb), store->dim);
          if (d < curr_dist) {
            curr_dist = d;
            curr = nb;
            changed = true;
          }
        }
      }
    }
  }

  hnsw_visited_t vis;
  if (!visited_init(&vis, hnsw->node_count)) {
    return NBASE_ERR_NOMEM;
  }

  nbase_heap_item_t *cands = NULL;
  size_t cand_count = 0;
  nbase_status_t st = search_layer(hnsw, store, query, curr, ef, 0, dist_fn, true, filter, filter_ctx, &vis, &cands, &cand_count);
  visited_free(&vis);
  if (st != NBASE_OK) {
    free(cands);
    return st;
  }

  nbase_result_t *tmp = malloc(cand_count * sizeof(nbase_result_t));
  if (!tmp && cand_count > 0) {
    free(cands);
    return NBASE_ERR_NOMEM;
  }
  for (size_t i = 0; i < cand_count; i++) {
    tmp[i].id = store->ids[cands[i].slot];
    tmp[i].distance = cands[i].dist;
  }
  free(cands);

  if (cand_count > 0) {
    qsort(tmp, cand_count, sizeof(nbase_result_t), nbase_cmp_result_asc);
  }
  uint32_t ret_count = (cand_count < (size_t)k) ? (uint32_t)cand_count : k;
  if (ret_count > 0U) {
    memcpy(out_results, tmp, (size_t)ret_count * sizeof(nbase_result_t));
  }
  free(tmp);

  *out_count = ret_count;
  return NBASE_OK;
}

nbase_status_t nbase_hnsw_save(const nbase_hnsw_t *hnsw, const char *file_path) {
  if (!hnsw || !file_path) {
    return NBASE_ERR_INVALID_ARG;
  }

  size_t payload_bytes = 0;
  for (uint32_t i = 0U; i < hnsw->node_count; i++) {
    payload_bytes += sizeof(int32_t) + sizeof(uint8_t);
    int32_t lvl = hnsw->levels[i];
    if (hnsw->states[i] != NBASE_SLOT_EMPTY && lvl >= 0 && hnsw->adj[i] != NULL) {
      for (int32_t lc = 0; lc <= lvl; lc++) {
        payload_bytes += sizeof(uint32_t) + (size_t)hnsw->adj[i][lc].len * sizeof(uint32_t);
      }
    }
  }

  uint8_t *buf = NULL;
  if (payload_bytes > 0) {
    buf = malloc(payload_bytes);
    if (!buf) {
      return NBASE_ERR_NOMEM;
    }
    size_t pos = 0;
    for (uint32_t i = 0U; i < hnsw->node_count; i++) {
      int32_t lvl = hnsw->levels[i];
      uint8_t state = hnsw->states[i];
      memcpy(buf + pos, &lvl, sizeof(lvl));
      pos += sizeof(lvl);
      memcpy(buf + pos, &state, sizeof(state));
      pos += sizeof(state);
      if (state != NBASE_SLOT_EMPTY && lvl >= 0 && hnsw->adj[i] != NULL) {
        for (int32_t lc = 0; lc <= lvl; lc++) {
          uint32_t len = hnsw->adj[i][lc].len;
          memcpy(buf + pos, &len, sizeof(len));
          pos += sizeof(len);
          if (len > 0U) {
            size_t bytes = (size_t)len * sizeof(uint32_t);
            memcpy(buf + pos, hnsw->adj[i][lc].neighbors, bytes);
            pos += bytes;
          }
        }
      }
    }
  }

  nbi_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBI_MAGIC, 8);
  hdr.version = NBASE_NBI_VERSION;
  hdr.m = hnsw->m;
  hdr.ef_construction = hnsw->ef_construction;
  hdr.ef_search = hnsw->ef_search;
  hdr.max_level = hnsw->max_level;
  hdr.entry_point = hnsw->entry_point;
  hdr.node_count = hnsw->node_count;
  hdr.active_nodes = hnsw->active_nodes;
  hdr.deleted_nodes = hnsw->deleted_nodes;
  hdr.rng_state = hnsw->rng_state;
  hdr.payload_bytes = (uint64_t)payload_bytes;
  hdr.payload_crc32 = nbase_crc32(buf, payload_bytes);
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nbi_header_t, header_crc32));

  nbase_io_chunk_t chunk = {buf, payload_bytes};
  nbase_status_t st = nbase_write_file_atomic(file_path, &hdr, sizeof(hdr), &chunk, 1);
  free(buf);
  return st;
}

nbase_status_t nbase_hnsw_load(const char *file_path, nbase_hnsw_t **out_hnsw) {
  if (!file_path || !out_hnsw) {
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
  if (st_buf.st_size < (off_t)sizeof(nbi_header_t)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;

  nbi_header_t hdr;
  nbase_status_t status = nbase_read_exact(fd, &hdr, sizeof(hdr));
  if (status != NBASE_OK) {
    close(fd);
    return status;
  }

  if (memcmp(hdr.magic, NBASE_NBI_MAGIC, 8) != 0 || hdr.version != NBASE_NBI_VERSION) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.header_crc32 != nbase_crc32(&hdr, offsetof(nbi_header_t, header_crc32))) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.m < 2U || hdr.ef_construction == 0U || hdr.ef_search == 0U || hdr.node_count >= NBASE_INVALID_SLOT || hdr.active_nodes + hdr.deleted_nodes > hdr.node_count || hdr.max_level < -1 || hdr.max_level > NBASE_HNSW_MAX_LEVEL) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.entry_point != NBASE_INVALID_SLOT && hdr.entry_point >= hdr.node_count) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (file_size != sizeof(nbi_header_t) + hdr.payload_bytes || hdr.payload_bytes > (uint64_t)SIZE_MAX) {
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
    status = nbase_read_exact(fd, buf, (size_t)hdr.payload_bytes);
    close(fd);
    if (status != NBASE_OK) {
      free(buf);
      return status;
    }
    if (nbase_crc32(buf, (size_t)hdr.payload_bytes) != hdr.payload_crc32) {
      free(buf);
      return NBASE_ERR_CORRUPT;
    }
  } else {
    close(fd);
  }

  nbase_hnsw_t *hnsw = NULL;
  status = nbase_hnsw_create(hdr.m, hdr.ef_construction, hdr.ef_search, hdr.rng_state, &hnsw);
  if (status != NBASE_OK) {
    free(buf);
    return status;
  }
  hnsw->max_level = hdr.max_level;
  hnsw->entry_point = hdr.entry_point;

  if (hdr.node_count > 0U) {
    status = hnsw_ensure_node_capacity(hnsw, (uint32_t)hdr.node_count);
    if (status != NBASE_OK) {
      free(buf);
      nbase_hnsw_destroy(hnsw);
      return status;
    }
  }
  hnsw->node_count = (uint32_t)hdr.node_count;

  size_t pos = 0;
  size_t payload_len = (size_t)hdr.payload_bytes;
  uint64_t active = 0;
  uint64_t deleted = 0;

  for (uint32_t i = 0U; i < hnsw->node_count; i++) {
    if (pos + sizeof(int32_t) + sizeof(uint8_t) > payload_len) {
      free(buf);
      nbase_hnsw_destroy(hnsw);
      return NBASE_ERR_CORRUPT;
    }
    int32_t lvl = 0;
    uint8_t state = 0;
    memcpy(&lvl, buf + pos, sizeof(lvl));
    pos += sizeof(lvl);
    memcpy(&state, buf + pos, sizeof(state));
    pos += sizeof(state);

    if (lvl < -1 || lvl > hnsw->max_level || (state != NBASE_SLOT_EMPTY && state != NBASE_SLOT_ACTIVE && state != NBASE_SLOT_TOMBSTONE)) {
      free(buf);
      nbase_hnsw_destroy(hnsw);
      return NBASE_ERR_CORRUPT;
    }
    hnsw->levels[i] = lvl;
    hnsw->states[i] = state;

    if (state == NBASE_SLOT_ACTIVE) {
      active++;
    } else if (state == NBASE_SLOT_TOMBSTONE) {
      deleted++;
    }

    if (state != NBASE_SLOT_EMPTY && lvl >= 0) {
      nbase_adj_t *node_adj = calloc((size_t)(lvl + 1), sizeof(nbase_adj_t));
      if (!node_adj) {
        free(buf);
        nbase_hnsw_destroy(hnsw);
        return NBASE_ERR_NOMEM;
      }
      hnsw->adj[i] = node_adj;

      for (int32_t lc = 0; lc <= lvl; lc++) {
        if (pos + sizeof(uint32_t) > payload_len) {
          free(buf);
          nbase_hnsw_destroy(hnsw);
          return NBASE_ERR_CORRUPT;
        }
        uint32_t len = 0U;
        memcpy(&len, buf + pos, sizeof(len));
        pos += sizeof(len);
        if (len > hnsw->node_count || pos + (size_t)len * sizeof(uint32_t) > payload_len) {
          free(buf);
          nbase_hnsw_destroy(hnsw);
          return NBASE_ERR_CORRUPT;
        }
        if (len > 0U) {
          node_adj[lc].neighbors = malloc((size_t)len * sizeof(uint32_t));
          if (!node_adj[lc].neighbors) {
            free(buf);
            nbase_hnsw_destroy(hnsw);
            return NBASE_ERR_NOMEM;
          }
          memcpy(node_adj[lc].neighbors, buf + pos, (size_t)len * sizeof(uint32_t));
          pos += (size_t)len * sizeof(uint32_t);
          node_adj[lc].len = len;
          node_adj[lc].cap = len;
          for (uint32_t e = 0U; e < len; e++) {
            if (node_adj[lc].neighbors[e] >= hnsw->node_count) {
              free(buf);
              nbase_hnsw_destroy(hnsw);
              return NBASE_ERR_CORRUPT;
            }
          }
        }
      }
    }
  }

  free(buf);

  if (pos != payload_len || active != hdr.active_nodes || deleted != hdr.deleted_nodes) {
    nbase_hnsw_destroy(hnsw);
    return NBASE_ERR_CORRUPT;
  }
  hnsw->active_nodes = active;
  hnsw->deleted_nodes = deleted;

  *out_hnsw = hnsw;
  return NBASE_OK;
}

uint64_t nbase_hnsw_memory_bytes(const nbase_hnsw_t *hnsw) {
  if (!hnsw) {
    return 0ULL;
  }
  uint64_t total = (uint64_t)sizeof(*hnsw);
  if (hnsw->levels != NULL) {
    total += (uint64_t)hnsw->node_capacity * (uint64_t)sizeof(int32_t);
  }
  if (hnsw->states != NULL) {
    total += (uint64_t)hnsw->node_capacity * (uint64_t)sizeof(uint8_t);
  }
  if (hnsw->adj != NULL) {
    total += (uint64_t)hnsw->node_capacity * (uint64_t)sizeof(nbase_adj_t *);
    for (uint32_t i = 0U; i < hnsw->node_count; i++) {
      if (hnsw->adj[i] != NULL && hnsw->levels != NULL && hnsw->levels[i] >= 0) {
        total += (uint64_t)(hnsw->levels[i] + 1) * (uint64_t)sizeof(nbase_adj_t);
        for (int32_t lc = 0; lc <= hnsw->levels[i]; lc++) {
          total += (uint64_t)hnsw->adj[i][lc].cap * (uint64_t)sizeof(uint32_t);
        }
      }
    }
  }
  return total;
}
