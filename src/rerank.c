#include "nbase_internal.h"

#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static uint64_t hash_u64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static int cmp_edge_canonical(const void *pa, const void *pb) {
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

static void mmr_update_pairwise_min_div(
    const nbase_store_t *store,
    const uint32_t *slots,
    const bool *used,
    uint32_t candidate_count,
    uint32_t ref_slot,
    bool take_min,
    float *min_div) {
  const float *ref_vec = nbase_store_get_vector(store, ref_slot);
  uint32_t bidx[64];
  uint32_t bslots[64];
  float bdists[64];
  uint32_t bn = 0U;
  for (uint32_t i = 0U; i < candidate_count; i++) {
    if (used[i]) {
      continue;
    }
    bidx[bn] = i;
    bslots[bn] = slots[i];
    bn++;
    if (bn == 64U) {
      nbase_distance_indexed_batch(
          store->metric, ref_vec, store->vectors, bslots, bn, store->dim, bdists);
      for (uint32_t j = 0U; j < bn; j++) {
        uint32_t ci = bidx[j];
        if (!take_min || bdists[j] < min_div[ci]) {
          min_div[ci] = bdists[j];
        }
      }
      bn = 0U;
    }
  }
  if (bn > 0U) {
    nbase_distance_indexed_batch(
        store->metric, ref_vec, store->vectors, bslots, bn, store->dim, bdists);
    for (uint32_t j = 0U; j < bn; j++) {
      uint32_t ci = bidx[j];
      if (!take_min || bdists[j] < min_div[ci]) {
        min_div[ci] = bdists[j];
      }
    }
  }
}

nbase_status_t nbase_rerank_mmr(
    const nbase_db_t *db,
    const nbase_result_t *candidates,
    uint32_t candidate_count,
    uint32_t k,
    float lambda,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!db || !out_count || (candidate_count > 0U && !candidates) || (k > 0U && !out_results) ||
      isnan(lambda) || lambda < 0.0f || lambda > 1.0f) {
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

  nbase_db_rdlock(db);
  const nbase_store_t *store = nbase_db_store(db);

  uint32_t *slots = malloc((size_t)candidate_count * sizeof(uint32_t));
  float *rel = malloc((size_t)candidate_count * sizeof(float));
  float *min_div = malloc((size_t)candidate_count * sizeof(float));
  bool *used = calloc((size_t)candidate_count, sizeof(bool));
  if (!slots || !rel || !min_div || !used) {
    free(slots);
    free(rel);
    free(min_div);
    free(used);
    nbase_db_unlock(db);
    return NBASE_ERR_NOMEM;
  }

  float d_min = candidates[0].distance;
  float d_max = candidates[0].distance;
  for (uint32_t i = 0U; i < candidate_count; i++) {
    if (!nbase_store_find_slot(store, candidates[i].id, &slots[i])) {
      free(slots);
      free(rel);
      free(min_div);
      free(used);
      nbase_db_unlock(db);
      return NBASE_ERR_NOT_FOUND;
    }
    if (candidates[i].distance < d_min) {
      d_min = candidates[i].distance;
    }
    if (candidates[i].distance > d_max) {
      d_max = candidates[i].distance;
    }
  }

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

  out_results[0] = candidates[first];
  used[first] = true;

  float pair_min = 0.0f;
  float pair_max = 0.0f;
  bool first_pair_set = false;
  mmr_update_pairwise_min_div(store, slots, used, candidate_count, slots[first], false, min_div);
  for (uint32_t i = 0U; i < candidate_count; i++) {
    if (used[i]) {
      continue;
    }
    float pd = min_div[i];
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

    mmr_update_pairwise_min_div(store, slots, used, candidate_count, slots[best_idx], true, min_div);
  }

  *out_count = eff_k;

  free(slots);
  free(rel);
  free(min_div);
  free(used);
  nbase_db_unlock(db);
  return NBASE_OK;
}

typedef struct weighted_cand {
  uint64_t id;
  float weighted_dist;
  float orig_dist;
} weighted_cand_t;

static int cmp_weighted_cand_asc(const void *pa, const void *pb) {
  const weighted_cand_t *a = (const weighted_cand_t *)pa;
  const weighted_cand_t *b = (const weighted_cand_t *)pb;
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

nbase_status_t nbase_rerank_weighted(
    const nbase_db_t *db,
    const nbase_result_t *candidates,
    uint32_t candidate_count,
    const nbase_rerank_weight_t *weights,
    uint32_t weight_count,
    uint32_t k,
    nbase_result_t *out_results,
    uint32_t *out_count) {
  if (!db || !out_count || (candidate_count > 0U && !candidates) ||
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

  weighted_cand_t *items = malloc((size_t)candidate_count * sizeof(weighted_cand_t));
  if (!items) {
    return NBASE_ERR_NOMEM;
  }

  nbase_db_rdlock(db);
  const nbase_store_t *store = nbase_db_store(db);
  const nbase_meta_store_t *meta = nbase_db_meta(db);
  for (uint32_t i = 0U; i < candidate_count; i++) {
    if (!nbase_store_find_slot(store, candidates[i].id, NULL)) {
      free(items);
      nbase_db_unlock(db);
      return NBASE_ERR_NOT_FOUND;
    }
    double w_dist = (double)candidates[i].distance;
    for (uint32_t w = 0U; w < weight_count; w++) {
      nbase_meta_value_t val;
      if (nbase_meta_get(meta, candidates[i].id, weights[w].key, &val, NULL, 0U) == NBASE_OK) {
        if (val.type == NBASE_META_INT64) {
          w_dist -= (double)val.as.i64 * (double)weights[w].weight;
        } else if (val.type == NBASE_META_DOUBLE) {
          w_dist -= val.as.f64 * (double)weights[w].weight;
        }
      }
    }
    items[i].id = candidates[i].id;
    items[i].weighted_dist = (float)w_dist;
    items[i].orig_dist = candidates[i].distance;
  }
  nbase_db_unlock(db);

  qsort(items, candidate_count, sizeof(weighted_cand_t), cmp_weighted_cand_asc);
  uint32_t eff_k = (k < candidate_count) ? k : candidate_count;
  for (uint32_t i = 0U; i < eff_k; i++) {
    out_results[i].id = items[i].id;
    out_results[i].distance = items[i].weighted_dist;
  }
  *out_count = eff_k;
  free(items);
  return NBASE_OK;
}

typedef struct rel_filter_ctx {
  const nbase_meta_store_t *meta;
  const nbase_meta_filter_t *filter;
} rel_filter_ctx_t;

static bool rel_filter_cb(uint64_t id, void *raw) {
  const rel_filter_ctx_t *ctx = (const rel_filter_ctx_t *)raw;
  return nbase_meta_matches(ctx->meta, id, ctx->filter);
}

static void rel_knn_consider(
    nbase_result_t *knn,
    uint32_t *got,
    uint32_t eff_k,
    uint64_t other_id,
    float d) {
  if (*got < eff_k) {
    uint32_t pos = (*got)++;
    while (pos > 0U &&
           (d < knn[pos - 1U].distance ||
            (d == knn[pos - 1U].distance && other_id < knn[pos - 1U].id))) {
      knn[pos] = knn[pos - 1U];
      pos--;
    }
    knn[pos].id = other_id;
    knn[pos].distance = d;
  } else if (d < knn[*got - 1U].distance ||
             (d == knn[*got - 1U].distance && other_id < knn[*got - 1U].id)) {
    uint32_t pos = *got - 1U;
    while (pos > 0U &&
           (d < knn[pos - 1U].distance ||
            (d == knn[pos - 1U].distance && other_id < knn[pos - 1U].id))) {
      knn[pos] = knn[pos - 1U];
      pos--;
    }
    knn[pos].id = other_id;
    knn[pos].distance = d;
  }
}

typedef struct rel_extract_ctx {
  const nbase_store_t *store;
  const nbase_hnsw_t *hnsw;
  const nbase_meta_store_t *meta;
  const nbase_meta_filter_t *meta_filter;
  const uint32_t *slots;
  const float *dense_vecs;
  nbase_dist_fn dist_fn;
  nbase_edge_t *edges;
  uint32_t active_cnt;
  uint32_t eff_k;
  float max_distance;
  bool use_hnsw;
  bool has_filter;
  uint32_t total_workers;
  size_t counts[4];
  nbase_status_t statuses[4];
} rel_extract_ctx_t;

typedef struct rel_extract_arg {
  rel_extract_ctx_t *ctx;
  uint32_t tid;
} rel_extract_arg_t;

static void rel_extract_slice(rel_extract_ctx_t *ctx, uint32_t tid) {
  uint32_t start =
      (uint32_t)(((uint64_t)tid * (uint64_t)ctx->active_cnt) / (uint64_t)ctx->total_workers);
  uint32_t end = (uint32_t)(((uint64_t)(tid + 1U) * (uint64_t)ctx->active_cnt) /
                            (uint64_t)ctx->total_workers);

  nbase_result_t *knn = malloc(((size_t)ctx->eff_k + 1U) * sizeof(nbase_result_t));
  float *row_dists = (!ctx->use_hnsw && ctx->dense_vecs != NULL)
                         ? malloc((size_t)ctx->active_cnt * sizeof(float))
                         : NULL;
  if (!knn || (!ctx->use_hnsw && ctx->dense_vecs != NULL && !row_dists)) {
    free(knn);
    free(row_dists);
    ctx->counts[tid] = 0U;
    ctx->statuses[tid] = NBASE_ERR_NOMEM;
    return;
  }

  rel_filter_ctx_t fctx = {ctx->meta, ctx->meta_filter};
  size_t base_off = (size_t)start * (size_t)ctx->eff_k;
  size_t wlen = 0U;
  nbase_status_t st = NBASE_OK;

  for (uint32_t i = start; i < end && st == NBASE_OK; i++) {
    uint32_t self_slot = ctx->slots[i];
    uint64_t self_id = ctx->store->ids[self_slot];
    const float *self_vec = nbase_store_get_vector(ctx->store, self_slot);
    uint32_t got = 0U;

    if (ctx->use_hnsw) {
      st = nbase_hnsw_search(
          ctx->hnsw,
          ctx->store,
          self_vec,
          ctx->eff_k + 1U,
          0U,
          ctx->store->metric,
          ctx->has_filter ? rel_filter_cb : NULL,
          ctx->has_filter ? &fctx : NULL,
          knn,
          &got);
      if (st != NBASE_OK) {
        break;
      }
    } else if (row_dists != NULL &&
               nbase_distance_batch(
                   ctx->store->metric,
                   self_vec,
                   ctx->dense_vecs,
                   ctx->active_cnt,
                   ctx->store->dim,
                   row_dists) == NBASE_OK) {
      for (uint32_t j = 0U; j < ctx->active_cnt; j++) {
        uint32_t other_slot = ctx->slots[j];
        if (other_slot == self_slot) {
          continue;
        }
        rel_knn_consider(knn, &got, ctx->eff_k, ctx->store->ids[other_slot], row_dists[j]);
      }
    } else {
      for (uint32_t j = 0U; j < ctx->active_cnt; j++) {
        uint32_t other_slot = ctx->slots[j];
        if (other_slot == self_slot) {
          continue;
        }
        float d =
            ctx->dist_fn(self_vec, nbase_store_get_vector(ctx->store, other_slot), ctx->store->dim);
        rel_knn_consider(knn, &got, ctx->eff_k, ctx->store->ids[other_slot], d);
      }
    }

    uint32_t added_for_node = 0U;
    for (uint32_t r = 0U; r < got && added_for_node < ctx->eff_k; r++) {
      if (knn[r].id == self_id) {
        continue;
      }
      if (knn[r].distance <= ctx->max_distance) {
        uint64_t a = self_id < knn[r].id ? self_id : knn[r].id;
        uint64_t b = self_id < knn[r].id ? knn[r].id : self_id;
        nbase_edge_t *dst_e = &ctx->edges[base_off + wlen];
        dst_e->src_id = a;
        dst_e->dst_id = b;
        dst_e->weight = knn[r].distance;
        wlen++;
      }
      added_for_node++;
    }
  }

  free(row_dists);
  free(knn);
  ctx->counts[tid] = wlen;
  ctx->statuses[tid] = st;
}

static void *rel_extract_worker(void *raw) {
  rel_extract_arg_t *arg = (rel_extract_arg_t *)raw;
  rel_extract_slice(arg->ctx, arg->tid);
  return NULL;
}

nbase_status_t nbase_extract_relationships_filtered(
    const nbase_db_t *db,
    uint32_t k_neighbors,
    float max_distance,
    const nbase_meta_filter_t *meta_filter,
    nbase_edge_t **out_edges,
    uint64_t *out_edge_count) {
  if (!db || !out_edges || !out_edge_count || k_neighbors == 0U || isnan(max_distance) ||
      (meta_filter != NULL && meta_filter->count > 0U && !meta_filter->conds)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_edges = NULL;
  *out_edge_count = 0ULL;

  nbase_db_rdlock(db);
  const nbase_store_t *store = nbase_db_store(db);
  const nbase_meta_store_t *meta = nbase_db_meta(db);
  if (store->active_count < 2ULL) {
    nbase_db_unlock(db);
    return NBASE_OK;
  }

  bool has_filter = (meta_filter != NULL && meta_filter->count > 0U);
  uint32_t active_cap = (uint32_t)store->active_count;
  uint32_t *slots = malloc((size_t)active_cap * sizeof(uint32_t));
  if (!slots) {
    nbase_db_unlock(db);
    return NBASE_ERR_NOMEM;
  }
  uint32_t s_pos = 0U;
  for (uint32_t s = 0U; s < store->slot_count && s_pos < active_cap; s++) {
    if (store->slot_flags[s] == NBASE_SLOT_ACTIVE) {
      if (has_filter && !nbase_meta_matches(meta, store->ids[s], meta_filter)) {
        continue;
      }
      slots[s_pos++] = s;
    }
  }
  if (s_pos < 2U) {
    free(slots);
    nbase_db_unlock(db);
    return NBASE_OK;
  }
  uint32_t active_cnt = s_pos;

  uint32_t eff_k = (k_neighbors < active_cnt - 1U) ? k_neighbors : (active_cnt - 1U);
  size_t raw_cap = (size_t)active_cnt * (size_t)eff_k;
  nbase_edge_t *edges = malloc(raw_cap * sizeof(nbase_edge_t));
  if (!edges) {
    free(slots);
    nbase_db_unlock(db);
    return NBASE_ERR_NOMEM;
  }

  const nbase_hnsw_t *hnsw = nbase_db_hnsw(db);
  bool use_hnsw = nbase_db_hnsw_enabled(db) && hnsw != NULL && active_cnt > 64U;
  nbase_dist_fn dist_fn = nbase_get_distance_fn(store->metric);
  float *dense_vecs = NULL;
  if (!use_hnsw && (size_t)active_cnt <= SIZE_MAX / ((size_t)store->dim * sizeof(float))) {
    dense_vecs = malloc((size_t)active_cnt * (size_t)store->dim * sizeof(float));
    if (dense_vecs) {
      size_t row_bytes = (size_t)store->dim * sizeof(float);
      for (uint32_t j = 0U; j < active_cnt; j++) {
        memcpy(
            dense_vecs + (size_t)j * (size_t)store->dim,
            nbase_store_get_vector(store, slots[j]),
            row_bytes);
      }
    }
  }

  rel_extract_ctx_t ctx;
  ctx.store = store;
  ctx.hnsw = hnsw;
  ctx.meta = meta;
  ctx.meta_filter = meta_filter;
  ctx.slots = slots;
  ctx.dense_vecs = dense_vecs;
  ctx.dist_fn = dist_fn;
  ctx.edges = edges;
  ctx.active_cnt = active_cnt;
  ctx.eff_k = eff_k;
  ctx.max_distance = max_distance;
  ctx.use_hnsw = use_hnsw;
  ctx.has_filter = has_filter;
  ctx.total_workers = (active_cnt >= 128U) ? 4U : 1U;
  for (uint32_t w = 0U; w < 4U; w++) {
    ctx.counts[w] = 0U;
    ctx.statuses[w] = NBASE_OK;
  }

  pthread_t workers[3];
  rel_extract_arg_t args[3];
  uint32_t spawned = 0U;
  if (ctx.total_workers > 1U) {
    for (uint32_t w = 1U; w < ctx.total_workers; w++) {
      args[w - 1U].ctx = &ctx;
      args[w - 1U].tid = w;
      if (pthread_create(&workers[spawned], NULL, rel_extract_worker, &args[w - 1U]) != 0) {
        break;
      }
      spawned++;
    }
  }
  rel_extract_slice(&ctx, 0U);
  for (uint32_t w = spawned + 1U; w < ctx.total_workers; w++) {
    rel_extract_slice(&ctx, w);
  }
  for (uint32_t w = 0U; w < spawned; w++) {
    pthread_join(workers[w], NULL);
  }

  nbase_status_t st = NBASE_OK;
  for (uint32_t w = 0U; w < ctx.total_workers; w++) {
    if (ctx.statuses[w] != NBASE_OK) {
      st = ctx.statuses[w];
      break;
    }
  }

  size_t edge_len = ctx.counts[0];
  if (st == NBASE_OK && ctx.total_workers > 1U) {
    for (uint32_t w = 1U; w < ctx.total_workers; w++) {
      uint32_t start_w =
          (uint32_t)(((uint64_t)w * (uint64_t)active_cnt) / (uint64_t)ctx.total_workers);
      size_t src_off = (size_t)start_w * (size_t)eff_k;
      size_t cnt_w = ctx.counts[w];
      if (cnt_w > 0U && edge_len != src_off) {
        memmove(edges + edge_len, edges + src_off, cnt_w * sizeof(nbase_edge_t));
      }
      edge_len += cnt_w;
    }
  }

  free(dense_vecs);
  free(slots);
  nbase_db_unlock(db);

  if (st != NBASE_OK) {
    free(edges);
    return st;
  }

  if (edge_len == 0U) {
    free(edges);
    return NBASE_OK;
  }

  qsort(edges, edge_len, sizeof(nbase_edge_t), cmp_edge_canonical);
  size_t unique_len = 0U;
  for (size_t i = 0U; i < edge_len; i++) {
    if (unique_len == 0U || edges[i].src_id != edges[unique_len - 1U].src_id || edges[i].dst_id != edges[unique_len - 1U].dst_id) {
      edges[unique_len++] = edges[i];
    }
  }

  nbase_edge_t *shrunk = realloc(edges, unique_len * sizeof(nbase_edge_t));
  *out_edges = shrunk ? shrunk : edges;
  *out_edge_count = (uint64_t)unique_len;
  return NBASE_OK;
}

nbase_status_t nbase_extract_relationships(
    const nbase_db_t *db,
    uint32_t k_neighbors,
    float max_distance,
    nbase_edge_t **out_edges,
    uint64_t *out_edge_count) {
  return nbase_extract_relationships_filtered(
      db, k_neighbors, max_distance, NULL, out_edges, out_edge_count);
}

static uint32_t uf_find_iter(uint32_t *parent, uint32_t x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

nbase_status_t nbase_extract_communities(
    const uint64_t *ids,
    uint32_t node_count,
    const nbase_edge_t *edges,
    uint64_t edge_count,
    uint32_t *out_labels,
    uint32_t *out_community_count) {
  if (!out_community_count || (node_count > 0U && (!ids || !out_labels)) || (edge_count > 0ULL && !edges)) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_community_count = 0U;
  if (node_count == 0U) {
    return NBASE_OK;
  }

  size_t map_cap = 16U;
  while (map_cap < (size_t)node_count * 2U) {
    map_cap *= 2U;
  }
  uint64_t *map_keys = malloc(map_cap * sizeof(uint64_t));
  uint32_t *map_vals = malloc(map_cap * sizeof(uint32_t));
  uint8_t *map_used = calloc(map_cap, sizeof(uint8_t));
  uint32_t *parent = malloc((size_t)node_count * sizeof(uint32_t));
  uint32_t *rank = calloc((size_t)node_count, sizeof(uint32_t));
  uint32_t *root_label = malloc((size_t)node_count * sizeof(uint32_t));
  if (!map_keys || !map_vals || !map_used || !parent || !rank || !root_label) {
    free(map_keys);
    free(map_vals);
    free(map_used);
    free(parent);
    free(rank);
    free(root_label);
    return NBASE_ERR_NOMEM;
  }

  size_t mask = map_cap - 1U;
  for (uint32_t i = 0U; i < node_count; i++) {
    parent[i] = i;
    root_label[i] = UINT32_MAX;

    uint64_t key = ids[i];
    size_t pos = (size_t)hash_u64(key) & mask;
    while (map_used[pos] != 0U) {
      if (map_keys[pos] == key) {
        free(map_keys);
        free(map_vals);
        free(map_used);
        free(parent);
        free(rank);
        free(root_label);
        return NBASE_ERR_ALREADY_EXISTS;
      }
      pos = (pos + 1U) & mask;
    }
    map_used[pos] = 1U;
    map_keys[pos] = key;
    map_vals[pos] = i;
  }

  for (uint64_t e = 0ULL; e < edge_count; e++) {
    uint64_t k1 = edges[e].src_id;
    uint64_t k2 = edges[e].dst_id;
    if (k1 == k2) {
      continue;
    }

    size_t p1 = (size_t)hash_u64(k1) & mask;
    bool f1 = false;
    uint32_t u = 0U;
    while (map_used[p1] != 0U) {
      if (map_keys[p1] == k1) {
        f1 = true;
        u = map_vals[p1];
        break;
      }
      p1 = (p1 + 1U) & mask;
    }

    size_t p2 = (size_t)hash_u64(k2) & mask;
    bool f2 = false;
    uint32_t v = 0U;
    while (map_used[p2] != 0U) {
      if (map_keys[p2] == k2) {
        f2 = true;
        v = map_vals[p2];
        break;
      }
      p2 = (p2 + 1U) & mask;
    }

    if (f1 && f2) {
      uint32_t ru = uf_find_iter(parent, u);
      uint32_t rv = uf_find_iter(parent, v);
      if (ru != rv) {
        if (rank[ru] < rank[rv]) {
          parent[ru] = rv;
        } else if (rank[ru] > rank[rv]) {
          parent[rv] = ru;
        } else {
          parent[rv] = ru;
          rank[ru]++;
        }
      }
    }
  }

  uint32_t next_label = 0U;
  for (uint32_t i = 0U; i < node_count; i++) {
    uint32_t r = uf_find_iter(parent, i);
    if (root_label[r] == UINT32_MAX) {
      root_label[r] = next_label++;
    }
    out_labels[i] = root_label[r];
  }
  *out_community_count = next_label;

  free(map_keys);
  free(map_vals);
  free(map_used);
  free(parent);
  free(rank);
  free(root_label);
  return NBASE_OK;
}

void nbase_free_edges(nbase_edge_t *edges) {
  free(edges);
}
