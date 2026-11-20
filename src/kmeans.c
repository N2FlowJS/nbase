#include "nbase_internal.h"

#include <float.h>
#include <math.h>
#include <string.h>

nbase_status_t nbase_kmeans_train(
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    uint32_t k,
    uint32_t max_iters,
    uint64_t seed,
    float *out_centroids,
    uint32_t *out_assignments) {
  if (!vectors || !out_centroids || count == 0U || dim == 0U || dim > NBASE_MAX_DIM || k == 0U || k > count) {
    return NBASE_ERR_INVALID_ARG;
  }
  if ((size_t)count > SIZE_MAX / (size_t)dim || (size_t)k > SIZE_MAX / (size_t)dim) {
    return NBASE_ERR_NOMEM;
  }
  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)dim, dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  nbase_dist_fn sq_l2 = nbase_get_distance_fn(NBASE_METRIC_SQUARED_EUCLIDEAN);
  uint64_t rng = (seed != 0ULL) ? seed : 42ULL;

  float *min_dist_sq = malloc((size_t)count * sizeof(float));
  uint32_t *assign = malloc((size_t)count * sizeof(uint32_t));
  float *point_err = malloc((size_t)count * sizeof(float));
  double *sums = malloc((size_t)k * (size_t)dim * sizeof(double));
  uint32_t *counts = malloc((size_t)k * sizeof(uint32_t));

  if (!min_dist_sq || !assign || !point_err || !sums || !counts) {
    free(min_dist_sq);
    free(assign);
    free(point_err);
    free(sums);
    free(counts);
    return NBASE_ERR_NOMEM;
  }

  /* K-Means++ initialization */
  uint32_t first = (uint32_t)(nbase_rng_next_u64(&rng) % (uint64_t)count);
  memcpy(out_centroids, vectors + (size_t)first * (size_t)dim, (size_t)dim * sizeof(float));
  for (uint32_t i = 0U; i < count; i++) {
    min_dist_sq[i] = sq_l2(vectors + (size_t)i * (size_t)dim, out_centroids, dim);
  }

  for (uint32_t c = 1U; c < k; c++) {
    double total = 0.0;
    for (uint32_t i = 0U; i < count; i++) {
      total += (double)min_dist_sq[i];
    }

    uint32_t chosen = count - 1U;
    if (total <= 0.0) {
      chosen = (uint32_t)(nbase_rng_next_u64(&rng) % (uint64_t)count);
    } else {
      double target = nbase_rng_next_double(&rng) * total;
      double acc = 0.0;
      for (uint32_t i = 0U; i < count; i++) {
        acc += (double)min_dist_sq[i];
        if (acc >= target) {
          chosen = i;
          break;
        }
      }
    }

    float *dst_centroid = out_centroids + (size_t)c * (size_t)dim;
    memcpy(dst_centroid, vectors + (size_t)chosen * (size_t)dim, (size_t)dim * sizeof(float));

    for (uint32_t i = 0U; i < count; i++) {
      float d = sq_l2(vectors + (size_t)i * (size_t)dim, dst_centroid, dim);
      if (d < min_dist_sq[i]) {
        min_dist_sq[i] = d;
      }
    }
  }

  for (uint32_t i = 0U; i < count; i++) {
    assign[i] = UINT32_MAX;
  }

  uint32_t iters = (max_iters > 0U) ? max_iters : 25U;
  for (uint32_t iter = 0U; iter < iters; iter++) {
    uint32_t changed = 0U;
    memset(sums, 0, (size_t)k * (size_t)dim * sizeof(double));
    memset(counts, 0, (size_t)k * sizeof(uint32_t));

    for (uint32_t i = 0U; i < count; i++) {
      const float *vec = vectors + (size_t)i * (size_t)dim;
      uint32_t best_c = 0U;
      float best_d = FLT_MAX;
      for (uint32_t c = 0U; c < k; c++) {
        float d = sq_l2(vec, out_centroids + (size_t)c * (size_t)dim, dim);
        if (d < best_d) {
          best_d = d;
          best_c = c;
        }
      }

      point_err[i] = best_d;
      if (assign[i] != best_c) {
        assign[i] = best_c;
        changed++;
      }
      counts[best_c]++;
      double *csum = sums + (size_t)best_c * (size_t)dim;
      for (uint32_t d = 0U; d < dim; d++) {
        csum[d] += (double)vec[d];
      }
    }

    /* Recover any empty cluster by stealing the farthest point from a multi-member cluster. */
    for (uint32_t c = 0U; c < k; c++) {
      if (counts[c] > 0U) {
        continue;
      }
      uint32_t far_i = UINT32_MAX;
      float max_err = -1.0f;
      for (uint32_t i = 0U; i < count; i++) {
        uint32_t owner = assign[i];
        if (counts[owner] > 1U && point_err[i] > max_err) {
          max_err = point_err[i];
          far_i = i;
        }
      }
      if (far_i == UINT32_MAX) {
        far_i = c % count;
      }
      uint32_t old_c = assign[far_i];
      const float *vec = vectors + (size_t)far_i * (size_t)dim;
      double *old_sum = sums + (size_t)old_c * (size_t)dim;
      double *new_sum = sums + (size_t)c * (size_t)dim;
      for (uint32_t d = 0U; d < dim; d++) {
        old_sum[d] -= (double)vec[d];
        new_sum[d] += (double)vec[d];
      }
      counts[old_c]--;
      counts[c]++;
      assign[far_i] = c;
      point_err[far_i] = 0.0f;
      changed++;
    }

    double max_shift_sq = 0.0;
    for (uint32_t c = 0U; c < k; c++) {
      float *dst = out_centroids + (size_t)c * (size_t)dim;
      const double *csum = sums + (size_t)c * (size_t)dim;
      double inv = 1.0 / (double)counts[c];
      double shift_sq = 0.0;
      for (uint32_t d = 0U; d < dim; d++) {
        float next_v = (float)(csum[d] * inv);
        double delta = (double)next_v - (double)dst[d];
        shift_sq += delta * delta;
        dst[d] = next_v;
      }
      if (shift_sq > max_shift_sq) {
        max_shift_sq = shift_sq;
      }
    }

    if (changed == 0U || max_shift_sq < 1e-12) {
      break;
    }
  }

  if (out_assignments != NULL) {
    memcpy(out_assignments, assign, (size_t)count * sizeof(uint32_t));
  }

  free(min_dist_sq);
  free(assign);
  free(point_err);
  free(sums);
  free(counts);
  return NBASE_OK;
}
