#include "nbase_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define NBASE_SQ8_HAS_AVX2 1
#else
#define NBASE_SQ8_HAS_AVX2 0
#endif

struct nbase_sq8 {
  uint32_t dim;
  float *mins;
  float *scales;
  float *inv_scales;
};

nbase_status_t nbase_sq8_train(
    const float *vectors,
    uint32_t count,
    uint32_t dim,
    nbase_sq8_t **out_sq8) {
  if (!vectors || !out_sq8 || count == 0U || dim == 0U || dim > NBASE_MAX_DIM) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_sq8 = NULL;

  for (uint32_t i = 0U; i < count; i++) {
    if (!nbase_vec_is_finite(vectors + (size_t)i * (size_t)dim, dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  nbase_sq8_t *sq8 = calloc(1, sizeof(*sq8));
  float *mins = malloc((size_t)dim * sizeof(float));
  float *scales = malloc((size_t)dim * sizeof(float));
  float *inv_scales = malloc((size_t)dim * sizeof(float));
  if (!sq8 || !mins || !scales || !inv_scales) {
    free(sq8);
    free(mins);
    free(scales);
    free(inv_scales);
    return NBASE_ERR_NOMEM;
  }

  for (uint32_t d = 0U; d < dim; d++) {
    float vmin = vectors[d];
    float vmax = vectors[d];
    for (uint32_t i = 1U; i < count; i++) {
      float x = vectors[(size_t)i * (size_t)dim + d];
      if (x < vmin) {
        vmin = x;
      }
      if (x > vmax) {
        vmax = x;
      }
    }
    mins[d] = vmin;
    float span = vmax - vmin;
    if (span > 0.0f && isfinite(span)) {
      scales[d] = span / 255.0f;
      inv_scales[d] = (scales[d] > 0.0f) ? (1.0f / scales[d]) : 0.0f;
    } else {
      scales[d] = 0.0f;
      inv_scales[d] = 0.0f;
    }
  }

  sq8->dim = dim;
  sq8->mins = mins;
  sq8->scales = scales;
  sq8->inv_scales = inv_scales;
  *out_sq8 = sq8;
  return NBASE_OK;
}

void nbase_sq8_destroy(nbase_sq8_t *sq8) {
  if (!sq8) {
    return;
  }
  free(sq8->mins);
  free(sq8->scales);
  free(sq8->inv_scales);
  free(sq8);
}

uint32_t nbase_sq8_dim(const nbase_sq8_t *sq8) {
  return sq8 ? sq8->dim : 0U;
}

nbase_status_t nbase_sq8_encode(
    const nbase_sq8_t *sq8,
    const float *vectors,
    uint32_t count,
    uint8_t *out_codes) {
  if (!sq8 || (count > 0U && (!vectors || !out_codes))) {
    return NBASE_ERR_INVALID_ARG;
  }
  for (uint32_t i = 0U; i < count; i++) {
    const float *v = vectors + (size_t)i * (size_t)sq8->dim;
    if (!nbase_vec_is_finite(v, sq8->dim)) {
      return NBASE_ERR_INVALID_VECTOR;
    }
  }

  for (uint32_t i = 0U; i < count; i++) {
    const float *v = vectors + (size_t)i * (size_t)sq8->dim;
    uint8_t *dst = out_codes + (size_t)i * (size_t)sq8->dim;
    for (uint32_t d = 0U; d < sq8->dim; d++) {
      if (sq8->inv_scales[d] <= 0.0f) {
        dst[d] = 0U;
      } else {
        float q = roundf((v[d] - sq8->mins[d]) * sq8->inv_scales[d]);
        if (q <= 0.0f) {
          dst[d] = 0U;
        } else if (q >= 255.0f) {
          dst[d] = 255U;
        } else {
          dst[d] = (uint8_t)q;
        }
      }
    }
  }
  return NBASE_OK;
}

#if NBASE_SQ8_HAS_AVX2
static inline __attribute__((target("avx2"))) float sq8_hsum256_ps(__m256 v);
static __attribute__((target("avx2"))) void sq8_decode_avx2(
    const nbase_sq8_t *sq8,
    const uint8_t *codes,
    uint32_t count,
    float *out_vectors) {
  const float *mins = sq8->mins;
  const float *scales = sq8->scales;
  uint32_t dim = sq8->dim;

  for (uint32_t i = 0U; i < count; i++) {
    const uint8_t *src = codes + (size_t)i * (size_t)dim;
    float *dst = out_vectors + (size_t)i * (size_t)dim;
    uint32_t d = 0U;
    for (; d + 8U <= dim; d += 8U) {
      __m128i u8 = _mm_loadl_epi64((const __m128i *)(const void *)(src + d));
      __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8));
      __m256 mv = _mm256_loadu_ps(mins + d);
      __m256 sv = _mm256_loadu_ps(scales + d);
      __m256 res = _mm256_add_ps(mv, _mm256_mul_ps(qf, sv));
      _mm256_storeu_ps(dst + d, res);
    }
    for (; d < dim; d++) {
      dst[d] = mins[d] + (float)src[d] * scales[d];
    }
  }
}
#endif

nbase_status_t nbase_sq8_decode(
    const nbase_sq8_t *sq8,
    const uint8_t *codes,
    uint32_t count,
    float *out_vectors) {
  if (!sq8 || (count > 0U && (!codes || !out_vectors))) {
    return NBASE_ERR_INVALID_ARG;
  }
#if NBASE_SQ8_HAS_AVX2
  if (nbase_cpu_has_avx2_fma() && sq8->dim >= 8U) {
    sq8_decode_avx2(sq8, codes, count, out_vectors);
    return NBASE_OK;
  }
#endif
  for (uint32_t i = 0U; i < count; i++) {
    const uint8_t *src = codes + (size_t)i * (size_t)sq8->dim;
    float *dst = out_vectors + (size_t)i * (size_t)sq8->dim;
    for (uint32_t d = 0U; d < sq8->dim; d++) {
      dst[d] = sq8->mins[d] + (float)src[d] * sq8->scales[d];
    }
  }
  return NBASE_OK;
}

#if NBASE_SQ8_HAS_AVX2
static inline __attribute__((target("avx2"))) float sq8_hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(sum);
  sum = _mm_add_ps(sum, shuf);
  shuf = _mm_movehl_ps(shuf, sum);
  sum = _mm_add_ss(sum, shuf);
  return _mm_cvtss_f32(sum);
}

static __attribute__((target("avx2,fma"))) float sq8_l2_sq_asymmetric_avx2(
    const nbase_sq8_t *sq8,
    const float *query,
    const uint8_t *code) {
  __m256 acc = _mm256_setzero_ps();
  uint32_t d = 0U;
  for (; d + 8U <= sq8->dim; d += 8U) {
    uint64_t raw8 = 0ULL;
    memcpy(&raw8, code + d, 8U);
    __m128i v8 = _mm_cvtsi64_si128((long long)raw8);
    __m256i v32 = _mm256_cvtepu8_epi32(v8);
    __m256 vf = _mm256_cvtepi32_ps(v32);
    __m256 vmin = _mm256_loadu_ps(sq8->mins + d);
    __m256 vscale = _mm256_loadu_ps(sq8->scales + d);
    __m256 approx = _mm256_fmadd_ps(vf, vscale, vmin);
    __m256 vq = _mm256_loadu_ps(query + d);
    __m256 diff = _mm256_sub_ps(vq, approx);
    acc = _mm256_fmadd_ps(diff, diff, acc);
  }
  float sum = sq8_hsum256_ps(acc);
  for (; d < sq8->dim; d++) {
    float approx = sq8->mins[d] + (float)code[d] * sq8->scales[d];
    float diff = query[d] - approx;
    sum += diff * diff;
  }
  return sum;
}

static __attribute__((target("avx2,fma"))) float sq8_l2_sq_symmetric_avx2(
    const nbase_sq8_t *sq8,
    const uint8_t *code_a,
    const uint8_t *code_b) {
  __m256 acc = _mm256_setzero_ps();
  uint32_t d = 0U;
  for (; d + 8U <= sq8->dim; d += 8U) {
    uint64_t ra = 0ULL;
    uint64_t rb = 0ULL;
    memcpy(&ra, code_a + d, 8U);
    memcpy(&rb, code_b + d, 8U);
    __m256i va32 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)ra));
    __m256i vb32 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)rb));
    __m256 vdiff_i = _mm256_cvtepi32_ps(_mm256_sub_epi32(va32, vb32));
    __m256 vscale = _mm256_loadu_ps(sq8->scales + d);
    __m256 diff = _mm256_mul_ps(vdiff_i, vscale);
    acc = _mm256_fmadd_ps(diff, diff, acc);
  }
  float sum = sq8_hsum256_ps(acc);
  for (; d < sq8->dim; d++) {
    float diff = ((float)code_a[d] - (float)code_b[d]) * sq8->scales[d];
    sum += diff * diff;
  }
  return sum;
}

static __attribute__((target("avx2,fma"))) void sq8_l2_sq_asymmetric_batch_avx2(
    const float *q_shifted,
    const float *scales,
    uint32_t dim,
    const uint8_t *codes,
    uint32_t count,
    float *out_distances) {
  uint32_t i = 0U;
  for (; i + 3U < count; i += 4U) {
    const uint8_t *c0 = codes + (size_t)i * (size_t)dim;
    const uint8_t *c1 = c0 + dim;
    const uint8_t *c2 = c1 + dim;
    const uint8_t *c3 = c2 + dim;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    uint32_t d = 0U;
    for (; d + 8U <= dim; d += 8U) {
      __m256 vs = _mm256_loadu_ps(scales + d);
      __m256 vq = _mm256_loadu_ps(q_shifted + d);
      uint64_t r0 = 0ULL;
      uint64_t r1 = 0ULL;
      uint64_t r2 = 0ULL;
      uint64_t r3 = 0ULL;
      memcpy(&r0, c0 + d, 8U);
      memcpy(&r1, c1 + d, 8U);
      memcpy(&r2, c2 + d, 8U);
      memcpy(&r3, c3 + d, 8U);
      __m256 vf0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)r0)));
      __m256 vf1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)r1)));
      __m256 vf2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)r2)));
      __m256 vf3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)r3)));
      __m256 df0 = _mm256_fnmadd_ps(vf0, vs, vq);
      __m256 df1 = _mm256_fnmadd_ps(vf1, vs, vq);
      __m256 df2 = _mm256_fnmadd_ps(vf2, vs, vq);
      __m256 df3 = _mm256_fnmadd_ps(vf3, vs, vq);
      acc0 = _mm256_fmadd_ps(df0, df0, acc0);
      acc1 = _mm256_fmadd_ps(df1, df1, acc1);
      acc2 = _mm256_fmadd_ps(df2, df2, acc2);
      acc3 = _mm256_fmadd_ps(df3, df3, acc3);
    }
    float s0 = sq8_hsum256_ps(acc0);
    float s1 = sq8_hsum256_ps(acc1);
    float s2 = sq8_hsum256_ps(acc2);
    float s3 = sq8_hsum256_ps(acc3);
    for (; d < dim; d++) {
      float qd = q_shifted[d];
      float sd = scales[d];
      float diff0 = qd - (float)c0[d] * sd;
      float diff1 = qd - (float)c1[d] * sd;
      float diff2 = qd - (float)c2[d] * sd;
      float diff3 = qd - (float)c3[d] * sd;
      s0 += diff0 * diff0;
      s1 += diff1 * diff1;
      s2 += diff2 * diff2;
      s3 += diff3 * diff3;
    }
    out_distances[i] = s0;
    out_distances[i + 1U] = s1;
    out_distances[i + 2U] = s2;
    out_distances[i + 3U] = s3;
  }
  for (; i < count; i++) {
    const uint8_t *code = codes + (size_t)i * (size_t)dim;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t d = 0U;
    for (; d + 16U <= dim; d += 16U) {
      __m128i v16;
      memcpy(&v16, code + d, 16U);
      __m256 vf0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v16));
      __m256 vf1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(v16, 8)));
      __m256 diff0 = _mm256_fnmadd_ps(
          vf0, _mm256_loadu_ps(scales + d), _mm256_loadu_ps(q_shifted + d));
      __m256 diff1 = _mm256_fnmadd_ps(
          vf1, _mm256_loadu_ps(scales + d + 8U), _mm256_loadu_ps(q_shifted + d + 8U));
      acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
      acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
    }
    if (d + 8U <= dim) {
      uint64_t raw8 = 0ULL;
      memcpy(&raw8, code + d, 8U);
      __m128i v8 = _mm_cvtsi64_si128((long long)raw8);
      __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v8));
      __m256 diff = _mm256_fnmadd_ps(
          vf, _mm256_loadu_ps(scales + d), _mm256_loadu_ps(q_shifted + d));
      acc0 = _mm256_fmadd_ps(diff, diff, acc0);
      d += 8U;
    }
    float sum = sq8_hsum256_ps(_mm256_add_ps(acc0, acc1));
    for (; d < dim; d++) {
      float diff = q_shifted[d] - (float)code[d] * scales[d];
      sum += diff * diff;
    }
    out_distances[i] = sum;
  }
}

static __attribute__((target("avx2,fma"))) void sq8_l2_sq_symmetric_batch_avx2(
    const nbase_sq8_t *sq8,
    const uint8_t *query_code,
    const uint8_t *codes,
    uint32_t count,
    float *out_distances) {
  const uint32_t dim = sq8->dim;
  const float *scales = sq8->scales;
  uint32_t i = 0U;
  for (; i + 3U < count; i += 4U) {
    const uint8_t *c0 = codes + (size_t)i * (size_t)dim;
    const uint8_t *c1 = c0 + dim;
    const uint8_t *c2 = c1 + dim;
    const uint8_t *c3 = c2 + dim;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    uint32_t d = 0U;
    for (; d + 8U <= dim; d += 8U) {
      uint64_t rqa = 0ULL;
      uint64_t r0 = 0ULL;
      uint64_t r1 = 0ULL;
      uint64_t r2 = 0ULL;
      uint64_t r3 = 0ULL;
      memcpy(&rqa, query_code + d, 8U);
      memcpy(&r0, c0 + d, 8U);
      memcpy(&r1, c1 + d, 8U);
      memcpy(&r2, c2 + d, 8U);
      memcpy(&r3, c3 + d, 8U);
      __m256i vqa = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)rqa));
      __m256 vs = _mm256_loadu_ps(scales + d);
      __m256 df0 = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_sub_epi32(
              vqa, _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)r0)))),
          vs);
      __m256 df1 = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_sub_epi32(
              vqa, _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)r1)))),
          vs);
      __m256 df2 = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_sub_epi32(
              vqa, _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)r2)))),
          vs);
      __m256 df3 = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_sub_epi32(
              vqa, _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)r3)))),
          vs);
      acc0 = _mm256_fmadd_ps(df0, df0, acc0);
      acc1 = _mm256_fmadd_ps(df1, df1, acc1);
      acc2 = _mm256_fmadd_ps(df2, df2, acc2);
      acc3 = _mm256_fmadd_ps(df3, df3, acc3);
    }
    float s0 = sq8_hsum256_ps(acc0);
    float s1 = sq8_hsum256_ps(acc1);
    float s2 = sq8_hsum256_ps(acc2);
    float s3 = sq8_hsum256_ps(acc3);
    for (; d < dim; d++) {
      float qd = (float)query_code[d];
      float sd = scales[d];
      float diff0 = (qd - (float)c0[d]) * sd;
      float diff1 = (qd - (float)c1[d]) * sd;
      float diff2 = (qd - (float)c2[d]) * sd;
      float diff3 = (qd - (float)c3[d]) * sd;
      s0 += diff0 * diff0;
      s1 += diff1 * diff1;
      s2 += diff2 * diff2;
      s3 += diff3 * diff3;
    }
    out_distances[i] = s0;
    out_distances[i + 1U] = s1;
    out_distances[i + 2U] = s2;
    out_distances[i + 3U] = s3;
  }
  for (; i < count; i++) {
    const uint8_t *code = codes + (size_t)i * (size_t)dim;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t d = 0U;
    for (; d + 16U <= dim; d += 16U) {
      __m128i qa16, cb16;
      memcpy(&qa16, query_code + d, 16U);
      memcpy(&cb16, code + d, 16U);
      __m256 d0 = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_sub_epi32(
              _mm256_cvtepu8_epi32(qa16), _mm256_cvtepu8_epi32(cb16))),
          _mm256_loadu_ps(scales + d));
      __m256 d1 = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_sub_epi32(
              _mm256_cvtepu8_epi32(_mm_srli_si128(qa16, 8)),
              _mm256_cvtepu8_epi32(_mm_srli_si128(cb16, 8)))),
          _mm256_loadu_ps(scales + d + 8U));
      acc0 = _mm256_fmadd_ps(d0, d0, acc0);
      acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    if (d + 8U <= dim) {
      uint64_t ra = 0ULL;
      uint64_t rb = 0ULL;
      memcpy(&ra, query_code + d, 8U);
      memcpy(&rb, code + d, 8U);
      __m256 diff = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_sub_epi32(
              _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)ra)),
              _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)rb)))),
          _mm256_loadu_ps(scales + d));
      acc0 = _mm256_fmadd_ps(diff, diff, acc0);
      d += 8U;
    }
    float sum = sq8_hsum256_ps(_mm256_add_ps(acc0, acc1));
    for (; d < dim; d++) {
      float diff = ((float)query_code[d] - (float)code[d]) * scales[d];
      sum += diff * diff;
    }
    out_distances[i] = sum;
  }
}
#endif

float nbase_sq8_l2_sq_asymmetric(const nbase_sq8_t *sq8, const float *query, const uint8_t *code) {
  if (!sq8 || !query || !code) {
    return 0.0f;
  }
#if NBASE_SQ8_HAS_AVX2
  if (nbase_cpu_has_avx2_fma()) {
    return sq8_l2_sq_asymmetric_avx2(sq8, query, code);
  }
#endif
  float sum = 0.0f;
  for (uint32_t d = 0U; d < sq8->dim; d++) {
    float approx = sq8->mins[d] + (float)code[d] * sq8->scales[d];
    float diff = query[d] - approx;
    sum += diff * diff;
  }
  return sum;
}

float nbase_sq8_l2_sq_symmetric(const nbase_sq8_t *sq8, const uint8_t *code_a, const uint8_t *code_b) {
  if (!sq8 || !code_a || !code_b) {
    return 0.0f;
  }
#if NBASE_SQ8_HAS_AVX2
  if (nbase_cpu_has_avx2_fma()) {
    return sq8_l2_sq_symmetric_avx2(sq8, code_a, code_b);
  }
#endif
  float sum = 0.0f;
  for (uint32_t d = 0U; d < sq8->dim; d++) {
    float diff = ((float)code_a[d] - (float)code_b[d]) * sq8->scales[d];
    sum += diff * diff;
  }
  return sum;
}

nbase_status_t nbase_sq8_l2_sq_asymmetric_batch(
    const nbase_sq8_t *sq8,
    const float *query,
    const uint8_t *codes,
    uint32_t count,
    float *out_distances) {
  if (!sq8 || !query || (count > 0U && (!codes || !out_distances))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }
  if (!nbase_vec_is_finite(query, sq8->dim)) {
    return NBASE_ERR_INVALID_VECTOR;
  }

  float stack_buf[512] = {0.0f};
  float *q_shifted = (sq8->dim <= 512U) ? stack_buf : malloc((size_t)sq8->dim * sizeof(float));
  if (!q_shifted) {
    return NBASE_ERR_NOMEM;
  }
  for (uint32_t d = 0U; d < sq8->dim; d++) {
    q_shifted[d] = query[d] - sq8->mins[d];
  }

#if NBASE_SQ8_HAS_AVX2
  if (nbase_cpu_has_avx2_fma()) {
    sq8_l2_sq_asymmetric_batch_avx2(q_shifted, sq8->scales, sq8->dim, codes, count, out_distances);
    if (q_shifted != stack_buf) {
      free(q_shifted);
    }
    return NBASE_OK;
  }
#endif

  for (uint32_t i = 0U; i < count; i++) {
    const uint8_t *code = codes + (size_t)i * (size_t)sq8->dim;
    float sum = 0.0f;
    for (uint32_t d = 0U; d < sq8->dim; d++) {
      float diff = q_shifted[d] - (float)code[d] * sq8->scales[d];
      sum += diff * diff;
    }
    out_distances[i] = sum;
  }
  if (q_shifted != stack_buf) {
    free(q_shifted);
  }
  return NBASE_OK;
}

nbase_status_t nbase_sq8_l2_sq_symmetric_batch(
    const nbase_sq8_t *sq8,
    const uint8_t *query_code,
    const uint8_t *codes,
    uint32_t count,
    float *out_distances) {
  if (!sq8 || !query_code || (count > 0U && (!codes || !out_distances))) {
    return NBASE_ERR_INVALID_ARG;
  }
  if (count == 0U) {
    return NBASE_OK;
  }
  if ((size_t)count > SIZE_MAX / (size_t)sq8->dim) {
    return NBASE_ERR_NOMEM;
  }

#if NBASE_SQ8_HAS_AVX2
  if (nbase_cpu_has_avx2_fma()) {
    sq8_l2_sq_symmetric_batch_avx2(sq8, query_code, codes, count, out_distances);
    return NBASE_OK;
  }
#endif

  for (uint32_t i = 0U; i < count; i++) {
    const uint8_t *code = codes + (size_t)i * (size_t)sq8->dim;
    float sum = 0.0f;
    for (uint32_t d = 0U; d < sq8->dim; d++) {
      float diff = ((float)query_code[d] - (float)code[d]) * sq8->scales[d];
      sum += diff * diff;
    }
    out_distances[i] = sum;
  }
  return NBASE_OK;
}

nbase_status_t nbase_sq8_save(const nbase_sq8_t *sq8, const char *file_path) {
  if (!sq8 || !file_path || file_path[0] == '\0') {
    return NBASE_ERR_INVALID_ARG;
  }

  size_t arr_bytes = (size_t)sq8->dim * sizeof(float);
  nbase_io_chunk_t chunks[2] = {
      {sq8->mins, arr_bytes},
      {sq8->scales, arr_bytes},
  };
  uint32_t payload_crc = nbase_crc32_update(0U, sq8->mins, arr_bytes);
  payload_crc = nbase_crc32_update(payload_crc, sq8->scales, arr_bytes);

  nbsq_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, NBASE_NBSQ_MAGIC, 8);
  hdr.version = NBASE_NBSQ_VERSION;
  hdr.dim = sq8->dim;
  hdr.payload_bytes = (uint64_t)arr_bytes * 2ULL;
  hdr.payload_crc32 = payload_crc;
  hdr.header_crc32 = nbase_crc32(&hdr, offsetof(nbsq_header_t, header_crc32));

  return nbase_write_file_atomic(file_path, &hdr, sizeof(hdr), chunks, 2U);
}

nbase_status_t nbase_sq8_load(const char *file_path, nbase_sq8_t **out_sq8) {
  if (!file_path || file_path[0] == '\0' || !out_sq8) {
    return NBASE_ERR_INVALID_ARG;
  }
  *out_sq8 = NULL;

  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    return errno == ENOENT ? NBASE_ERR_NOT_FOUND : NBASE_ERR_IO;
  }

  struct stat st_buf;
  if (fstat(fd, &st_buf) != 0) {
    close(fd);
    return NBASE_ERR_IO;
  }
  if (st_buf.st_size < (off_t)sizeof(nbsq_header_t)) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  uint64_t file_size = (uint64_t)st_buf.st_size;

  nbsq_header_t hdr;
  nbase_status_t st = nbase_read_exact(fd, &hdr, sizeof(hdr));
  if (st != NBASE_OK) {
    close(fd);
    return st;
  }
  if (memcmp(hdr.magic, NBASE_NBSQ_MAGIC, 8) != 0 || hdr.version != NBASE_NBSQ_VERSION) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.header_crc32 != nbase_crc32(&hdr, offsetof(nbsq_header_t, header_crc32))) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }
  if (hdr.dim == 0U || hdr.dim > NBASE_MAX_DIM) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  uint64_t arr_bytes_u64 = (uint64_t)hdr.dim * (uint64_t)sizeof(float);
  if (hdr.payload_bytes != arr_bytes_u64 * 2ULL ||
      file_size != sizeof(nbsq_header_t) + hdr.payload_bytes ||
      arr_bytes_u64 > (uint64_t)SIZE_MAX) {
    close(fd);
    return NBASE_ERR_CORRUPT;
  }

  nbase_sq8_t *sq8 = calloc(1, sizeof(*sq8));
  float *mins = malloc((size_t)arr_bytes_u64);
  float *scales = malloc((size_t)arr_bytes_u64);
  float *inv_scales = malloc((size_t)arr_bytes_u64);
  if (!sq8 || !mins || !scales || !inv_scales) {
    close(fd);
    free(sq8);
    free(mins);
    free(scales);
    free(inv_scales);
    return NBASE_ERR_NOMEM;
  }

  st = nbase_read_exact(fd, mins, (size_t)arr_bytes_u64);
  if (st == NBASE_OK) {
    st = nbase_read_exact(fd, scales, (size_t)arr_bytes_u64);
  }
  close(fd);
  if (st != NBASE_OK) {
    free(sq8);
    free(mins);
    free(scales);
    free(inv_scales);
    return st;
  }

  uint32_t crc = nbase_crc32_update(0U, mins, (size_t)arr_bytes_u64);
  crc = nbase_crc32_update(crc, scales, (size_t)arr_bytes_u64);
  if (crc != hdr.payload_crc32 || !nbase_vec_is_finite(mins, hdr.dim) || !nbase_vec_is_finite(scales, hdr.dim)) {
    free(sq8);
    free(mins);
    free(scales);
    free(inv_scales);
    return NBASE_ERR_CORRUPT;
  }

  for (uint32_t d = 0U; d < hdr.dim; d++) {
    if (scales[d] < 0.0f) {
      free(sq8);
      free(mins);
      free(scales);
      free(inv_scales);
      return NBASE_ERR_CORRUPT;
    }
    inv_scales[d] = (scales[d] > 0.0f) ? (1.0f / scales[d]) : 0.0f;
  }

  sq8->dim = hdr.dim;
  sq8->mins = mins;
  sq8->scales = scales;
  sq8->inv_scales = inv_scales;
  *out_sq8 = sq8;
  return NBASE_OK;
}
