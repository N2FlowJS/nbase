#include "nbase_internal.h"

#include <math.h>
#include <stdatomic.h>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define NBASE_HAS_X86_AVX2 1
#else
#define NBASE_HAS_X86_AVX2 0
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define NBASE_HAS_ARM_NEON 1
#else
#define NBASE_HAS_ARM_NEON 0
#endif

static float squared_euclidean_scalar(const float *a, const float *b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; i++) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

static float dot_scalar(const float *a, const float *b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

static float cosine_scalar(const float *a, const float *b, uint32_t dim) {
  float dot = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;
  for (uint32_t i = 0; i < dim; i++) {
    float ai = a[i];
    float bi = b[i];
    dot += ai * bi;
    norm_a += ai * ai;
    norm_b += bi * bi;
  }
  float denom = sqrtf(norm_a) * sqrtf(norm_b);
  if (denom == 0.0f) {
    return 1.0f;
  }
  float sim = dot / denom;
  if (sim > 1.0f) {
    sim = 1.0f;
  } else if (sim < -1.0f) {
    sim = -1.0f;
  }
  return 1.0f - sim;
}

static float manhattan_scalar(const float *a, const float *b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; i++) {
    sum += fabsf(a[i] - b[i]);
  }
  return sum;
}

static float chebyshev_scalar(const float *a, const float *b, uint32_t dim) {
  float max_diff = 0.0f;
  for (uint32_t i = 0; i < dim; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > max_diff) {
      max_diff = d;
    }
  }
  return max_diff;
}

static float hamming_scalar(const float *a, const float *b, uint32_t dim) {
  uint32_t count = 0;
  for (uint32_t i = 0; i < dim; i++) {
    if (a[i] != b[i]) {
      count++;
    }
  }
  return (float)count;
}

static float euclidean_scalar(const float *a, const float *b, uint32_t dim) {
  return sqrtf(squared_euclidean_scalar(a, b, dim));
}

static float inner_product_scalar(const float *a, const float *b, uint32_t dim) {
  return -dot_scalar(a, b, dim);
}

static nbase_dist_fn get_scalar_distance_fn(nbase_metric_t metric) {
  switch (metric) {
    case NBASE_METRIC_EUCLIDEAN:
      return euclidean_scalar;
    case NBASE_METRIC_SQUARED_EUCLIDEAN:
      return squared_euclidean_scalar;
    case NBASE_METRIC_COSINE:
      return cosine_scalar;
    case NBASE_METRIC_INNER_PRODUCT:
      return inner_product_scalar;
    case NBASE_METRIC_MANHATTAN:
      return manhattan_scalar;
    case NBASE_METRIC_CHEBYSHEV:
      return chebyshev_scalar;
    case NBASE_METRIC_HAMMING:
      return hamming_scalar;
    default:
      return euclidean_scalar;
  }
}

float nbase_distance_scalar(nbase_metric_t metric, const float *a, const float *b, uint32_t dim) {
  if (a == NULL || b == NULL) {
    return 0.0f;
  }
  return get_scalar_distance_fn(metric)(a, b, dim);
}

#if NBASE_HAS_X86_AVX2

static inline __attribute__((target("avx2,fma"))) float hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(s);
  s = _mm_add_ps(s, shuf);
  shuf = _mm_movehl_ps(shuf, s);
  s = _mm_add_ss(s, shuf);
  return _mm_cvtss_f32(s);
}

static inline __attribute__((target("avx2,fma"))) float hmax256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 m = _mm_max_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(m);
  m = _mm_max_ps(m, shuf);
  shuf = _mm_movehl_ps(shuf, m);
  m = _mm_max_ss(m, shuf);
  return _mm_cvtss_f32(m);
}

static __attribute__((target("avx2,fma"))) float squared_euclidean_avx2(const float *a, const float *b, uint32_t dim) {
  __m256 acc = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 8U <= dim; i += 8U) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 d = _mm256_sub_ps(va, vb);
    acc = _mm256_fmadd_ps(d, d, acc);
  }
  float sum = hsum256_ps(acc);
  for (; i < dim; i++) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

static __attribute__((target("avx2,fma"))) float dot_avx2(const float *a, const float *b, uint32_t dim) {
  __m256 acc = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 8U <= dim; i += 8U) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    acc = _mm256_fmadd_ps(va, vb, acc);
  }
  float sum = hsum256_ps(acc);
  for (; i < dim; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

static __attribute__((target("avx2,fma"))) float cosine_avx2(const float *a, const float *b, uint32_t dim) {
  __m256 acc_dot = _mm256_setzero_ps();
  __m256 acc_na = _mm256_setzero_ps();
  __m256 acc_nb = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 8U <= dim; i += 8U) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    acc_dot = _mm256_fmadd_ps(va, vb, acc_dot);
    acc_na = _mm256_fmadd_ps(va, va, acc_na);
    acc_nb = _mm256_fmadd_ps(vb, vb, acc_nb);
  }
  float dot = hsum256_ps(acc_dot);
  float norm_a = hsum256_ps(acc_na);
  float norm_b = hsum256_ps(acc_nb);
  for (; i < dim; i++) {
    float ai = a[i];
    float bi = b[i];
    dot += ai * bi;
    norm_a += ai * ai;
    norm_b += bi * bi;
  }
  float denom = sqrtf(norm_a) * sqrtf(norm_b);
  if (denom == 0.0f) {
    return 1.0f;
  }
  float sim = dot / denom;
  if (sim > 1.0f) {
    sim = 1.0f;
  } else if (sim < -1.0f) {
    sim = -1.0f;
  }
  return 1.0f - sim;
}

static __attribute__((target("avx2,fma"))) float manhattan_avx2(const float *a, const float *b, uint32_t dim) {
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  __m256 acc = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 8U <= dim; i += 8U) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 d = _mm256_andnot_ps(sign_mask, _mm256_sub_ps(va, vb));
    acc = _mm256_add_ps(acc, d);
  }
  float sum = hsum256_ps(acc);
  for (; i < dim; i++) {
    sum += fabsf(a[i] - b[i]);
  }
  return sum;
}

static __attribute__((target("avx2,fma"))) float chebyshev_avx2(const float *a, const float *b, uint32_t dim) {
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  __m256 vmax = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 8U <= dim; i += 8U) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 d = _mm256_andnot_ps(sign_mask, _mm256_sub_ps(va, vb));
    vmax = _mm256_max_ps(vmax, d);
  }
  float max_diff = hmax256_ps(vmax);
  for (; i < dim; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > max_diff) {
      max_diff = d;
    }
  }
  return max_diff;
}

static __attribute__((target("avx2,fma"))) float hamming_avx2(const float *a, const float *b, uint32_t dim) {
  uint32_t count = 0;
  uint32_t i = 0;
  for (; i + 8U <= dim; i += 8U) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_NEQ_UQ);
    int mask = _mm256_movemask_ps(cmp);
    count += (uint32_t)__builtin_popcount((unsigned int)mask);
  }
  for (; i < dim; i++) {
    if (a[i] != b[i]) {
      count++;
    }
  }
  return (float)count;
}

static __attribute__((target("avx2,fma"))) float euclidean_avx2(const float *a, const float *b, uint32_t dim) {
  return sqrtf(squared_euclidean_avx2(a, b, dim));
}

static __attribute__((target("avx2,fma"))) float inner_product_avx2(const float *a, const float *b, uint32_t dim) {
  return -dot_avx2(a, b, dim);
}

static int detect_avx2_fma(void) {
  static atomic_int cached = -1;
  int val = atomic_load_explicit(&cached, memory_order_acquire);
  if (val >= 0) {
    return val;
  }
  int supported = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? 1 : 0;
  atomic_store_explicit(&cached, supported, memory_order_release);
  return supported;
}

#endif /* NBASE_HAS_X86_AVX2 */

#if NBASE_HAS_ARM_NEON

static float squared_euclidean_neon(const float *a, const float *b, uint32_t dim) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + 4U <= dim; i += 4U) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    float32x4_t d = vsubq_f32(va, vb);
    acc = vmlaq_f32(acc, d, d);
  }
  float sum = vaddvq_f32(acc);
  for (; i < dim; i++) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

static float dot_neon(const float *a, const float *b, uint32_t dim) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + 4U <= dim; i += 4U) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    acc = vmlaq_f32(acc, va, vb);
  }
  float sum = vaddvq_f32(acc);
  for (; i < dim; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

static float cosine_neon(const float *a, const float *b, uint32_t dim) {
  float32x4_t acc_dot = vdupq_n_f32(0.0f);
  float32x4_t acc_na = vdupq_n_f32(0.0f);
  float32x4_t acc_nb = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + 4U <= dim; i += 4U) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    acc_dot = vmlaq_f32(acc_dot, va, vb);
    acc_na = vmlaq_f32(acc_na, va, va);
    acc_nb = vmlaq_f32(acc_nb, vb, vb);
  }
  float dot = vaddvq_f32(acc_dot);
  float norm_a = vaddvq_f32(acc_na);
  float norm_b = vaddvq_f32(acc_nb);
  for (; i < dim; i++) {
    float ai = a[i];
    float bi = b[i];
    dot += ai * bi;
    norm_a += ai * ai;
    norm_b += bi * bi;
  }
  float denom = sqrtf(norm_a) * sqrtf(norm_b);
  if (denom == 0.0f) {
    return 1.0f;
  }
  float sim = dot / denom;
  if (sim > 1.0f) {
    sim = 1.0f;
  } else if (sim < -1.0f) {
    sim = -1.0f;
  }
  return 1.0f - sim;
}

static float manhattan_neon(const float *a, const float *b, uint32_t dim) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + 4U <= dim; i += 4U) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    acc = vaddq_f32(acc, vabdq_f32(va, vb));
  }
  float sum = vaddvq_f32(acc);
  for (; i < dim; i++) {
    sum += fabsf(a[i] - b[i]);
  }
  return sum;
}

static float chebyshev_neon(const float *a, const float *b, uint32_t dim) {
  float32x4_t vmax = vdupq_n_f32(0.0f);
  uint32_t i = 0;
  for (; i + 4U <= dim; i += 4U) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    vmax = vmaxq_f32(vmax, vabdq_f32(va, vb));
  }
  float max_diff = vmaxvq_f32(vmax);
  for (; i < dim; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > max_diff) {
      max_diff = d;
    }
  }
  return max_diff;
}

static float hamming_neon(const float *a, const float *b, uint32_t dim) {
  uint32_t count = 0;
  uint32_t i = 0;
  for (; i + 4U <= dim; i += 4U) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    uint32x4_t eq = vceqq_f32(va, vb);
    uint32_t lanes[4];
    vst1q_u32(lanes, eq);
    count += (lanes[0] == 0U) + (lanes[1] == 0U) + (lanes[2] == 0U) + (lanes[3] == 0U);
  }
  for (; i < dim; i++) {
    if (a[i] != b[i]) {
      count++;
    }
  }
  return (float)count;
}

static float euclidean_neon(const float *a, const float *b, uint32_t dim) {
  return sqrtf(squared_euclidean_neon(a, b, dim));
}

static float inner_product_neon(const float *a, const float *b, uint32_t dim) {
  return -dot_neon(a, b, dim);
}

#endif /* NBASE_HAS_ARM_NEON */

const char *nbase_simd_backend(void) {
#if NBASE_HAS_X86_AVX2
  if (detect_avx2_fma() != 0) {
    return "avx2+fma";
  }
#elif NBASE_HAS_ARM_NEON
  return "neon";
#endif
  return "scalar";
}

bool nbase_cpu_has_avx2_fma(void) {
#if NBASE_HAS_X86_AVX2
  return detect_avx2_fma() != 0;
#else
  return false;
#endif
}

nbase_dist_fn nbase_get_distance_fn(nbase_metric_t metric) {
#if NBASE_HAS_X86_AVX2
  if (detect_avx2_fma() != 0) {
    switch (metric) {
      case NBASE_METRIC_EUCLIDEAN:
        return euclidean_avx2;
      case NBASE_METRIC_SQUARED_EUCLIDEAN:
        return squared_euclidean_avx2;
      case NBASE_METRIC_COSINE:
        return cosine_avx2;
      case NBASE_METRIC_INNER_PRODUCT:
        return inner_product_avx2;
      case NBASE_METRIC_MANHATTAN:
        return manhattan_avx2;
      case NBASE_METRIC_CHEBYSHEV:
        return chebyshev_avx2;
      case NBASE_METRIC_HAMMING:
        return hamming_avx2;
      default:
        return euclidean_avx2;
    }
  }
#elif NBASE_HAS_ARM_NEON
  switch (metric) {
    case NBASE_METRIC_EUCLIDEAN:
      return euclidean_neon;
    case NBASE_METRIC_SQUARED_EUCLIDEAN:
      return squared_euclidean_neon;
    case NBASE_METRIC_COSINE:
      return cosine_neon;
    case NBASE_METRIC_INNER_PRODUCT:
      return inner_product_neon;
    case NBASE_METRIC_MANHATTAN:
      return manhattan_neon;
    case NBASE_METRIC_CHEBYSHEV:
      return chebyshev_neon;
    case NBASE_METRIC_HAMMING:
      return hamming_neon;
    default:
      return euclidean_neon;
  }
#endif
  return get_scalar_distance_fn(metric);
}

float nbase_distance(nbase_metric_t metric, const float *a, const float *b, uint32_t dim) {
  if (a == NULL || b == NULL) {
    return 0.0f;
  }
  return nbase_get_distance_fn(metric)(a, b, dim);
}
