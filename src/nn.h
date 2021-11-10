#ifndef NN_H
#define NN_H

#include <immintrin.h>

#include "types.h"
#include "util.h"

void NNPredict(NN* nn, Features* f, Color stm, NNAccumulators* results);

NN* LoadNN(char* path);
NN* LoadRandomNN();
void SaveNN(NN* nn, char* path);

INLINE void ReLU(float* v, size_t n) {
  const __m256 zero = _mm256_setzero_ps();

  for (size_t j = 0; j < n; j += sizeof(__m256) / sizeof(float))
    _mm256_store_ps(v + j, _mm256_max_ps(zero, _mm256_load_ps(v + j)));
}

INLINE float DotProduct(float* v1, float* v2, size_t n) {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();

  const size_t step = sizeof(__m256) / sizeof(float);

  for (size_t j = 0; j < n; j += 2 * step) {
    const __m256 a0 = _mm256_load_ps(v1 + j);
    const __m256 b0 = _mm256_load_ps(v2 + j);
    s0 = _mm256_add_ps(s0, _mm256_mul_ps(a0, b0));

    const __m256 a1 = _mm256_load_ps(v1 + j + step);
    const __m256 b1 = _mm256_load_ps(v2 + j + step);
    s1 = _mm256_add_ps(s0, _mm256_mul_ps(a1, b1));
  }

  const __m256 r8 = _mm256_add_ps(s0, s1);
  const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
  const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
  const __m128 r1 = _mm_add_ss(r2, _mm_shuffle_ps(r2, r2, 0x1));
  return _mm_cvtss_f32(r1);
}

#endif