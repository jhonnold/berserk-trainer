#ifndef NN_H
#define NN_H

#include <immintrin.h>

#include "types.h"
#include "util.h"

void NNFirstLayer(NN* nn, Board* board, NNActivations* results);
void NNPredict(NN* nn, Board* board, NNActivations* results);

NN* LoadNN(char* path);
NN* LoadRandomNN();
void SaveNN(NN* nn, char* path);

INLINE void ReLU(float* v, size_t n) {
  const __m256 zero = _mm256_setzero_ps();

  for (size_t j = 0; j < n; j += sizeof(__m256) / sizeof(float))
    _mm256_store_ps(v + j, _mm256_max_ps(zero, _mm256_load_ps(v + j)));
}

INLINE float DotProduct(float* v1, float* v2, size_t n) {
  __m256 r8 = _mm256_setzero_ps();
  for (size_t j = 0; j < n; j += sizeof(__m256) / sizeof(float)) {
    const __m256 a = _mm256_load_ps(v1 + j);
    const __m256 b = _mm256_load_ps(v2 + j);
    r8 = _mm256_add_ps(r8, _mm256_mul_ps(a, b));
  }

  const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
  const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
  const __m128 r1 = _mm_add_ss(r2, _mm_shuffle_ps(r2, r2, 0x1));
  return _mm_cvtss_f32(r1);
}

#endif