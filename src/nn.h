#ifndef NN_H
#define NN_H

#include <immintrin.h>

#include "types.h"
#include "util.h"

void NNPredict(NN* nn, Features* f, Color stm, NNAccumulators* results);

NN* LoadNN(char* path);
NN* LoadRandomNN();
void SaveNN(NN* nn, char* path);

INLINE void ReLU(float* v, const size_t n) {
  const size_t width = sizeof(__m256) / sizeof(float);
  const size_t chunks =  n / width;

  const __m256 zero = _mm256_setzero_ps();

  __m256* vector = (__m256*) v;

  for (size_t j = 0; j < chunks; j++)
    vector[j] = _mm256_max_ps(zero, vector[j]);
}

INLINE void CReLU(float* v, const size_t n) {
  const size_t width = sizeof(__m256) / sizeof(float);
  const size_t chunks =  n / width;

  const __m256 zero = _mm256_setzero_ps();
  const __m256 one = _mm256_set1_ps(1.0f);

  __m256* vector = (__m256*) v;

  for (size_t j = 0; j < chunks; j++)
    vector[j] = _mm256_min_ps(one, _mm256_max_ps(zero, vector[j]));
}

INLINE float DotProduct(float* v1, float* v2, const size_t n) {
  const size_t width = sizeof(__m256) / sizeof(float);
  const size_t chunks =  n / width;

  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();

  __m256* vector1 = (__m256*) v1;
  __m256* vector2 = (__m256*) v2;
  
  for (size_t j = 0; j < chunks; j += 2) {
    s0 = _mm256_add_ps(_mm256_mul_ps(vector1[j], vector2[j]), s0);
    s1 = _mm256_add_ps(_mm256_mul_ps(vector1[j + 1], vector2[j + 1]), s1);
  }

  const __m256 r8 = _mm256_add_ps(s0, s1);
  const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
  const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
  const __m128 r1 = _mm_add_ss(r2, _mm_shuffle_ps(r2, r2, 0x1));
  return _mm_cvtss_f32(r1);
}

#endif