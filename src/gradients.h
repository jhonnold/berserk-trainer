#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <immintrin.h>
#include <string.h>

#include "types.h"
#include "util.h"

void UpdateAndApplyGradientSingle(float* v, float* momentum, float* velocity, float g) {
  *momentum = BETA1 * *momentum + (1.0 - BETA1) * g;
  *velocity = BETA2 * *velocity + (1.0 - BETA2) * g * g;

  *v -= ALPHA * *momentum / (sqrtf(*velocity) + EPSILON);
}

void UpdateAndApplyGradient(__m256* values, __m256* momentums, __m256* velocities, __m256 gradients) {
  const __m256 lr = _mm256_set1_ps(ALPHA);
  const __m256 epsilon = _mm256_set1_ps(EPSILON);
  const __m256 beta1 = _mm256_set1_ps(BETA1);
  const __m256 beta2 = _mm256_set1_ps(BETA2);
  const __m256 oneMinusBeta1 = _mm256_set1_ps(1.0 - BETA1);
  const __m256 oneMinusBeta2 = _mm256_set1_ps(1.0 - BETA2);

  *momentums = _mm256_add_ps(_mm256_mul_ps(beta1, *momentums), _mm256_mul_ps(oneMinusBeta1, gradients));
  *velocities = _mm256_add_ps(_mm256_mul_ps(beta2, *velocities), _mm256_mul_ps(oneMinusBeta2, _mm256_mul_ps(gradients, gradients)));

  *values = _mm256_sub_ps(*values, _mm256_div_ps(
    _mm256_mul_ps(lr, *momentums),
    _mm256_add_ps(epsilon, _mm256_sqrt_ps(*velocities))
  ));
}

void ApplyGradients(NN* nn, NNGradients* grads, BatchGradients* local) {
  const size_t WIDTH = sizeof(__m256) / sizeof(float);

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (size_t i = 0; i < N_INPUT * N_HIDDEN; i += WIDTH) {
    __m256* values = (__m256*)&nn->inputWeights[i];
    __m256* momentums = (__m256*)&grads->inputWeightsM[i];
    __m256* velocities = (__m256*)&grads->inputWeightsV[i];

    __m256 gradients = _mm256_setzero_ps();
    for (size_t t = 0; t < THREADS; t++)
      gradients = _mm256_add_ps(gradients, *(__m256*)&local[t].inputWeights[i]);
    
    UpdateAndApplyGradient(values, momentums, velocities, gradients);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (size_t i = 0; i < N_HIDDEN; i += WIDTH) {
    __m256* values = (__m256*)&nn->inputBiases[i];
    __m256* momentums = (__m256*)&grads->inputBiasesM[i];
    __m256* velocities = (__m256*)&grads->inputBiasesV[i];

    __m256 gradients = _mm256_setzero_ps();
    for (size_t t = 0; t < THREADS; t++)
      gradients = _mm256_add_ps(gradients, *(__m256*)&local[t].inputBiases[i]);

    UpdateAndApplyGradient(values, momentums, velocities, gradients);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (size_t i = 0; i < N_L1; i += WIDTH) {
    __m256* values = (__m256*)&nn->outputWeights[i];
    __m256* momentums = (__m256*)&grads->outputWeightsM[i];
    __m256* velocities = (__m256*)&grads->outputWeightsV[i];

    __m256 gradients = _mm256_setzero_ps();
    for (size_t t = 0; t < THREADS; t++)
      gradients = _mm256_add_ps(gradients, *(__m256*)&local[t].outputWeights[i]);

    UpdateAndApplyGradient(values, momentums, velocities, gradients);
  }

  float g = 0;
  for (int t = 0; t < THREADS; t++) g += local[t].outputBias;

  UpdateAndApplyGradientSingle(&nn->outputBias, &grads->outputBiasM, &grads->outputBiasV, g);
}

void ClearGradients(NNGradients* gradients) {
  memset(gradients->inputWeightsM, 0, sizeof(gradients->inputWeightsM));
  memset(gradients->inputWeightsV, 0, sizeof(gradients->inputWeightsV));
  memset(gradients->inputBiasesM, 0, sizeof(gradients->inputBiasesM));
  memset(gradients->inputBiasesV, 0, sizeof(gradients->inputBiasesV));

  memset(gradients->outputWeightsM, 0, sizeof(gradients->outputWeightsM));
  memset(gradients->outputWeightsV, 0, sizeof(gradients->outputWeightsV));
  memset(&gradients->outputBiasM, 0, sizeof(gradients->outputBiasM));
  memset(&gradients->outputBiasV, 0, sizeof(gradients->outputBiasV));
}

#endif