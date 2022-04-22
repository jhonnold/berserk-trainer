#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <immintrin.h>
#include <string.h>

#include "types.h"
#include "util.h"

void SingleUpdateAndApplyGradient(float* val, float* momentum, float* velocity, float g) {
  *momentum = BETA1 * *momentum + (1.0 - BETA1) * g;
  *velocity = BETA2 * *velocity + (1.0 - BETA2) * g * g;

  float delta = ALPHA * *momentum / (sqrtf(*velocity) + EPSILON);

  *val -= delta;
}

void UpdateAndApplyGradient(__m256* values, __m256* momentums, __m256* velocities, __m256 gradients) {
  const __m256 a = _mm256_set1_ps(ALPHA);
  const __m256 b1 = _mm256_set1_ps(BETA1);
  const __m256 b2 = _mm256_set1_ps(BETA2);
  const __m256 e = _mm256_set1_ps(EPSILON);
  const __m256 mb1 = _mm256_sub_ps(_mm256_set1_ps(1.0), b1);
  const __m256 mb2 = _mm256_sub_ps(_mm256_set1_ps(1.0), b2);

  *momentums = _mm256_mul_ps(*momentums, b1);
  *momentums = _mm256_add_ps(*momentums, _mm256_mul_ps(mb1, gradients));

  *velocities = _mm256_mul_ps(*velocities, b2);
  *velocities = _mm256_add_ps(*velocities, _mm256_mul_ps(mb2, _mm256_mul_ps(gradients, gradients)));

  __m256 denom = _mm256_add_ps(e, _mm256_sqrt_ps(*velocities));
  __m256 delta = _mm256_div_ps(_mm256_mul_ps(a, *momentums), denom);

  *values = _mm256_sub_ps(*values, delta);
}

void ApplyGradients(NN* nn, Optimizer* optimizer, Gradients* gradients) {
  const size_t WIDTH = sizeof(__m256) / sizeof(float);

  __m256* networkInputWeights = (__m256*)nn->inputWeights;
  __m256* optimizerMInputWeights = (__m256*)optimizer->mInputWeights;
  __m256* optimizerVInputWeights = (__m256*)optimizer->vInputWeights;

  __m256* networkInputBiases = (__m256*)nn->inputBiases;
  __m256* optimizerMInputBiases = (__m256*)optimizer->mInputBiases;
  __m256* optimizerVInputBiases = (__m256*)optimizer->vInputBiases;

  __m256* networkOutputWeights = (__m256*)nn->outputWeights;
  __m256* optimizerMOutputWeights = (__m256*)optimizer->mOutputWeights;
  __m256* optimizerVOutputWeights = (__m256*)optimizer->vOutputWeights;

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (size_t i = 0; i < N_INPUT * N_HIDDEN / WIDTH; i++) {
    __m256 g = _mm256_setzero_ps();
    for (int k = 0; k < THREADS; k++) {
      __m256* weights = (__m256*)gradients[k].inputWeights;
      g = _mm256_add_ps(g, weights[i]);
    }

    UpdateAndApplyGradient(&networkInputWeights[i], &optimizerMInputWeights[i], &optimizerVInputWeights[i], g);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (size_t i = 0; i < N_HIDDEN / WIDTH; i++) {
    __m256 g = _mm256_setzero_ps();
    for (int k = 0; k < THREADS; k++) {
      __m256* biases = (__m256*)gradients[k].inputBiases;
      g = _mm256_add_ps(g, biases[i]);
    }

    UpdateAndApplyGradient(&networkInputBiases[i], &optimizerMInputBiases[i], &optimizerVInputBiases[i], g);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (size_t i = 0; i < 2 * N_HIDDEN / WIDTH; i++) {
    __m256 g = _mm256_setzero_ps();
    for (int k = 0; k < THREADS; k++) {
      __m256* weights = (__m256*)gradients[k].outputWeights;
      g = _mm256_add_ps(g, weights[i]);
    }

    UpdateAndApplyGradient(&networkOutputWeights[i], &optimizerMOutputWeights[i], &optimizerVOutputWeights[i], g);
  }

  float g = 0.0;
  for (int k = 0; k < THREADS; k++) g += gradients[k].outputBias;

  SingleUpdateAndApplyGradient(&nn->outputBias, &optimizer->mOutputBias, &optimizer->vOutputBias, g);
}

#endif