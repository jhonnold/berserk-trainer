#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

INLINE void UpdateAndApplyGradient(float* v, Gradient* grad) {
  if (!grad->g)
    return;

  grad->M = BETA1 * grad->M + (1.0 - BETA1) * grad->g;
  grad->V = BETA2 * grad->V + (1.0 - BETA2) * grad->g * grad->g;
  float delta = ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);

  *v -= delta;

  grad->g = 0;
}

INLINE void ApplyGradients(NN* nn, NNGradients* g) {
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_INPUT * N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->inputWeights[i], &g->inputWeights[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->inputBiases[i], &g->inputBiases[i]);

#pragma omp parallel for schedule(auto) num_threads(N_BUCKETS)
  for (int j = 0; j < N_BUCKETS; j++) {
    for (int i = 0; i < N_HIDDEN * 2; i++)
      UpdateAndApplyGradient(&nn->outputWeights[j][i], &g->outputWeights[j][i]);

    UpdateAndApplyGradient(&nn->outputBias[j], &g->outputBias[j]);

    for (int i = 0; i < N_INPUT; i++)
      UpdateAndApplyGradient(&nn->skipWeights[j][i], &g->skipWeights[j][i]);
  }
}

INLINE void ClearGradients(NNGradients* gradients) {
  memset(gradients->inputWeights, 0, sizeof(gradients->inputWeights));
  memset(gradients->inputBiases, 0, sizeof(gradients->inputBiases));
  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(gradients->outputBias, 0, sizeof(gradients->outputBias));
  memset(gradients->skipWeights, 0, sizeof(gradients->skipWeights));
}

#endif