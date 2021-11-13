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

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN * 2; i++)
    UpdateAndApplyGradient(&nn->outputWeights[i], &g->outputWeights[i]);

  UpdateAndApplyGradient(&nn->outputBias, &g->outputBias);
}

INLINE void ClearGradients(NNGradients* gradients) {
  memset(gradients->inputWeights, 0, sizeof(gradients->inputWeights));
  memset(gradients->inputBiases, 0, sizeof(gradients->inputBiases));

  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  gradients->outputBias = (Gradient){.g = 0.0f, .M = 0.0f, .V = 0.0f};
}

#endif