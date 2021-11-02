#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

INLINE void UpdateAndApplyGradient(float* v, Gradient* grad) {
  grad->epoch++;

  if (!grad->g)
    return;

  grad->M = BETA1 * grad->M + (1.0 - BETA1) * grad->g;
  grad->V = BETA2 * grad->V + (1.0 - BETA2) * grad->g * grad->g;

  float mHat = grad->M / (1 - pow(BETA1, grad->epoch));
  float vHat = grad->V / (1 - pow(BETA2, grad->epoch));
  float delta = ALPHA * mHat / (sqrt(vHat) + EPSILON);

  *v -= delta;

  grad->g = 0;
}

INLINE void ApplyGradients(NN* nn, NNGradients* g) {
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->featureWeights[i], &g->featureWeightGradients[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->hiddenBiases[i], &g->hiddenBiasGradients[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN * 2; i++)
    UpdateAndApplyGradient(&nn->hiddenWeights[i], &g->hiddenWeightGradients[i]);

  UpdateAndApplyGradient(&nn->outputBias, &g->outputBiasGradient);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_FEATURES; i++)
    UpdateAndApplyGradient(&nn->skipWeights[i], &g->skipWeightGradients[i]);
}

INLINE void ClearGradients(NNGradients* gradients) {
  memset(gradients->featureWeightGradients, 0, sizeof(gradients->featureWeightGradients));
  memset(gradients->hiddenBiasGradients, 0, sizeof(gradients->hiddenBiasGradients));
  memset(gradients->hiddenWeightGradients, 0, sizeof(gradients->hiddenWeightGradients));
  memset(gradients->skipWeightGradients, 0, sizeof(gradients->skipWeightGradients));
  gradients->outputBiasGradient = (Gradient){.epoch = 0, .g = 0.0f, .M = 0.0f, .V = 0.0f};
}

#endif