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
  for (int i = 0; i < N_FEATURES; i++)
    UpdateAndApplyGradient(&nn->skipWeights[i], &g->skipWeightGradients[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->inputWeights[i], &g->inputWeightGradients[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->inputBiases[i], &g->inputBiasGradients[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < 2 * N_HIDDEN * N_HIDDEN_2; i++)
    UpdateAndApplyGradient(&nn->hiddenWeights[i], &g->hiddenWeightGradients[i]);

  for (int i = 0; i < N_HIDDEN_2; i++)
    UpdateAndApplyGradient(&nn->hiddenBiases[i], &g->hiddenBiasGradients[i]);

  for (int i = 0; i < N_HIDDEN_2; i++)
    UpdateAndApplyGradient(&nn->outputWeights[i], &g->outputWeightGradients[i]);

  UpdateAndApplyGradient(&nn->outputBias, &g->outputBiasGradient);
}

INLINE void ClearGradients(NNGradients* gradients) {
  memset(gradients->skipWeightGradients, 0, sizeof(gradients->skipWeightGradients));

  memset(gradients->inputWeightGradients, 0, sizeof(gradients->inputWeightGradients));
  memset(gradients->inputBiasGradients, 0, sizeof(gradients->inputBiasGradients));

  memset(gradients->hiddenWeightGradients, 0, sizeof(gradients->hiddenWeightGradients));
  memset(gradients->hiddenBiasGradients, 0, sizeof(gradients->hiddenBiasGradients));

  memset(gradients->outputWeightGradients, 0, sizeof(gradients->outputWeightGradients));
  gradients->outputBiasGradient = (Gradient){.g = 0.0f, .M = 0.0f, .V = 0.0f};
}

#endif