#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

void UpdateAndApplyGradient(float* v, Gradient* grad) {
  grad->M = BETA1 * grad->M + (1.0 - BETA1) * grad->g;
  grad->V = BETA2 * grad->V + (1.0 - BETA2) * grad->g * grad->g;
  float delta = ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);

  if (isnan(delta))
    grad->M = grad->V = 0.0;
  else
    *v -= delta;

  grad->g = 0;
}

void ApplyGradients(NN* nn, NNGradients* g) {
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_INPUT * N_HIDDEN; i++) UpdateAndApplyGradient(&nn->inputWeights[i], &g->inputWeights[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) UpdateAndApplyGradient(&nn->inputBiases[i], &g->inputBiases[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN * 2; i++) UpdateAndApplyGradient(&nn->outputWeights[i], &g->outputWeights[i]);

  UpdateAndApplyGradient(&nn->outputBias, &g->outputBias);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_P_INPUT * N_P_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->pawnInputWeights[i], &g->pawnInputWeights[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_P_HIDDEN; i++) UpdateAndApplyGradient(&nn->pawnInputBiases[i], &g->pawnInputBiases[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_P_HIDDEN * 2; i++) UpdateAndApplyGradient(&nn->pawnOutputWeights[i], &g->pawnOutputWeights[i]);

  UpdateAndApplyGradient(&nn->pawnOutputBias, &g->pawnOutputBias);
}

void ClearGradients(NNGradients* gradients) {
  memset(gradients->inputWeights, 0, sizeof(gradients->inputWeights));
  memset(gradients->inputBiases, 0, sizeof(gradients->inputBiases));
  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(&gradients->outputBias, 0, sizeof(gradients->outputBias));

  memset(gradients->pawnInputWeights, 0, sizeof(gradients->pawnInputWeights));
  memset(gradients->pawnInputBiases, 0, sizeof(gradients->pawnInputBiases));
  memset(gradients->pawnOutputWeights, 0, sizeof(gradients->pawnOutputWeights));
  memset(&gradients->pawnOutputBias, 0, sizeof(gradients->pawnOutputBias));
}

#endif