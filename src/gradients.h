#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

void UpdateAndApplyGradient(float* v, Gradient* grad) {
  if (!grad->g) return;

  grad->M = BETA1 * grad->M + (1.0 - BETA1) * grad->g;
  grad->V = BETA2 * grad->V + (1.0 - BETA2) * grad->g * grad->g;
  float delta = ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);

  *v -= delta;

  grad->g = 0;
}

void ApplyGradients(NN* nn, NNGradients* g) {
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_INPUT * N_HIDDEN; i++) UpdateAndApplyGradient(&nn->inputWeights[i], &g->inputWeights[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) UpdateAndApplyGradient(&nn->inputBiases[i], &g->inputBiases[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < 2 * N_HIDDEN * N_HIDDEN_2; i++) UpdateAndApplyGradient(&nn->l1Weights[i], &g->l1Weights[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_2; i++) UpdateAndApplyGradient(&nn->l1Biases[i], &g->l1Biases[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_2 * N_HIDDEN_3; i++) UpdateAndApplyGradient(&nn->l2Weights[i], &g->l2Weights[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_3; i++) UpdateAndApplyGradient(&nn->l2Biases[i], &g->l2Biases[i]);

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_3; i++) UpdateAndApplyGradient(&nn->outputWeights[i], &g->outputWeights[i]);

  UpdateAndApplyGradient(&nn->outputBias, &g->outputBias);
}

void ClearGradients(NNGradients* gradients) {
  memset(gradients->inputWeights, 0, sizeof(gradients->inputWeights));
  memset(gradients->inputBiases, 0, sizeof(gradients->inputBiases));

  memset(gradients->l1Weights, 0, sizeof(gradients->l1Weights));
  memset(gradients->l1Biases, 0, sizeof(gradients->l1Biases));

  memset(gradients->l2Weights, 0, sizeof(gradients->l2Weights));
  memset(gradients->l2Biases, 0, sizeof(gradients->l2Biases));

  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(&gradients->outputBias, 0, sizeof(gradients->outputBias));
}

#endif