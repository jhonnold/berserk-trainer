#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

void UpdateAndApplyGradient(float* v, Gradient* grad, float g) {
  if (!g) return;

  grad->M = BETA1 * grad->M + (1.0 - BETA1) * g;
  grad->V = BETA2 * grad->V + (1.0 - BETA2) * g * g;
  float delta = ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);

  *v -= delta;
}

void ApplyGradients(NN* nn, NNGradients* grads, BatchGradients* local) {
#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_INPUT * N_HIDDEN; i++) {
    float g = 0.0;
    for (int t = 0; t < THREADS; t++) g += local[t].inputWeights[i];

    UpdateAndApplyGradient(&nn->inputWeights[i], &grads->inputWeights[i], g);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) {
    float g = 0.0;
    for (int t = 0; t < THREADS; t++) g += local[t].inputBiases[i];

    UpdateAndApplyGradient(&nn->inputBiases[i], &grads->inputBiases[i], g);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN * 2; i++) {
    float g = 0.0;
    for (int t = 0; t < THREADS; t++) g += local[t].outputWeights[i];

    UpdateAndApplyGradient(&nn->outputWeights[i], &grads->outputWeights[i], g);
  }

  float g = 0.0;
  for (int t = 0; t < THREADS; t++) g += local[t].outputBias;

  UpdateAndApplyGradient(&nn->outputBias, &grads->outputBias, g);
}

void ClearGradients(NNGradients* gradients) {
  memset(gradients->inputWeights, 0, sizeof(gradients->inputWeights));
  memset(gradients->inputBiases, 0, sizeof(gradients->inputBiases));

  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(&gradients->outputBias, 0, sizeof(gradients->outputBias));
}

#endif