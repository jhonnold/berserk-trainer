#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

void UpdateAndApplyGradient(float* v, Gradient* grad, float g) {
  grad->M = BETA1 * grad->M + (1.0 - BETA1) * g;
  grad->V = BETA2 * grad->V + (1.0 - BETA2) * g * g;
  float delta = ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);

  *v -= delta;
}

void ApplyGradients(NN* nn, NNGradients* grads, BatchGradients* batches) {
#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_INPUT; i++) {
    for (int j = 0; j < N_HIDDEN; j++) {
      int idx = i * j;

      float g = 0.0;
      for (int k = 0; k < THREADS; k++) g += batches[k].inputWeights[idx];

      UpdateAndApplyGradient(&nn->inputWeights[idx], &grads->inputWeights[idx], g);
    }
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) {
    float g = 0.0;
    for (int k = 0; k < THREADS; k++) g += batches[k].inputBiases[i];

    UpdateAndApplyGradient(&nn->inputBiases[i], &grads->inputBiases[i], g);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < 2 * N_HIDDEN; i++) {
    float g = 0.0;
    for (int k = 0; k < THREADS; k++) g += batches[k].outputWeights[i];

    UpdateAndApplyGradient(&nn->outputWeights[i], &grads->outputWeights[i], g);
  }

  float g = 0.0;
  for (int k = 0; k < THREADS; k++) g += batches[k].outputBias;

  UpdateAndApplyGradient(&nn->outputBias, &grads->outputBias, g);
}

void ClearGradients(NNGradients* gradients) {
  memset(gradients->inputWeights, 0, sizeof(gradients->inputWeights));
  memset(gradients->inputBiases, 0, sizeof(gradients->inputBiases));

  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(&gradients->outputBias, 0, sizeof(gradients->outputBias));
}

#endif