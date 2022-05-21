#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "types.h"
#include "util.h"

void UpdateAndApplyGradientWithAge(float* v, Gradient* grad, float g, int age) {
  grad->M = powf(BETA1, age) * grad->M + (1.0 - BETA1) * g;
  grad->V = powf(BETA2, age) * grad->V + (1.0 - BETA2) * g * g;

  *v -= ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);
}

void UpdateAndApplyGradient(float* v, Gradient* grad, float g) {
  grad->M = BETA1 * grad->M + (1.0 - BETA1) * g;
  grad->V = BETA2 * grad->V + (1.0 - BETA2) * g * g;

  *v -= ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);
}

void ApplyGradients(NN* nn, NNGradients* grads, BatchGradients* local, uint8_t* active) {
#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_INPUT; i++) {
    if (!active[i]) continue;

    int age = ITERATION - LAST_SEEN[i];
    LAST_SEEN[i] = ITERATION;

    for (int j = 0; j < N_HIDDEN; j++) {
      int idx = i * N_HIDDEN + j;

      float g = 0.0;
      for (int t = 0; t < THREADS; t++) g += local[t].inputWeights[idx];

      UpdateAndApplyGradientWithAge(&nn->inputWeights[idx], &grads->inputWeights[idx], g, age);
    }
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) {
    float g = 0.0;
    for (int t = 0; t < THREADS; t++) g += local[t].inputBiases[i];

    UpdateAndApplyGradient(&nn->inputBiases[i], &grads->inputBiases[i], g);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_L1 * N_L2; i++) {
    float g = 0.0;
    for (int t = 0; t < THREADS; t++) g += local[t].l2Weights[i];

    UpdateAndApplyGradient(&nn->l2Weights[i], &grads->l2Weights[i], g);

    nn->l2Weights[i] = fmaxf(fminf(nn->l2Weights[i], QUANT_IN / QUANT_W_H), -QUANT_IN / QUANT_W_H);
  }

  for (int i = 0; i < N_L2; i++) {
    float g = 0.0;
    for (int t = 0; t < THREADS; t++) g += local[t].l2Biases[i];

    UpdateAndApplyGradient(&nn->l2Biases[i], &grads->l2Biases[i], g);
  }

  for (int i = 0; i < N_L2; i++) {
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

  memset(gradients->l2Weights, 0, sizeof(gradients->l2Weights));
  memset(gradients->l2Biases, 0, sizeof(gradients->l2Biases));

  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(&gradients->outputBias, 0, sizeof(gradients->outputBias));
}

#endif