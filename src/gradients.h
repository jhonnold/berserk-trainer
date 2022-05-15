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

    nn->l2Weights[i] = fminf(127.0 / 64, fmaxf(-127.0 / 64, nn->l2Weights[i]));
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_L2 * N_L3; i++) {
    float g = 0.0;
    for (int t = 0; t < THREADS; t++) g += local[t].l3Weights[i];

    UpdateAndApplyGradient(&nn->l3Weights[i], &grads->l3Weights[i], g);

    nn->l3Weights[i] = fminf(127.0 / 64, fmaxf(-127.0 / 64, nn->l3Weights[i]));
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_L2; i++) {
    float g = 0.0;
    for (int t = 0; t < THREADS; t++) g += local[t].l2Biases[i];

    UpdateAndApplyGradient(&nn->l2Biases[i], &grads->l2Biases[i], g);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_L3; i++) {
    float g = 0.0;
    for (int t = 0; t < THREADS; t++) g += local[t].l3Biases[i];

    UpdateAndApplyGradient(&nn->l3Biases[i], &grads->l3Biases[i], g);
  }

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int i = 0; i < N_L3; i++) {
    float g = 0.0;
    for (int t = 0; t < THREADS; t++) g += local[t].outputWeights[i];

    UpdateAndApplyGradient(&nn->outputWeights[i], &grads->outputWeights[i], g);

    nn->outputWeights[i] = fminf(127.0 * 127.0 / 16, fmaxf(-127.0 * 127.0 / 16, nn->outputWeights[i]));
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

  memset(gradients->l3Weights, 0, sizeof(gradients->l3Weights));
  memset(gradients->l3Biases, 0, sizeof(gradients->l3Biases));

  memset(gradients->outputWeights, 0, sizeof(gradients->outputWeights));
  memset(&gradients->outputBias, 0, sizeof(gradients->outputBias));
}

#endif