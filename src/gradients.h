#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <math.h>
#include <string.h>

#include "types.h"
#include "util.h"

void UpdateAndApplyGradient(float* v, Gradient* grad, float g) {
  if (!g) return;

  grad->M = BETA1 * grad->M + (1.0f - BETA1) * g;
  grad->V = BETA2 * grad->V + (1.0f - BETA2) * g * g;
  float delta = ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);

  *v -= delta;
}

void ApplyGradients(NN* nn, NNGradients* grads, BatchGradients* local) {
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_INPUT * N_HIDDEN; i++) {
    float g = 0.0f;
    for (int t = 0; t < THREADS; t++) g += local[t].inputWeights[i];

    UpdateAndApplyGradient(&nn->inputWeights[i], &grads->inputWeights[i], g);
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN; i++) {
    float g = 0.0f;
    for (int t = 0; t < THREADS; t++) g += local[t].inputBiases[i];

    UpdateAndApplyGradient(&nn->inputBiases[i], &grads->inputBiases[i], g);
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < 2 * N_HIDDEN * N_HIDDEN_2; i++) {
    float g = 0.0f;
    for (int t = 0; t < THREADS; t++) g += local[t].l1Weights[i];

    UpdateAndApplyGradient(&nn->l1Weights[i], &grads->l1Weights[i], g);

    nn->l1Weights[i] = fmaxf(-127.0f / 64, fminf(127.0f / 64, nn->l1Weights[i]));
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_2; i++) {
    float g = 0.0f;
    for (int t = 0; t < THREADS; t++) g += local[t].l1Biases[i];

    UpdateAndApplyGradient(&nn->l1Biases[i], &grads->l1Biases[i], g);
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_2 * N_HIDDEN_3; i++) {
    float g = 0.0f;
    for (int t = 0; t < THREADS; t++) g += local[t].l2Weights[i];

    UpdateAndApplyGradient(&nn->l2Weights[i], &grads->l2Weights[i], g);

    nn->l2Weights[i] = fmaxf(-127.0f / 64, fminf(127.0f / 64, nn->l2Weights[i]));
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_3; i++) {
    float g = 0.0f;
    for (int t = 0; t < THREADS; t++) g += local[t].l2Biases[i];

    UpdateAndApplyGradient(&nn->l2Biases[i], &grads->l2Biases[i], g);
  }

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int i = 0; i < N_HIDDEN_3; i++) {
    float g = 0.0f;
    for (int t = 0; t < THREADS; t++) g += local[t].outputWeights[i];

    UpdateAndApplyGradient(&nn->outputWeights[i], &grads->outputWeights[i], g);

    nn->outputWeights[i] =
        fmaxf(-127.0f * 127.0f / 16, fminf(127.0f * 127.0f / 16, nn->outputWeights[i]));
  }

  float g = 0.0f;
  for (int t = 0; t < THREADS; t++) g += local[t].outputBias;

  UpdateAndApplyGradient(&nn->outputBias, &grads->outputBias, g);
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