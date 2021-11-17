#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <string.h>

#include "nn.h"
#include "trainer.h"
#include "types.h"
#include "util.h"

INLINE void ValidateGradient(double calculatedGradient, float* toValidate, NN* nn, DataEntry* entry) {
  const double H = 0.001;

  NNAccumulators activations[1];

  Features features[1];
  ToFeatures(&entry->board, features);

  float temp = *toValidate;

  // Raise value
  *toValidate += H;
  NNPredict(nn, features, entry->board.stm, activations);
  double raisedSigmoidResult = Sigmoid(activations->output);
  double raisedError = Error(raisedSigmoidResult, entry);
  *toValidate = temp;

  // Lower value
  *toValidate -= H;
  NNPredict(nn, features, entry->board.stm, activations);
  double loweredSigmoidResult = Sigmoid(activations->output);
  double loweredError = Error(loweredSigmoidResult, entry);
  *toValidate = temp;

  double expectedGradient = (raisedError - loweredError) / (2 * H);

  double diff = fabs((calculatedGradient - expectedGradient) / calculatedGradient);

  if (diff > 0.05)
    printf("Failed! Calculated: %+0.10f, Expected: %+0.10f, Diff: %+0.10f\n", calculatedGradient, expectedGradient, diff);
  else
    printf("Correct! Calculated: %+0.10f, Expected: %+0.10f, Diff: %+0.10f\n", calculatedGradient, expectedGradient, diff);
}

INLINE void UpdateAndApplyGradient(float* v, Gradient* grad) {
  if (!grad->g)
    return;

  grad->M = BETA1 * grad->M + (1.0 - BETA1) * grad->g;
  grad->V = BETA2 * grad->V + (1.0 - BETA2) * grad->g * grad->g;
  float delta = ALPHA * grad->M / (sqrt(grad->V) + EPSILON);

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