#ifndef UTIL_H
#define UTIL_H

#include <math.h>
#include <stdlib.h>

#include "random.h"
#include "types.h"

#define INLINE static inline __attribute__((always_inline))
#define H(h, v) ((h) + (324723947ULL + (v))) ^ 93485734985ULL

long GetTimeMS();

INLINE float Sigmoid(float s) { return 1.0f / (1.0f + expf(-s * SS)); }

INLINE float SigmoidPrime(float s) { return s * (1.0 - s) * SS; }

INLINE float ReLUPrime(float s) { return s > 0.0f; }

INLINE uint64_t NetworkHash(NN* nn) {
  uint64_t hash = 0;

  for (int i = 0; i < N_HIDDEN * N_INPUT; i++)
    hash = H(hash, (int)nn->inputWeights[i]);

  for (int i = 0; i < N_HIDDEN; i++)
    hash = H(hash, (int)nn->inputBiases[i]);

  for (int i = 0; i < 2 * N_HIDDEN * N_HIDDEN_2; i++)
    hash = H(hash, (int)nn->h2Weights[i]);

  for (int i = 0; i < N_HIDDEN_2; i++)
    hash = H(hash, (int)nn->h2Biases[i]);

  for (int i = 0; i < N_HIDDEN_2; i++)
    hash = H(hash, (int)nn->outputWeights[i]);

  hash = H(hash, (int)nn->outputBias);

  return hash;
}

INLINE float Random(int s) {
  double m = sqrtf(2.0f / s);
  return (double) RandomUInt64() / UINT64_MAX * m;
}

#endif
