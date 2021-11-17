#ifndef UTIL_H
#define UTIL_H

#include <math.h>
#include <stdlib.h>

#include "types.h"

#define INLINE static inline __attribute__((always_inline))
#define H(h, v) ((h) + (324723947ULL + (v))) ^ 93485734985ULL

long GetTimeMS();

INLINE double Sigmoid(double s) { return 1.0 / (1.0 + exp(-SS * s)); }

INLINE double SigmoidPrime(double sig) { return SS * sig * (1.0 - sig); }

INLINE int ReLUPrime(float s) { return s > 0.0; }

INLINE float CReLUPrime(float s) { return s > 0.0f && s < 1.0f; }

INLINE uint64_t NetworkHash(NN* nn) {
  uint64_t hash = 0;

  for (int i = 0; i < N_HIDDEN * N_INPUT; i++)
    hash = H(hash, (int)nn->inputWeights[i]);

  for (int i = 0; i < N_HIDDEN; i++)
    hash = H(hash, (int)nn->inputBiases[i]);

  for (int i = 0; i < N_HIDDEN_2; i++)
    hash = H(hash, (int)nn->outputWeights[i]);

  hash = H(hash, (int)nn->outputBias);

  return hash;
}

INLINE float Random(int s) {
  float m = sqrt(2.0f / s);
  float r = rand() * m / RAND_MAX;
  return !(rand() & 1) ? -r : r;
}

#endif
