#ifndef UTIL_H
#define UTIL_H

#include <math.h>
#include <stdlib.h>

#include "types.h"

#define INLINE static inline __attribute__((always_inline))
#define H(h, v) ((h) + (324723947ULL + (v))) ^ 93485734985ULL

long GetTimeMS();

INLINE float Sigmoid(float s) { return 1.0f / (1.0f + expf(-s * SS)); }

INLINE float SigmoidPrime(float s) { return s * (1.0 - s) * SS; }

INLINE uint64_t NetworkHash(NN* nn) {
  uint64_t hash = 0;

  for (int i = 0; i < N_HIDDEN * N_FEATURES; i++)
    hash = H(hash, (int)nn->featureWeights[i]);

  for (int i = 0; i < N_HIDDEN; i++)
    hash = H(hash, (int)nn->hiddenBiases[i]);

  for (int i = 0; i < N_HIDDEN * 2; i++)
    hash = H(hash, (int)nn->hiddenWeights[i]);

  return H(hash, (int)nn->outputBias);
}

INLINE float Random(int s) {
  float m = sqrtf(2.0f / s);
  return rand() * m / RAND_MAX;
}

// https://en.wikipedia.org/wiki/Fast_inverse_square_root
INLINE float InvSQRT(float n) {
  float x2 = n * 0.5f;

  long i = *(long*)&n;
  i = 0x5f3759df - (i >> 1);
  n = *(float*)&i;
  n *= 1.5f - (x2 * n * n);

  return n;
}

#endif
