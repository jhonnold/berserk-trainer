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

INLINE float ReLUPrime(float s) { return s > 0; }

INLINE float CReLUPrime(float s) { return s > 0 && s < CRELU_MAX; }

INLINE uint64_t NetworkHash(NN* nn) {
  uint64_t hash = 0;

  for (int i = 0; i < N_HIDDEN * N_INPUT; i++) hash = H(hash, (int)nn->inputWeights[i]);

  for (int i = 0; i < N_HIDDEN; i++) hash = H(hash, (int)nn->inputBiases[i]);

  for (int b = 0; b < N_BUCKETS; b++) {
    for (int i = 0; i < N_HIDDEN * 2; i++) hash = H(hash, (int)nn->outputWeights[b][i]);

    hash = H(hash, (int)nn->outputBias[b]);
  }

  return hash;
}

void* AlignedMalloc(int size);
void AlignedFree(void* ptr);

#endif
