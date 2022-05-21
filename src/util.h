#ifndef UTIL_H
#define UTIL_H

#include <math.h>
#include <stdlib.h>

#include "types.h"

#define INLINE static inline __attribute__((always_inline))
#define H(h, v) ((h) + (324723947ULL + (v))) ^ 93485734985ULL

long GetTimeMS();

INLINE float Sigmoid(float s) { return 1.0 / (1.0 + expf(-s * SS)); }

INLINE float SigmoidPrime(float s) { return s * (1.0 - s) * SS; }

INLINE float ReLUPrime(float s) { return s > 0.0; }

INLINE float CReLUPrime(float s) { return s > 0.0 && s < CRELU_MAX; }

void* AlignedMalloc(int size);
void AlignedFree(void* ptr);

#endif
