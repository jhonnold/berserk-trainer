#ifndef TRAINER_H
#define TRAINER_H

#include "types.h"
#include "util.h"

float TotalError(DataSet* data, NN* nn);
float Train(int batch, DataSet* data, NN* nn, BatchGradients* local, uint8_t* active);

INLINE float Error(float r, Board* b) {
  float target = WDL * b->wdl / 2.0 + EVAL * b->eval;

  return powf(fabs(r - target), 2.5);
}

INLINE float ErrorGradient(float r, Board* b) {
  float target = WDL * b->wdl / 2.0 + EVAL * b->eval;

  return 2.5 * (r - target) * sqrtf(fabs(r - target));
}

#endif