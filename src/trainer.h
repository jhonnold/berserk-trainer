#ifndef TRAINER_H
#define TRAINER_H

#include "types.h"
#include "util.h"

float TotalError(DataSet* data, NN* nn);
float Train(int batch, DataSet* data, NN* nn, NNGradients* g, BatchGradients* local);

INLINE float Error(float r, Board* b) {
  return WDL * powf(fabs(r - b->wdl / 2.0), 2.5) +  //
         EVAL * powf(fabs(r - b->eval), 2.5);
}

INLINE float ErrorGradient(float r, Board* b) {
  return WDL * 2.5 * (r - b->wdl / 2.0) * sqrtf(fabs(r - b->wdl / 2.0)) +  //
         EVAL * 2.5 * (r - b->eval) * sqrtf(fabs(r - b->eval));
}

#endif