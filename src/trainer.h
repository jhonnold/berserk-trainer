#ifndef TRAINER_H
#define TRAINER_H

#include "types.h"
#include "util.h"

float TotalError(DataSet* data, NN* nn);
float Train(int batch, DataSet* data, NN* nn, BatchGradients* local);

INLINE float Error(float r, Board* b) {
  return WDL * powf(fabsf(r - b->wdl / 2.0f), 2.5f) +  //
         EVAL * powf(fabsf(r - b->eval), 2.5f);
}

INLINE float ErrorGradient(float r, Board* b) {
  return WDL * 2.5f* (r - b->wdl / 2.0f) * sqrtf(fabsf(r - b->wdl / 2.0f)) +  //
         EVAL * 2.5f * (r - b->eval) * sqrtf(fabsf(r - b->eval));
}

#endif