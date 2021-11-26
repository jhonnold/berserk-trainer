#ifndef TRAINER_H
#define TRAINER_H

#include "types.h"
#include "util.h"

float TotalError(DataSet* data, NN* nn);
void Train(int batch, DataSet* data, NN* nn, NNGradients* g, BatchGradients* local);

INLINE float Error(float r, DataEntry* e) {
  return WDL_WEIGHT * powf(fabs(r - e->wdl / 2.0f), 2.5f) + EVAL_WEIGHT * powf(fabs(r - e->eval), 2.5f);
}

INLINE float ErrorGradient(float r, DataEntry* e) {
  return 2.5f * WDL_WEIGHT * (r - e->wdl / 2.0f) * sqrtf(fabs(r - e->wdl / 2.0f)) +
         2.5f * EVAL_WEIGHT * (r - e->eval) * sqrtf(fabs(r - e->eval));
}

#endif