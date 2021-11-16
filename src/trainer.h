#ifndef TRAINER_H
#define TRAINER_H

#include "types.h"
#include "util.h"

float TotalError(DataSet* data, NN* nn);
void Train(int batch, DataSet* data, NN* nn, NNGradients* g, BatchGradients* local);

INLINE float WDL(int8_t wdl) { return wdl / 2.0f; }

INLINE float Error(float r, DataEntry* e) {
  return 0.5f * powf(r - WDL(e->wdl), 2.0f) + 0.5f * powf(r - e->eval, 2.0f);
}
INLINE float ErrorGradient(float r, DataEntry* e) { return (r - WDL(e->wdl)) + (r - e->eval); }

#endif