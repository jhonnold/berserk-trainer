#ifndef TRAINER_H
#define TRAINER_H

#include "types.h"
#include "util.h"

float TotalError(DataSet* data, NN* nn);
void Train(int batch, DataSet* data, NN* nn, NNGradients* g);

INLINE float Error(float r, DataEntry* e) { return (powf(r - e->wdl, 2) + powf(r - e->eval, 2)) / 2; }
INLINE float ErrorGradient(float r, DataEntry* e) { return (r - e->wdl) + (r - e->eval); }

#endif