#ifndef TRAINER_H
#define TRAINER_H

#include "types.h"
#include "util.h"

float TotalError(DataSet* data, NN* nn);
void Train(int batch, DataSet* data, NN* nn, Gradients* gradients);

INLINE float Error(float r, Board* b) { return powf(fabs(r - b->wdl / 2.0), 2.5); }

INLINE float ErrorGradient(float r, Board* b) { return 2.5 * (r - b->wdl / 2.0) * sqrtf(fabs(r - b->wdl / 2.0)); }

#endif