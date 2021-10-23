#ifndef TRAINER_H
#define TRAINER_H

#include "types.h"
#include "util.h"

float TotalError(DataSet* data, NN* nn);
void* CalculateError(void* arg);
void Train(int batch, DataSet* data, NN* nn, NNGradients* g, BatchGradients* threadGradients);
void* CalculateGradients(void* arg);
void UpdateNetwork(NN* nn, NNGradients* g);
void UpdateAndApplyGradient(float* v, Gradient* grad);
void ClearGradients(NNGradients* gradients);
void ClearBatchGradients(BatchGradients* gradients);

INLINE float Error(float r, DataEntry* e) { return (powf(r - e->wdl, 2) + powf(r - e->eval, 2)) / 2; }
INLINE float ErrorGradient(float r, DataEntry* e) { return (r - e->wdl) + (r - e->eval); }

#endif