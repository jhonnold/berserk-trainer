#ifndef TRAINER_H
#define TRAINER_H

#include "types.h"
#include "util.h"

double TotalError(DataSet* data, NN* nn);
void Train(int batch, DataSet* data, NN* nn, NNGradients* g, BatchGradients* local);

INLINE double WDL(int8_t wdl) { return wdl / 2.0; }

INLINE double Error(double r, DataEntry* e) {
  return 0.5 * pow(r - WDL(e->wdl), 2.0) + 0.5 * pow(r - e->eval, 2.0);
}
INLINE double ErrorGradient(double r, DataEntry* e) { return (r - WDL(e->wdl)) + (r - e->eval); }

#endif