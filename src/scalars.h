#include <float.h>
#include <stdio.h>

#include "nn.h"
#include "types.h"
#include "util.h"

INLINE void MinAndMax(float* acc, float* min, float* max) {
  for (int i = 0; i < N_HIDDEN; i++) {
    if (acc[i] < *min)
      *min = acc[i];

    if (acc[i] > *max)
      *max = acc[i];
  }
}

INLINE void PrintMinMax(DataSet* data, int n, NN* nn) {
  float min = FLT_MAX, max = FLT_MIN;

  for (int i = 0; i < n; i++) {
    NNActivations act[1];
    NNFirstLayer(nn, &data->entries[i].board, act);

    MinAndMax(act->accumulators[WHITE], &min, &max);
    MinAndMax(act->accumulators[BLACK], &min, &max);
  }

  printf("Min: %.4f, Max: %.4f\n", min, max);
}