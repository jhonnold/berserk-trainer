#ifndef NN_H
#define NN_H

#include "board.h"

#define N_FEATURES 1632
#define N_HIDDEN 128
#define N_OUTPUT 1

typedef struct {
  float outputBias;
  float featureWeights[N_FEATURES * N_HIDDEN] __attribute__((aligned(64)));
  float hiddenBiases[N_HIDDEN] __attribute__((aligned(64)));
  float hiddenWeights[N_HIDDEN * 2] __attribute__((aligned(64)));
} NN;

typedef struct {
  float result;
  float accumulators[2][N_HIDDEN] __attribute__((aligned(64)));
} NNActivations;

void NNPredict(NN* nn, Board* board, NNActivations* results, int stm);

NN* LoadNN(char* path);
NN* LoadRandomNN();

void SaveNN(NN* nn, char* path);

uint64_t NetworkHash(NN* nn);

#endif