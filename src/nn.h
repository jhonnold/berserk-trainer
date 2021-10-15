#ifndef NN_H
#define NN_H

#include "board.h"

#define N_FEATURES 768
#define N_HIDDEN 256
#define N_OUTPUT 1

typedef struct {
  float outputBias;
  float featureWeights[N_FEATURES * N_HIDDEN] __attribute__((aligned(32)));
  float hiddenBiases[N_HIDDEN] __attribute__((aligned(32)));
  float hiddenWeights[N_HIDDEN * 2] __attribute__((aligned(32)));
} NN;

typedef struct {
  float result;
  float accumulators[2][N_HIDDEN] __attribute__((aligned(32)));
} NNActivations;

void NNPredict(NN* nn, Board board, NNActivations* results, int stm);

NN* LoadNN(char* path);
NN* LoadRandomNN();

void SaveNN(NN* nn, char* path);

#endif