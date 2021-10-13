#ifndef NN_H
#define NN_H

#include "board.h"

#define N_FEATURES 768
#define N_HIDDEN 64
#define N_OUTPUT 1

typedef struct {
  float outputBias;
  float featureWeights[2][N_FEATURES * N_HIDDEN];
  float hiddenBiases[2][N_HIDDEN];
  float hiddenWeights[N_HIDDEN * 2];
} NN;

typedef struct {
  float result;
  float accumulators[2][N_HIDDEN];
} NNActivations;

void NNPredict(NN* nn, Board board, NNActivations* results, int stm);

NN* LoadNN(char* path);
NN* LoadRandomNN();

void SaveNN(NN* nn, char* path);

#endif