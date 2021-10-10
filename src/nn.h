#ifndef NN_H
#define NN_H

#include "board.h"

#define N_FEATURES 768
#define N_HIDDEN 512
#define N_OUTPUT 1

typedef struct {
    float featureWeights[N_FEATURES * N_HIDDEN];
    float hiddenWeights[N_HIDDEN * N_OUTPUT];
    float hiddenBiases[N_HIDDEN];
    float outputBiases[N_OUTPUT];
} NN;

typedef struct {
    float hiddenActivations[N_HIDDEN];
    float outputActivations[N_OUTPUT];
} NNActivations;

void NNPredict(NN* nn, Board board, NNActivations* results);

NN* LoadNN(char* path);
NN* LoadRandomNN();

void SaveNN(NN* nn, char* path);

#endif