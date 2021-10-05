#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bits.h"
#include "board.h"
#include "nn.h"

const int NETWORK_MAGIC = 'B' | 'Z' << 8 | 1 << 16 | 0 << 24;
const int NETWORK_ID = 0;
const int INPUT_SIZE = N_FEATURES;
const int OUTPUT_SIZE = N_OUTPUT;
const int N_HIDDEN_LAYERS = 1;
const int HIDDEN_SIZES[1] = {N_HIDDEN};

void NNPredict(NN* nn, Board board, NNActivations* results) {
  // Apply first layer
  memcpy(results->hiddenActivations, nn->hiddenBiases, sizeof(float) * N_HIDDEN);

  for (int i = 0; i < 32; i++)
    if (board[i])
      for (int j = 0; j < N_HIDDEN; j++)
        results->hiddenActivations[j] += nn->featureWeights[board[i] * N_HIDDEN + j];
    else
      break;

  for (int i = 0; i < N_HIDDEN; i++)
    results->hiddenActivations[i] = fmax(0.0f, results->hiddenActivations[i]);

  // Apply second layer
  results->outputActivations[0] = nn->outputBiases[0];

  for (int i = 0; i < N_HIDDEN; i++)
    results->outputActivations[0] += nn->hiddenWeights[i] * results->hiddenActivations[i];
}

NN* LoadNN(char* path) {
  FILE* fp = fopen(path, "rb");
  if (fp == NULL) {
    printf("Unable to read network at %s!\n", path);
    exit(1);
  }

  int magic;
  fread(&magic, 4, 1, fp);

  if (magic != NETWORK_MAGIC) {
    printf("Magic header does not match!\n");
    exit(1);
  }

  // Skip past the topology as we only support one
  fread(NULL, 4, 5, fp);

  NN* nn = malloc(sizeof(NN));

  fread(nn->featureWeights, 4, N_FEATURES * N_HIDDEN, fp);
  fread(nn->hiddenBiases, 4, N_HIDDEN, fp);
  fread(nn->hiddenWeights, 4, N_HIDDEN * N_OUTPUT, fp);
  fread(nn->outputBiases, 4, N_OUTPUT, fp);

  fclose(fp);

  return nn;
}

NN* LoadRandomNN() {
  NN* nn = malloc(sizeof(NN));

  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    nn->featureWeights[i] = (float)rand() / RAND_MAX / 2;

  for (int i = 0; i < N_HIDDEN; i++)
    nn->hiddenBiases[i] = (float)rand() / RAND_MAX / 2;

  for (int i = 0; i < N_HIDDEN * N_OUTPUT; i++)
    nn->hiddenWeights[i] = (float)rand() / RAND_MAX / 2;

  for (int i = 0; i < N_OUTPUT; i++)
    nn->outputBiases[i] = (float)rand() / RAND_MAX / 2;

  return nn;
}

// https://github.com/amanjpro/zahak-trainer/blob/master/network.go#L105
void SaveNN(NN* nn, char* path) {
  FILE* fp = fopen(path, "wb");
  if (fp == NULL) {
    printf("Unable to save network to %s!\n", path);
    return;
  }

  fwrite(&NETWORK_MAGIC, 4, 1, fp);
  fwrite(&NETWORK_ID, 4, 1, fp);
  fwrite(&INPUT_SIZE, 4, 1, fp);
  fwrite(&OUTPUT_SIZE, 4, 1, fp);
  fwrite(&N_HIDDEN_LAYERS, 4, 1, fp);
  fwrite(HIDDEN_SIZES, 4, N_HIDDEN_LAYERS, fp);

  fwrite(nn->featureWeights, 4, N_FEATURES * N_HIDDEN, fp);
  fwrite(nn->hiddenBiases, 4, N_HIDDEN, fp);
  fwrite(nn->hiddenWeights, 4, N_HIDDEN * N_OUTPUT, fp);
  fwrite(nn->outputBiases, 4, N_OUTPUT, fp);

  fclose(fp);
}