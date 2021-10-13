#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bits.h"
#include "board.h"
#include "nn.h"

const int NETWORK_MAGIC = 'B' | 'Z' << 8 | 1 << 16 | 0 << 24;
const int NETWORK_ID = 0;
const int INPUT_SIZE = N_FEATURES;
const int OUTPUT_SIZE = N_OUTPUT;
const int N_HIDDEN_LAYERS = 1;
const int HIDDEN_SIZES[1] = {N_HIDDEN};

void NNPredict(NN* nn, Board board, NNActivations* results, int stm) {
  // Apply first layer
  memcpy(results->accumulators[WHITE], nn->hiddenBiases[WHITE], sizeof(float) * N_HIDDEN);
  memcpy(results->accumulators[BLACK], nn->hiddenBiases[BLACK], sizeof(float) * N_HIDDEN);

  for (int i = 0; i < 32; i++) {
    if (!board[WHITE][i])
      break;

    for (int j = 0; j < N_HIDDEN; j++) {
      results->accumulators[WHITE][j] += nn->featureWeights[WHITE][board[WHITE][i] * N_HIDDEN + j];
      results->accumulators[BLACK][j] += nn->featureWeights[BLACK][board[BLACK][i] * N_HIDDEN + j];
    }
  }

  // Apply second layer
  results->result = nn->outputBias;
  for (int i = 0; i < N_HIDDEN; i++) {
    results->result += nn->hiddenWeights[i] * fmax(0.0f, results->accumulators[stm][i]);
    results->result += nn->hiddenWeights[i + N_HIDDEN] * fmax(0.0f, results->accumulators[stm ^ 1][i]);
  }
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
  int temp;
  fread(&temp, 4, 1, fp);
  fread(&temp, 4, 1, fp);
  fread(&temp, 4, 1, fp);
  fread(&temp, 4, 1, fp);
  fread(&temp, 4, 1, fp);

  NN* nn = malloc(sizeof(NN));

  fread(nn->featureWeights[WHITE], 4, N_FEATURES * N_HIDDEN, fp);
  fread(nn->featureWeights[BLACK], 4, N_FEATURES * N_HIDDEN, fp);
  fread(nn->hiddenBiases[WHITE], 4, N_HIDDEN, fp);
  fread(nn->hiddenBiases[BLACK], 4, N_HIDDEN, fp);
  fread(nn->hiddenWeights, 4, N_HIDDEN * 2, fp);
  fread(&nn->outputBias, 4, N_OUTPUT, fp);

  fclose(fp);

  return nn;
}

NN* LoadRandomNN() {
  srand(time(NULL));

  NN* nn = malloc(sizeof(NN));

  float max = sqrtf(2.0f / (N_FEATURES * N_HIDDEN));
  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    nn->featureWeights[WHITE][i] = rand() * max / RAND_MAX;

  max = sqrtf(2.0f / (N_FEATURES * N_HIDDEN));
  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    nn->featureWeights[BLACK][i] = rand() * max / RAND_MAX;

  max = sqrtf(2.0f / N_HIDDEN);
  for (int i = 0; i < N_HIDDEN; i++)
    nn->hiddenBiases[WHITE][i] = rand() * max / RAND_MAX;

  max = sqrtf(2.0f / N_HIDDEN);
  for (int i = 0; i < N_HIDDEN; i++)
    nn->hiddenBiases[BLACK][i] = rand() * max / RAND_MAX;

  max = sqrtf(1.0f / N_HIDDEN);
  for (int i = 0; i < N_HIDDEN * 2; i++)
    nn->hiddenWeights[i] = rand() * max / RAND_MAX;

  max = sqrtf(2.0f);
  nn->outputBias = rand() * max / RAND_MAX;

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

  fwrite(nn->featureWeights[WHITE], 4, N_FEATURES * N_HIDDEN, fp);
  fwrite(nn->featureWeights[BLACK], 4, N_FEATURES * N_HIDDEN, fp);
  fwrite(nn->hiddenBiases[WHITE], 4, N_HIDDEN, fp);
  fwrite(nn->hiddenBiases[BLACK], 4, N_HIDDEN, fp);
  fwrite(nn->hiddenWeights, 4, N_HIDDEN * 2, fp);
  fwrite(&nn->outputBias, 4, N_OUTPUT, fp);

  fclose(fp);
}