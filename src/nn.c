#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bits.h"
#include "board.h"
#include "nn.h"

const int NETWORK_MAGIC = 'B' | 'R' << 8 | 'K' << 16 | 'R' << 24;

void NNPredict(NN* nn, Board* board, NNAccumulators* acc) {
  acc->output = nn->outputBias;

  // Input Layer
  memcpy(acc->input[WHITE], nn->inputBiases, sizeof(float) * N_HIDDEN);
  memcpy(acc->input[BLACK], nn->inputBiases, sizeof(float) * N_HIDDEN);

  for (int i = 0; i < board->n; i++) {
    Feature wf = idx(board->pieces[i], board->wk, WHITE);
    Feature bf = idx(board->pieces[i], board->bk, BLACK);

    // skip connections -> output
    acc->output += nn->skipWeights[wf];

    for (size_t j = 0; j < N_HIDDEN; j++) {
      acc->input[WHITE][j] += nn->inputWeights[wf * N_HIDDEN + j];
      acc->input[BLACK][j] += nn->inputWeights[bf * N_HIDDEN + j];
    }
  }

  ClippedReLU(acc->input[WHITE], N_HIDDEN);
  ClippedReLU(acc->input[BLACK], N_HIDDEN);

  // Hidden layers
  memcpy(acc->hidden, nn->hiddenBiases, sizeof(float) * N_HIDDEN_2);

  for (int i = 0; i < N_HIDDEN_2; i++)
    acc->hidden[i] += DotProduct(acc->input[WHITE], nn->hiddenWeights + 2 * N_HIDDEN * i, N_HIDDEN) +
                      DotProduct(acc->input[BLACK], nn->hiddenWeights + 2 * N_HIDDEN * i + N_HIDDEN, N_HIDDEN);

  ReLU(acc->hidden, N_HIDDEN_2);

  // Output layer
  acc->output += DotProduct(acc->hidden, nn->outputWeights, N_HIDDEN_2);
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

  uint64_t hash;
  fread(&hash, sizeof(uint64_t), 1, fp);
  printf("Reading network with hash %llx\n", hash);

  NN* nn = malloc(sizeof(NN));

  fread(nn->skipWeights, sizeof(float), N_FEATURES, fp);

  fread(nn->inputWeights, sizeof(float), N_FEATURES * N_HIDDEN, fp);
  fread(nn->inputBiases, sizeof(float), N_HIDDEN, fp);

  fread(nn->hiddenWeights, sizeof(float), 2 * N_HIDDEN * N_HIDDEN_2, fp);
  fread(nn->hiddenBiases, sizeof(float), N_HIDDEN_2 * 2, fp);

  fread(nn->outputWeights, sizeof(float), N_HIDDEN_2, fp);
  fread(&nn->outputBias, sizeof(float), 1, fp);

  fclose(fp);

  return nn;
}

NN* LoadRandomNN() {
  srand(time(NULL));
  NN* nn = malloc(sizeof(NN));

  for (int i = 0; i < N_FEATURES; i++)
    nn->skipWeights[i] = Random(N_FEATURES);

  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    nn->inputWeights[i] = Random(N_FEATURES * N_HIDDEN);

  for (int i = 0; i < N_HIDDEN; i++)
    nn->inputBiases[i] = Random(N_HIDDEN);

  for (int i = 0; i < 2 * N_HIDDEN * N_HIDDEN_2; i++)
    nn->hiddenWeights[i] = Random(2 * N_HIDDEN * N_HIDDEN_2);

  for (int i = 0; i < N_HIDDEN_2; i++)
    nn->hiddenBiases[i] = Random(N_HIDDEN_2);

  for (int i = 0; i < N_HIDDEN_2; i++)
    nn->outputWeights[i] = Random(N_HIDDEN_2);

  nn->outputBias = Random(1);

  return nn;
}

void SaveNN(NN* nn, char* path) {
  FILE* fp = fopen(path, "wb");
  if (fp == NULL) {
    printf("Unable to save network to %s!\n", path);
    return;
  }

  fwrite(&NETWORK_MAGIC, sizeof(int), 1, fp);

  uint64_t hash = NetworkHash(nn);
  fwrite(&hash, sizeof(uint64_t), 1, fp);

  fwrite(nn->skipWeights, sizeof(float), N_FEATURES, fp);

  fwrite(nn->inputWeights, sizeof(float), N_FEATURES * N_HIDDEN, fp);
  fwrite(nn->inputBiases, sizeof(float), N_HIDDEN, fp);

  fwrite(nn->hiddenWeights, sizeof(float), 2 * N_HIDDEN * N_HIDDEN_2, fp);
  fwrite(nn->hiddenBiases, sizeof(float), N_HIDDEN_2 * 2, fp);

  fwrite(nn->outputWeights, sizeof(float), N_HIDDEN_2, fp);
  fwrite(&nn->outputBias, sizeof(float), 1, fp);

  fclose(fp);
}