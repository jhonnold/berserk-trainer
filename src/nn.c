#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bits.h"
#include "board.h"
#include "nn.h"
#include "random.h"

const int NETWORK_MAGIC = 'B' | 'R' << 8 | 'K' << 16 | 'R' << 24;

void NNPredict(NN* nn, Features* f, Color stm, NNAccumulators* results) {
  results->output = 0.0f;

  // Apply first layer
  memcpy(results->acc1[WHITE], nn->inputBiases, sizeof(float) * N_HIDDEN);
  memcpy(results->acc1[BLACK], nn->inputBiases, sizeof(float) * N_HIDDEN);

  for (int i = 0; i < f->n; i++) {
    for (size_t j = 0; j < N_HIDDEN; j++) {
      results->acc1[WHITE][j] += nn->inputWeights[f->features[WHITE][i] * N_HIDDEN + j];
      results->acc1[BLACK][j] += nn->inputWeights[f->features[BLACK][i] * N_HIDDEN + j];
    }
  }

  ReLU(results->acc1[WHITE], N_HIDDEN);
  ReLU(results->acc1[BLACK], N_HIDDEN);

  results->output += DotProduct(results->acc1[stm], nn->outputWeights, N_HIDDEN) +
                     DotProduct(results->acc1[stm ^ 1], nn->outputWeights + N_HIDDEN, N_HIDDEN) + //
                     nn->outputBias;
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

  fread(nn->inputWeights, sizeof(float), N_INPUT * N_HIDDEN, fp);
  fread(nn->inputBiases, sizeof(float), N_HIDDEN, fp);
  fread(nn->outputWeights, sizeof(float), N_HIDDEN * 2, fp);
  fread(&nn->outputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);

  return nn;
}

NN* LoadRandomNN() {
  srand(time(NULL));
  NN* nn = malloc(sizeof(NN));

  for (int i = 0; i < N_INPUT * N_HIDDEN; i++)
    nn->inputWeights[i] = RandomGaussian(0, sqrt(1.0 / 32));

  for (int i = 0; i < N_HIDDEN; i++)
    nn->inputBiases[i] = 0;

  for (int i = 0; i < N_HIDDEN * 2; i++)
    nn->outputWeights[i] = RandomGaussian(0, sqrt(1.0 / N_HIDDEN));

  nn->outputBias = 0;

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

  fwrite(nn->inputWeights, sizeof(float), N_INPUT * N_HIDDEN, fp);
  fwrite(nn->inputBiases, sizeof(float), N_HIDDEN, fp);
  fwrite(nn->outputWeights, sizeof(float), N_HIDDEN * 2, fp);
  fwrite(&nn->outputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);
}