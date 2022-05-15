#include "nn.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bits.h"
#include "board.h"
#include "random.h"
#include "util.h"

const int NETWORK_MAGIC = 'B' | 'R' << 8 | 'K' << 16 | 'R' << 24;

void NNPredict(NN* nn, Features* f, Color stm, NetworkTrace* trace) {
  trace->output = nn->outputBias;

  // Apply first layer
  float* stmAccumulator = trace->accumulator;
  float* xstmAccumulator = &trace->accumulator[N_HIDDEN];

  memcpy(stmAccumulator, nn->inputBiases, sizeof(float) * N_HIDDEN);
  memcpy(xstmAccumulator, nn->inputBiases, sizeof(float) * N_HIDDEN);

  for (int i = 0; i < f->n; i++) {
    for (size_t j = 0; j < N_HIDDEN; j++) {
      stmAccumulator[j] += nn->inputWeights[f->features[i][stm] * N_HIDDEN + j];
      xstmAccumulator[j] += nn->inputWeights[f->features[i][stm ^ 1] * N_HIDDEN + j];
    }
  }

  ReLU(trace->accumulator, N_L1);

  memcpy(trace->l2acc, nn->l2Biases, sizeof(float) * N_L2);
  for (int i = 0; i < N_L2; i++) trace->l2acc[i] += DotProduct(trace->accumulator, &nn->l2Weights[N_L1 * i], N_L1);
  ReLU(trace->l2acc, N_L2);

  memcpy(trace->l3acc, nn->l3Biases, sizeof(float) * N_L3);
  for (int i = 0; i < N_L3; i++) trace->l3acc[i] += DotProduct(trace->l2acc, &nn->l3Weights[N_L2 * i], N_L2);
  ReLU(trace->l3acc, N_L3);

  trace->output += DotProduct(trace->l3acc, nn->outputWeights, N_L3);
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

  NN* nn = AlignedMalloc(sizeof(NN));

  fread(nn->inputWeights, sizeof(float), N_INPUT * N_HIDDEN, fp);
  fread(nn->inputBiases, sizeof(float), N_HIDDEN, fp);
  fread(nn->l2Weights, sizeof(float), N_L1 * N_L2, fp);
  fread(nn->l2Biases, sizeof(float), N_L2, fp);
  fread(nn->l3Weights, sizeof(float), N_L2 * N_L3, fp);
  fread(nn->l3Biases, sizeof(float), N_L3, fp);
  fread(nn->outputWeights, sizeof(float), N_L3, fp);
  fread(&nn->outputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);

  return nn;
}

NN* LoadRandomNN() {
  srand(time(NULL));
  NN* nn = AlignedMalloc(sizeof(NN));

  for (int i = 0; i < N_INPUT * N_HIDDEN; i++) nn->inputWeights[i] = RandomGaussian(0, sqrt(2.0 / 32));

  for (int i = 0; i < N_HIDDEN; i++) nn->inputBiases[i] = 0;

  for (int i = 0; i < N_L1 * N_L2; i++) nn->l2Weights[i] = RandomGaussian(0, sqrt(2.0 / N_L1));

  for (int i = 0; i < N_L2; i++) nn->l2Biases[i] = 0;

  for (int i = 0; i < N_L2 * N_L3; i++) nn->l3Weights[i] = RandomGaussian(0, sqrt(2.0 / N_L2));

  for (int i = 0; i < N_L3; i++) nn->l3Biases[i] = 0;

  for (int i = 0; i < N_L3; i++) nn->outputWeights[i] = RandomGaussian(0, sqrt(2.0 / N_L3));

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
  fwrite(nn->l2Weights, sizeof(float), N_L1 * N_L2, fp);
  fwrite(nn->l2Biases, sizeof(float), N_L2, fp);
  fwrite(nn->l3Weights, sizeof(float), N_L2 * N_L3, fp);
  fwrite(nn->l3Biases, sizeof(float), N_L3, fp);
  fwrite(nn->outputWeights, sizeof(float), N_L3, fp);
  fwrite(&nn->outputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);
}