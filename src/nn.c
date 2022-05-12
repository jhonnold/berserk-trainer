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

  float psqt = 0.0f;

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

    psqt += (nn->psqtWeights[f->features[i][stm]] - nn->psqtWeights[f->features[i][stm ^ 1]]) / 2;
  }

  ReLU(trace->accumulator, N_L1);

  trace->output += DotProduct(trace->accumulator, nn->outputWeights, N_L1);
  trace->output += psqt;
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
  fread(nn->outputWeights, sizeof(float), N_L1, fp);
  fread(&nn->outputBias, sizeof(float), N_OUTPUT, fp);
  fread(nn->psqtWeights, sizeof(float), N_INPUT, fp);

  fclose(fp);

  return nn;
}

NN* LoadRandomNN() {
  srand(time(NULL));
  NN* nn = AlignedMalloc(sizeof(NN));

  for (int i = 0; i < N_INPUT * N_HIDDEN; i++) nn->inputWeights[i] = RandomGaussian(0, sqrt(1.0 / 32));

  for (int i = 0; i < N_HIDDEN; i++) nn->inputBiases[i] = 0;

  for (int i = 0; i < N_L1; i++) nn->outputWeights[i] = RandomGaussian(0, sqrt(1.0 / N_HIDDEN));

  nn->outputBias = 0;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 128; j++) {
      int whiteIdx = i * 128 + j;
      int blackIdx = whiteIdx + 768;

      float v;
      if (i == 0)
        v = 60.0f;
      else if (i == 1)
        v = 375.0f;
      else if (i == 2)
        v = 400.0f;
      else if (i == 3)
        v = 625.0f;
      else if (i == 4)
        v = 1250.0f;

      nn->psqtWeights[whiteIdx] = v; 
      nn->psqtWeights[blackIdx] = -v;
    }
  }

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
  fwrite(nn->outputWeights, sizeof(float), N_L1, fp);
  fwrite(&nn->outputBias, sizeof(float), N_OUTPUT, fp);
  fwrite(nn->psqtWeights, sizeof(float), N_INPUT, fp);

  fclose(fp);
}