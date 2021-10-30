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

void NNPredict(NN* nn, Board* board, NNActivations* results) {
  results->result = 0.0f;

  // Apply first layer
  memcpy(results->accumulators[WHITE], nn->hiddenBiases, sizeof(float) * N_HIDDEN);
  memcpy(results->accumulators[BLACK], nn->hiddenBiases, sizeof(float) * N_HIDDEN);

  memcpy(results->kpAccumulators[WHITE], nn->kpHiddenBiases, sizeof(float) * N_KP_HIDDEN);
  memcpy(results->kpAccumulators[BLACK], nn->kpHiddenBiases, sizeof(float) * N_KP_HIDDEN);

  for (int i = 0; i < board->n; i++) {
    OccupiedSquare occ = board->pieces[i];

    Feature wf = idx(occ, board->wk, WHITE);
    Feature bf = idx(occ, board->bk, BLACK);

    results->result += nn->skipWeights[wf];

    for (size_t j = 0; j < N_HIDDEN; j++) {
      results->accumulators[WHITE][j] += nn->featureWeights[wf * N_HIDDEN + j];
      results->accumulators[BLACK][j] += nn->featureWeights[bf * N_HIDDEN + j];
    }

    if (occ.pc != WHITE_PAWN && occ.pc != BLACK_PAWN && occ.pc != WHITE_KING && occ.pc != BLACK_KING)
      continue;

    wf = kpIdx(occ, WHITE);
    bf = kpIdx(occ, BLACK);

    for (size_t j = 0; j < N_KP_HIDDEN; j++) {
      results->kpAccumulators[WHITE][j] += nn->kpFeatureWeights[wf * N_KP_HIDDEN + j];
      results->kpAccumulators[BLACK][j] += nn->kpFeatureWeights[bf * N_KP_HIDDEN + j];
    }
  }

  ReLU(results->accumulators[WHITE], N_HIDDEN);
  ReLU(results->accumulators[BLACK], N_HIDDEN);

  ReLU(results->kpAccumulators[WHITE], N_KP_HIDDEN);
  ReLU(results->kpAccumulators[BLACK], N_KP_HIDDEN);

  results->result += DotProduct(results->accumulators[WHITE], nn->hiddenWeights, N_HIDDEN) +
                     DotProduct(results->accumulators[BLACK], nn->hiddenWeights + N_HIDDEN, N_HIDDEN) +         //
                     nn->outputBias +                                                                           //
                     DotProduct(results->kpAccumulators[WHITE], nn->hiddenWeights, N_KP_HIDDEN) +               //
                     DotProduct(results->kpAccumulators[BLACK], nn->hiddenWeights + N_KP_HIDDEN, N_KP_HIDDEN) + //
                     nn->kpOutputBias;
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

  fread(nn->featureWeights, sizeof(float), N_FEATURES * N_HIDDEN, fp);
  fread(nn->hiddenBiases, sizeof(float), N_HIDDEN, fp);
  fread(nn->hiddenWeights, sizeof(float), N_HIDDEN * 2, fp);
  fread(&nn->outputBias, sizeof(float), N_OUTPUT, fp);

  fread(nn->skipWeights, sizeof(float), N_FEATURES, fp);

  fread(nn->kpFeatureWeights, sizeof(float), N_KP_FEATURES * N_KP_HIDDEN, fp);
  fread(nn->kpHiddenBiases, sizeof(float), N_KP_HIDDEN, fp);
  fread(nn->kpHiddenWeights, sizeof(float), N_KP_HIDDEN * 2, fp);
  fread(&nn->kpOutputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);

  return nn;
}

NN* LoadRandomNN() {
  srand(time(NULL));
  NN* nn = malloc(sizeof(NN));

  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    nn->featureWeights[i] = Random(N_FEATURES * N_HIDDEN);

  for (int i = 0; i < N_HIDDEN; i++)
    nn->hiddenBiases[i] = Random(N_HIDDEN);

  for (int i = 0; i < N_HIDDEN * 2; i++)
    nn->hiddenWeights[i] = Random(N_HIDDEN * 2);

  nn->outputBias = Random(1);

  for (int i = 0; i < N_FEATURES; i++)
    nn->skipWeights[i] = Random(N_FEATURES);

  for (int i = 0; i < N_KP_FEATURES * N_KP_HIDDEN; i++)
    nn->kpFeatureWeights[i] = Random(N_KP_FEATURES * N_KP_HIDDEN);

  for (int i = 0; i < N_KP_HIDDEN; i++)
    nn->kpHiddenBiases[i] = Random(N_KP_HIDDEN);

  for (int i = 0; i < N_KP_HIDDEN * 2; i++)
    nn->kpHiddenWeights[i] = Random(N_KP_HIDDEN * 2);

  nn->kpOutputBias = Random(1);

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

  fwrite(nn->featureWeights, sizeof(float), N_FEATURES * N_HIDDEN, fp);
  fwrite(nn->hiddenBiases, sizeof(float), N_HIDDEN, fp);
  fwrite(nn->hiddenWeights, sizeof(float), N_HIDDEN * 2, fp);
  fwrite(&nn->outputBias, sizeof(float), N_OUTPUT, fp);
  fwrite(nn->skipWeights, sizeof(float), N_FEATURES, fp);
  fwrite(nn->kpFeatureWeights, sizeof(float), N_KP_FEATURES * N_KP_HIDDEN, fp);
  fwrite(nn->kpHiddenBiases, sizeof(float), N_KP_HIDDEN, fp);
  fwrite(nn->kpHiddenWeights, sizeof(float), N_KP_HIDDEN * 2, fp);
  fwrite(&nn->kpOutputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);
}