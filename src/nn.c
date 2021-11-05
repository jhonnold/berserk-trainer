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

void NNFirstLayer(NN* nn, Board* board, NNAccumulators* results) {

  // Apply first layer
  memset(results->acc1[WHITE], 0, sizeof(float) * N_HIDDEN);
  memset(results->acc1[BLACK], 0, sizeof(float) * N_HIDDEN);

  uint64_t bb = board->occupancies;
  int p = 0;

  while (bb) {
    Square sq = lsb(bb);
    Piece pc = getPiece(board->pieces, p++);

    Feature wf = idx(pc, sq, board->kings[WHITE], WHITE);
    Feature bf = idx(pc, sq, board->kings[BLACK], BLACK);

    for (size_t j = 0; j < N_HIDDEN; j++) {
      results->acc1[WHITE][j] += nn->inputWeights[wf * N_HIDDEN + j];
      results->acc1[BLACK][j] += nn->inputWeights[bf * N_HIDDEN + j];
    }

    popLsb(bb);
  }
}

void NNPredict(NN* nn, Board* board, NNAccumulators* results) {
  results->output = 0.0f;

  // Apply first layer
  memcpy(results->acc1[WHITE], nn->inputBiases, sizeof(float) * N_HIDDEN);
  memcpy(results->acc1[BLACK], nn->inputBiases, sizeof(float) * N_HIDDEN);

  uint64_t bb = board->occupancies;
  int p = 0;

  while (bb) {
    Square sq = lsb(bb);
    Piece pc = getPiece(board->pieces, p++);

    Feature wf = idx(pc, sq, board->kings[WHITE], WHITE);
    Feature bf = idx(pc, sq, board->kings[BLACK], BLACK);

    for (size_t j = 0; j < N_HIDDEN; j++) {
      results->acc1[WHITE][j] += nn->inputWeights[wf * N_HIDDEN + j];
      results->acc1[BLACK][j] += nn->inputWeights[bf * N_HIDDEN + j];
    }

    results->output += nn->skipWeights[board->stm == WHITE ? wf : bf];

    popLsb(bb);
  }

  ReLU(results->acc1[WHITE], N_HIDDEN);
  ReLU(results->acc1[BLACK], N_HIDDEN);

  results->output += DotProduct(results->acc1[board->stm], nn->outputWeights, N_HIDDEN) +
                     DotProduct(results->acc1[board->stm ^ 1], nn->outputWeights + N_HIDDEN, N_HIDDEN) + //
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
  fread(nn->skipWeights, sizeof(float), N_INPUT, fp);

  fclose(fp);

  return nn;
}

NN* LoadRandomNN() {
  srand(time(NULL));
  NN* nn = malloc(sizeof(NN));

  for (int i = 0; i < N_INPUT * N_HIDDEN; i++)
    nn->inputWeights[i] = Random(N_INPUT * N_HIDDEN);

  for (int i = 0; i < N_HIDDEN; i++)
    nn->inputBiases[i] = Random(N_HIDDEN);

  for (int i = 0; i < N_HIDDEN * 2; i++)
    nn->outputWeights[i] = Random(N_HIDDEN * 2);

  nn->outputBias = Random(1);

  for (int i = 0; i < N_INPUT; i++)
    nn->skipWeights[i] = Random(N_INPUT);

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
  fwrite(nn->skipWeights, sizeof(float), N_INPUT, fp);

  fclose(fp);
}