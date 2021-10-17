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

void NNPredict(NN* nn, Board board, NNActivations* results, int stm) {
  // Apply first layer
  memcpy(results->accumulators[WHITE], nn->hiddenBiases, sizeof(float) * N_HIDDEN);
  memcpy(results->accumulators[BLACK], nn->hiddenBiases, sizeof(float) * N_HIDDEN);

  for (int i = 0; i < 32; i++) {
    if (board[i].pc < 0)
      break;

    int wf = feature(board[i], WHITE);
    int bf = feature(board[i], BLACK);

    for (int j = 0; j < N_HIDDEN; j += 8) {
      __m256 weights = _mm256_load_ps(&nn->featureWeights[wf * N_HIDDEN + j]);
      __m256 neurons = _mm256_load_ps(&results->accumulators[WHITE][j]);
      _mm256_store_ps(&results->accumulators[WHITE][j], _mm256_add_ps(weights, neurons));

      weights = _mm256_load_ps(&nn->featureWeights[bf * N_HIDDEN + j]);
      neurons = _mm256_load_ps(&results->accumulators[BLACK][j]);
      _mm256_store_ps(&results->accumulators[BLACK][j], _mm256_add_ps(weights, neurons));
    }
  }

  // Apply second layer
  const __m256 zero = _mm256_setzero_ps();
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();

  for (size_t j = 0; j < N_HIDDEN; j += 8) {
    const __m256 stmActivations = _mm256_max_ps(_mm256_load_ps(results->accumulators[stm] + j), zero);
    const __m256 stmWeights = _mm256_load_ps(nn->hiddenWeights + j);
    _mm256_store_ps(results->accumulators[stm] + j, stmActivations);

    const __m256 xstmActivations = _mm256_max_ps(_mm256_load_ps(results->accumulators[stm ^ 1] + j), zero);
    const __m256 xstmWeights = _mm256_load_ps(nn->hiddenWeights + (j + N_HIDDEN));
    _mm256_store_ps(results->accumulators[stm ^ 1] + j, xstmActivations);

    s0 = _mm256_add_ps(s0, _mm256_mul_ps(stmActivations, stmWeights));
    s1 = _mm256_add_ps(s1, _mm256_mul_ps(xstmActivations, xstmWeights));
  }

  const __m256 r8 = _mm256_add_ps(s0, s1);
  const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
  const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
  const __m128 r1 = _mm_add_ss(r2, _mm_shuffle_ps(r2, r2, 0x1));
  const float sum = _mm_cvtss_f32(r1);

  results->result = sum + nn->outputBias;
}

uint64_t NetworkHash(NN* nn) {
  uint64_t hash = 0;
  for (int i = 0; i < N_HIDDEN * N_FEATURES; i++) {
    int v = *((int*)&nn->featureWeights[i]);
    hash = (hash + (324723947ULL + v)) ^ 93485734985ULL;
  }

  for (int i = 0; i < N_HIDDEN; i++) {
    int v = *((int*)&nn->hiddenBiases[i]);
    hash = (hash + (324723947ULL + v)) ^ 93485734985ULL;
  }

  for (int i = 0; i < N_HIDDEN * 2; i++) {
    int v = *((int*)&nn->hiddenWeights[i]);
    hash = (hash + (324723947ULL + v)) ^ 93485734985ULL;
  }

  int v = *((int*)&nn->outputBias);
  return (hash + (324723947ULL + v)) ^ 93485734985ULL;
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
  printf("Reading network with hash %ld\n", hash);

  NN* nn = malloc(sizeof(NN));

  fread(nn->featureWeights, sizeof(float), N_FEATURES * N_HIDDEN, fp);
  fread(nn->hiddenBiases, sizeof(float), N_HIDDEN, fp);
  fread(nn->hiddenWeights, sizeof(float), N_HIDDEN * 2, fp);
  fread(&nn->outputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);

  return nn;
}

NN* LoadRandomNN() {
  srand(time(NULL));

  NN* nn = malloc(sizeof(NN));

  float max = sqrtf(2.0f / (N_FEATURES * N_HIDDEN));
  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    nn->featureWeights[i] = rand() * max / RAND_MAX;

  max = sqrtf(2.0f / N_HIDDEN);
  for (int i = 0; i < N_HIDDEN; i++)
    nn->hiddenBiases[i] = rand() * max / RAND_MAX;

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

  fwrite(&NETWORK_MAGIC, sizeof(int), 1, fp);
  
  uint64_t hash = NetworkHash(nn);
  fwrite(&hash, sizeof(uint64_t), 1, fp);

  fwrite(nn->featureWeights, sizeof(float), N_FEATURES * N_HIDDEN, fp);
  fwrite(nn->hiddenBiases, sizeof(float), N_HIDDEN, fp);
  fwrite(nn->hiddenWeights, sizeof(float), N_HIDDEN * 2, fp);
  fwrite(&nn->outputBias, sizeof(float), N_OUTPUT, fp);

  fclose(fp);
}