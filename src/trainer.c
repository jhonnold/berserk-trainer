#include "trainer.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "bits.h"
#include "board.h"
#include "data.h"
#include "gradients.h"
#include "nn.h"
#include "random.h"
#include "util.h"

int main(int argc, char** argv) {
  SeedRandom();

  int c;

  char nnPath[128] = {0};
  char entriesPath[128] = {0};

  while ((c = getopt(argc, argv, "d:n:")) != -1) {
    switch (c) {
      case 'd':
        strcpy(entriesPath, optarg);
        break;
      case 'n':
        strcpy(nnPath, optarg);
        break;
      case '?':
        return 1;
    }
  }

  if (!entriesPath[0]) {
    printf("No data file specified!\n");
    return 1;
  }

  NN* nn;
  if (!nnPath[0]) {
    printf("No net specified, generating a random net.\n");
    nn = LoadRandomNN();
  } else {
    printf("Loading net from %s\n", nnPath);
    nn = LoadNN(nnPath);
  }

  printf("Loading entries from %s\n", entriesPath);

  DataSet* validation = malloc(sizeof(DataSet));
  validation->n = 0;
  validation->entries = malloc(sizeof(Board) * VALIDATION_POSITIONS);
  LoadEntries(entriesPath, validation, VALIDATION_POSITIONS, 0);

  DataSet* data = malloc(sizeof(DataSet));
  data->n = 0;
  data->entries = malloc(sizeof(Board) * MAX_POSITIONS);
  LoadEntries(entriesPath, data, MAX_POSITIONS, VALIDATION_POSITIONS);

  Optimizer* optimizer = malloc(sizeof(Optimizer));
  memset(optimizer, 0, sizeof(Optimizer));

  Gradients* gradients = malloc(sizeof(Gradients) * THREADS);

  printf("Calculating Validation Error...\n");
  float error = TotalError(validation, nn);
  printf("Starting Error: [%1.8f]\n", error);

  for (int epoch = 1; epoch <= 22; epoch++) {
    long epochStart = GetTimeMS();

    printf("Shuffling...\n");
    ShuffleData(data);

    uint32_t batches = data->n / BATCH_SIZE;
    for (uint32_t b = 0; b < batches; b++) {
      Train(b, data, nn, gradients);
      ApplyGradients(nn, optimizer, gradients);

      if ((b + 1) % 1000 == 0) printf("Batch: [#%d/%d]\n", b + 1, batches);
    }

    char buffer[64];
    sprintf(buffer, "../nets/berserk-kq.e%d.2x%d.nn", epoch, N_HIDDEN);
    SaveNN(nn, buffer);

    printf("Calculating Validation Error...\n");
    float newError = TotalError(validation, nn);

    long now = GetTimeMS();
    printf("Epoch: [#%5d], Error: [%1.8f], Delta: [%+1.8f], LR: [%.5f], Speed: [%9.0f pos/s], Time: [%lds]\n", epoch,
           newError, error - newError, ALPHA, 1000.0 * data->n / (now - epochStart), (now - epochStart) / 1000);

    error = newError;

    // LR DROP
    if (epoch == 20) {
      ALPHA = 0.001;
      memset(optimizer, 0, sizeof(Optimizer));
    } else if (epoch == 21) {
      ALPHA = 0.0001;
      memset(optimizer, 0, sizeof(Optimizer));
    }
  }
}

float TotalError(DataSet* data, NN* nn) {
  float e = 0.0;

#pragma omp parallel for schedule(static) num_threads(THREADS) reduction(+ : e)
  for (uint32_t i = 0; i < data->n; i++) {
    Board* board = &data->entries[i];

    NNAccumulators results[1];
    Features f[1];

    ToFeatures(board, f);
    NNPredict(nn, f, board->stm, results);

    e += Error(Sigmoid(results->output), board);
  }

  return e / data->n;
}

void Train(int batch, DataSet* data, NN* nn, Gradients* gradients) {
  for (int t = 0; t < THREADS; t++) memset(&gradients[t], 0, sizeof(Gradients));

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int n = 0; n < BATCH_SIZE; n++) {
    const int t = omp_get_thread_num();

    Board board = data->entries[n + batch * BATCH_SIZE];

    NNAccumulators activations[1];
    Features f[1];

    ToFeatures(&board, f);
    NNPredict(nn, f, board.stm, activations);

    float out = Sigmoid(activations->output);
    float delta = SigmoidPrime(out) * ErrorGradient(out, &board);

    // LOSS CALCULATIONS ------------------------------------------------------------------------
    const size_t WIDTH = sizeof(__m256) / sizeof(float);
    const size_t CHUNKS = N_HIDDEN / WIDTH;

    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0);
    const __m256 lambda = _mm256_set1_ps(LAMBDA);

    __m256 outputLoss = _mm256_set1_ps(delta);

    __m256 hiddenLosses[2 * CHUNKS] ALIGN64;

    __m256* acc = (__m256*)activations->acc1;
    __m256* nnOutputWeights = (__m256*)nn->outputWeights;

    for (size_t i = 0; i < 2 * CHUNKS; i++) {
      __m256 reluPrime = _mm256_blendv_ps(zero, one, _mm256_cmp_ps(acc[i], zero, 30));

      hiddenLosses[i] = _mm256_mul_ps(outputLoss, _mm256_mul_ps(nnOutputWeights[i], reluPrime));
    }
    // ------------------------------------------------------------------------------------------

    // OUTPUT LAYER GRADIENTS -------------------------------------------------------------------
    gradients[t].outputBias += delta;

    __m256* outputWeights = (__m256*)gradients[t].outputWeights;
    for (size_t i = 0; i < 2 * CHUNKS; i++)
      outputWeights[i] = _mm256_add_ps(outputWeights[i], _mm256_mul_ps(acc[i], outputLoss));
    // ------------------------------------------------------------------------------------------

    // INPUT LAYER GRADIENTS --------------------------------------------------------------------
    __m256* stmActivations = &acc[0];
    __m256* xstmActivations = &acc[CHUNKS];

    __m256* stmLosses = &hiddenLosses[0];
    __m256* xstmLosses = &hiddenLosses[CHUNKS];

    __m256* biasGradients = (__m256*)gradients[t].inputBiases;
    __m256* weightGradients = (__m256*)gradients[t].inputWeights;

    for (size_t i = 0; i < CHUNKS; i++) {
      __m256 stmLasso = _mm256_blendv_ps(zero, lambda, _mm256_cmp_ps(stmActivations[i], zero, 30));
      __m256 xstmLasso = _mm256_blendv_ps(zero, lambda, _mm256_cmp_ps(xstmActivations[i], zero, 30));

      __m256 lassos = _mm256_add_ps(stmLasso, xstmLasso);
      __m256 losses = _mm256_add_ps(stmLosses[i], xstmLosses[i]);

      biasGradients[i] = _mm256_add_ps(biasGradients[i], _mm256_add_ps(lassos, losses));
    }

    for (int i = 0; i < f->n; i++) {
      for (size_t j = 0; j < CHUNKS; j++) {
        __m256 stmLasso = _mm256_blendv_ps(zero, lambda, _mm256_cmp_ps(stmActivations[j], zero, 30));
        __m256 xstmLasso = _mm256_blendv_ps(zero, lambda, _mm256_cmp_ps(xstmActivations[j], zero, 30));

        weightGradients[f->features[i][board.stm] * CHUNKS + j] = _mm256_add_ps(
            weightGradients[f->features[i][board.stm] * CHUNKS + j], _mm256_add_ps(stmLosses[j], stmLasso));
        weightGradients[f->features[i][board.stm ^ 1] * CHUNKS + j] = _mm256_add_ps(
            weightGradients[f->features[i][board.stm ^ 1] * CHUNKS + j], _mm256_add_ps(xstmLosses[j], xstmLasso));
      }
    }
    // ------------------------------------------------------------------------------------------
  }
}