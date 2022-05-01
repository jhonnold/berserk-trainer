#include "trainer.h"

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

  NNGradients* gradients = malloc(sizeof(NNGradients));
  ClearGradients(gradients);

  BatchGradients* local = malloc(sizeof(BatchGradients) * THREADS);

  printf("Calculating Validation Error...\n");
  float error = TotalError(validation, nn);
  printf("Starting Error: [%1.8f]\n", error);

  int totalBatches = 1, saveRate = data->n / BATCH_SIZE < 6144 ? data->n / BATCH_SIZE : 6144;

  for (int epoch = 1; epoch <= 22; epoch++) {
    long epochStart = GetTimeMS();

    printf("\rShuffling...");
    ShuffleData(data);

    uint32_t batches = data->n / BATCH_SIZE;
    for (uint32_t b = 0; b < batches; b++, totalBatches++) {
      uint8_t active[N_INPUT] = {0};
      ITERATION++;

      float e = Train(b, data, nn, local, active);
      ApplyGradients(nn, gradients, local, active);

      if ((b + 1) % 50 == 0) printf("\rBatch: [#%d/%d], Error: [%1.8f]", b + 1, batches, e);

      if (totalBatches % saveRate == 0) {
        char buffer[64];
        sprintf(buffer, "../nets/berserk-kb.e%d.2x%d.nn", totalBatches / saveRate, N_HIDDEN);
        SaveNN(nn, buffer);
      }
    }

    float newError = TotalError(validation, nn);

    long now = GetTimeMS();
    printf("\rEpoch: [#%5d], Error: [%1.8f], Delta: [%+1.8f], LR: [%.5f], Speed: [%9.0f pos/s], Time: [%lds]\n", epoch,
           newError, error - newError, ALPHA, 1000.0 * data->n / (now - epochStart), (now - epochStart) / 1000);

    error = newError;

    // LR DROP
    if (epoch == 20) ALPHA = 0.001;

    if (epoch == 21) ALPHA = 0.0001;
  }
}

float TotalError(DataSet* data, NN* nn) {
  float e = 0.0;

#pragma omp parallel for schedule(static) num_threads(THREADS) reduction(+ : e)
  for (uint32_t i = 0; i < data->n; i++) {
    Board* board = &data->entries[i];

    NetworkTrace trace[1];
    Features f[1];

    ToFeatures(board, f);
    NNPredict(nn, f, board->stm, trace);

    e += Error(Sigmoid(trace->output), board);
  }

  return e / data->n;
}

float Train(int batch, DataSet* data, NN* nn, BatchGradients* local, uint8_t* active) {
#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int t = 0; t < THREADS; t++) memset(&local[t], 0, sizeof(BatchGradients));

  uint8_t actives[THREADS][N_INPUT] = {0};
  float e = 0.0;

#pragma omp parallel for schedule(static) num_threads(THREADS) reduction(+ : e)
  for (int n = 0; n < BATCH_SIZE; n++) {
    const int t = omp_get_thread_num();

    Board board = data->entries[n + batch * BATCH_SIZE];

    NetworkTrace trace[1];
    Features f[1];

    ToFeatures(&board, f);
    NNPredict(nn, f, board.stm, trace);

    float out = Sigmoid(trace->output);
    e += Error(out, &board);

    // LOSS CALCULATIONS ------------------------------------------------------------------------
    float outputLoss = SigmoidPrime(out) * ErrorGradient(out, &board);

    float hiddenLosses[2 * N_HIDDEN];
    for (int i = 0; i < 2 * N_HIDDEN; i++)
      hiddenLosses[i] = outputLoss * nn->outputWeights[i] * ReLUPrime(trace->accumulator[i]);
    // ------------------------------------------------------------------------------------------

    // OUTPUT LAYER GRADIENTS -------------------------------------------------------------------
    local[t].outputBias += outputLoss;
    for (int i = 0; i < 2 * N_HIDDEN; i++) local[t].outputWeights[i] += trace->accumulator[i] * outputLoss;
    // ------------------------------------------------------------------------------------------

    // INPUT LAYER GRADIENTS --------------------------------------------------------------------
    float lassos[2 * N_HIDDEN];
    for (int i = 0; i < 2 * N_HIDDEN; i++) lassos[i] = LAMBDA * (trace->accumulator[i] > 0);

    float* stmLosses = hiddenLosses;
    float* xstmLosses = &hiddenLosses[N_HIDDEN];

    float* stmLassos = lassos;
    float* xstmLassos = &lassos[N_HIDDEN];

    for (int i = 0; i < N_HIDDEN; i++)
      local[t].inputBiases[i] += stmLosses[i] + xstmLosses[i] + stmLassos[i] + xstmLassos[i];

    for (int i = 0; i < f->n; i++) {
      actives[t][f->features[i][board.stm]] = actives[t][f->features[i][board.stm ^ 1]] = 1;

      for (int j = 0; j < N_HIDDEN; j++) {
        local[t].inputWeights[f->features[i][board.stm] * N_HIDDEN + j] += stmLosses[j] + stmLassos[j];
        local[t].inputWeights[f->features[i][board.stm ^ 1] * N_HIDDEN + j] += xstmLosses[j] + xstmLassos[j];
      }
    }
    // ------------------------------------------------------------------------------------------
  }

  for (int t = 0; t < THREADS; t++)
    for (int i = 0; i < N_INPUT; i++) active[i] |= actives[t][i];

  return e / BATCH_SIZE;
}