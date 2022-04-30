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

  for (int epoch = 1; epoch <= 22; epoch++) {
    long epochStart = GetTimeMS();

    printf("Shuffling...\n");
    ShuffleData(data);

    uint32_t batches = data->n / BATCH_SIZE;
    for (uint32_t b = 0; b < batches; b++) {
      float e = Train(b, data, nn, local);
      ApplyGradients(nn, gradients, local);

      if (b == 0 || (b + 1) % 250 == 0) {
        long now = GetTimeMS();

        printf("Batch: [#%d/%d], Error: [%1.8f], Speed: [%9.0f pos/s]\n", b + 1, batches, e,
               1000.0f * BATCH_SIZE * (b + 1) / (now - epochStart));
      }
    }

    char buffer[64];
    sprintf(buffer, "../nets/berserk-kq.e%d.2x%d.%d.%d.nn", epoch, N_HIDDEN, N_HIDDEN_2, N_HIDDEN_3);
    SaveNN(nn, buffer);

    printf("Calculating Validation Error...\n");
    float newError = TotalError(validation, nn);

    long now = GetTimeMS();
    printf("Epoch: [#%5d], Error: [%1.8f], Delta: [%+1.8f], LR: [%.5f], Speed: [%9.0f pos/s], Time: [%lds]\n", epoch,
           newError, error - newError, ALPHA, 1000.0f * data->n / (now - epochStart), (now - epochStart) / 1000);

    error = newError;

    // LR DROP
    if (epoch == 20) ALPHA = 0.001f;

    if (epoch == 21) ALPHA = 0.0001f;
  }
}

float TotalError(DataSet* data, NN* nn) {
  float e = 0.0f;

#pragma omp parallel for schedule(auto) num_threads(THREADS) reduction(+ : e)
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

float Train(int batch, DataSet* data, NN* nn, BatchGradients* local) {
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int t = 0; t < THREADS; t++) memset(&local[t], 0, sizeof(BatchGradients));

  float e = 0.0;

#pragma omp parallel for schedule(auto) num_threads(THREADS) reduction(+ : e)
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

    float l2Losses[N_HIDDEN_3];
    for (int i = 0; i < N_HIDDEN_3; i++)
      l2Losses[i] = outputLoss * nn->outputWeights[i] * ReLUPrime(trace->l2Accumulator[i]);

    float l1Losses[N_HIDDEN_2] = {0};
    for (int i = 0; i < N_HIDDEN_2; i++)
      for (int j = 0; j < N_HIDDEN_3; j++)
        l1Losses[i] += l2Losses[j] * nn->l2Weights[j * N_HIDDEN_2 + i] * ReLUPrime(trace->l1Accumulator[i]);

    float hiddenLosses[2 * N_HIDDEN] = {0};
    for (int i = 0; i < 2 * N_HIDDEN; i++)
      for (int j = 0; j < N_HIDDEN_2; j++)
        hiddenLosses[i] += l1Losses[j] * nn->l1Weights[j * 2 * N_HIDDEN + i] * ReLUPrime(trace->accumulator[i]);
    // ------------------------------------------------------------------------------------------

    // OUTPUT LAYER GRADIENTS -------------------------------------------------------------------
    local[t].outputBias += outputLoss;
    for (int i = 0; i < N_HIDDEN_3; i++) local[t].outputWeights[i] += trace->l2Accumulator[i] * outputLoss;
    // ------------------------------------------------------------------------------------------

    // L2 LAYER GRADIENTS -------------------------------------------------------------------
    for (int i = 0; i < N_HIDDEN_3; i++) local[t].l2Biases[i] += l2Losses[i];

    for (int i = 0; i < N_HIDDEN_3; i++)
      for (int j = 0; j < N_HIDDEN_2; j++)
        local[t].l2Weights[i * N_HIDDEN_2 + j] += trace->l1Accumulator[j] * l2Losses[i];
    // ------------------------------------------------------------------------------------------

    // L1 LAYER GRADIENTS -------------------------------------------------------------------
    for (int i = 0; i < N_HIDDEN_2; i++) local[t].l1Biases[i] += l1Losses[i];

    for (int i = 0; i < N_HIDDEN_2; i++)
      for (int j = 0; j < 2 * N_HIDDEN; j++)
        local[t].l1Weights[i * 2 * N_HIDDEN + j] += trace->accumulator[j] * l1Losses[i];
    // ------------------------------------------------------------------------------------------

    // INPUT LAYER GRADIENTS --------------------------------------------------------------------
    float* stmLosses = hiddenLosses;
    float* xstmLosses = &hiddenLosses[N_HIDDEN];

    for (int i = 0; i < N_HIDDEN; i++) local[t].inputBiases[i] += stmLosses[i] + xstmLosses[i];

    for (int i = 0; i < f->n; i++) {
      for (int j = 0; j < N_HIDDEN; j++) {
        local[t].inputWeights[f->features[i][board.stm] * N_HIDDEN + j] += stmLosses[j];
        local[t].inputWeights[f->features[i][board.stm ^ 1] * N_HIDDEN + j] += xstmLosses[j];
      }
    }
    // ------------------------------------------------------------------------------------------
  }

  return e / BATCH_SIZE;
}