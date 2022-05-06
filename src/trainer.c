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

  uint8_t binaryRead = 0;

  uint64_t entries = 1000000000;
  uint64_t validations = 1000000;

  char baseNetworkPath[128] = {0};
  char samplesPath[128] = {0};
  char validationsPath[128] = {0};

  uint8_t writing = 0;
  char outputPath[128] = {0};

  int c;
  while ((c = getopt(argc, argv, "bc:v:z:w:d:n:")) != -1) {
    switch (c) {
      case 'd':
        strcpy(samplesPath, optarg);
        break;
      case 'c':
        entries = atoll(optarg);
        break;
      case 'v':
        strcpy(validationsPath, optarg);
        break;
      case 'z':
        validations = atoll(optarg);
        break;
      case 'n':
        strcpy(baseNetworkPath, optarg);
        break;
      case 'w':
        strcpy(outputPath, optarg);
        writing = 1;
        break;
      case 'b':
        binaryRead = 1;
        break;
      case '?':
        return 1;
    }
  }

  if (!samplesPath[0]) {
    printf("No data file specified!\n");
    return 1;
  }

  if (writing) {
    WriteToFile(outputPath, samplesPath);
    exit(0);
  }

  NN* nn;
  if (!baseNetworkPath[0]) {
    printf("No net specified, generating a random net.\n");
    nn = LoadRandomNN();
  } else {
    printf("Loading net from %s\n", baseNetworkPath);
    nn = LoadNN(baseNetworkPath);
  }

  printf("Loading entries from %s\n", samplesPath);

  DataSet* validation = malloc(sizeof(DataSet));
  validation->entries = NULL;
  validation->n = 0;

  DataSet* data = malloc(sizeof(DataSet));
  data->entries = malloc(sizeof(Board) * BATCHES_PER_LOAD * BATCH_SIZE);
  data->n = 0;

  if (binaryRead) {
    LoadEntriesBinary(validationsPath, validation, validations, 0);
  } else {
    LoadEntries(validationsPath, validation, validations, 0);
  }

  NNGradients* gradients = malloc(sizeof(NNGradients));
  ClearGradients(gradients);

  BatchGradients* local = malloc(sizeof(BatchGradients) * THREADS);

  printf("Calculating Validation Error...\n");
  float error = TotalError(validation, nn);
  printf("Starting Error: [%1.8f]\n", error);

  for (int epoch = 1; epoch <= 22; epoch++) {
    long epochStart = GetTimeMS();

    int totalBatches = 1;
    int batches = entries / BATCH_SIZE;
    int diskLoads = ceilf((double)batches / BATCHES_PER_LOAD);

    for (int d = 0; d < diskLoads; d++) {
      int loadCount = min(BATCH_SIZE * BATCHES_PER_LOAD, entries - d * (BATCH_SIZE * BATCHES_PER_LOAD));

      if (binaryRead)
        LoadEntriesBinary(samplesPath, data, loadCount, d * loadCount);
      else
        LoadEntries(samplesPath, data, loadCount, d * loadCount);

      printf("Shuffling...\n");
      ShuffleData(data);

      int diskLoadBatches = data->n / BATCH_SIZE;
      for (int b = 0; b < diskLoadBatches; b++, totalBatches++) {
        float be = Train(b, data, nn, gradients, local);
        ApplyGradients(nn, gradients, local);

        long now = GetTimeMS();
        printf("Batch: [#%d/%d], Error: [%1.8f], Speed: [%9.0f pos/s]\n", totalBatches, batches, be,
               1000.0 * totalBatches * BATCH_SIZE / (now - epochStart));
      }
    }

    char buffer[64];
    sprintf(buffer, "../nets/berserk-kq.e%d.2x%d.%d.%d.nn", epoch, N_HIDDEN, N_HIDDEN_2, N_HIDDEN_3);
    SaveNN(nn, buffer);

    printf("Calculating Validation Error...\n");
    float newError = TotalError(validation, nn);

    long now = GetTimeMS();
    printf("Epoch: [#%5d], Error: [%1.8f], Delta: [%+1.8f], LR: [%.5f], Time: [%lds]\n", epoch, newError,
           error - newError, ALPHA, (now - epochStart) / 1000);

    error = newError;

    // LR DROP
    if (epoch == 20) ALPHA = 0.001f;

    if (epoch == 21) ALPHA = 0.0001f;
  }
}

float TotalError(DataSet* data, NN* nn) {
  float e = 0.0f;

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