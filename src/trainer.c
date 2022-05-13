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
  setbuf(stdin, NULL);
  setbuf(stdout, NULL);

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
    WriteToFile(outputPath, samplesPath, entries);
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

  int batchesPerDataset = entries / BATCH_SIZE;
  int diskLoads = floor((double)batchesPerDataset / BATCHES_PER_LOAD);

  int epoch = 0;
  while (++epoch <= 400) {
    long epochStart = GetTimeMS();

    int d = epoch % diskLoads;
    if (binaryRead)
      LoadEntriesBinary(samplesPath, data, BATCH_SIZE * BATCHES_PER_LOAD, d * BATCH_SIZE * BATCHES_PER_LOAD);
    else
      LoadEntries(samplesPath, data, BATCH_SIZE * BATCHES_PER_LOAD, d * BATCH_SIZE * BATCHES_PER_LOAD);

    printf("\rShuffling...");
    ShuffleData(data);

    for (int b = 0; b < BATCHES_PER_LOAD; b++) {
      uint8_t active[N_INPUT] = {0};
      ITERATION++;

      float be = Train(b, data, nn, local, active);
      ApplyGradients(nn, gradients, local, active);

      long now = GetTimeMS();
      printf("\rBatch: [#%d/%d], Error: [%1.8f], Speed: [%9.0f pos/s]", b + 1, BATCHES_PER_LOAD, be,
             1000.0 * (b + 1) * BATCH_SIZE / (now - epochStart));
    }

    char buffer[64];
    sprintf(buffer, "../nets/berserk-kq.e%d.2x%d.nn", epoch, N_HIDDEN);
    SaveNN(nn, buffer);

    float newError = TotalError(validation, nn);

    long now = GetTimeMS();
    printf("\rEpoch: [#%5d], Error: [%1.8f], Delta: [%+1.8f], LR: [%.8f], Time: [%lds], Speed: [%9.0f pos/s]\n", epoch,
           newError, error - newError, ALPHA, (now - epochStart) / 1000,
           1000.0 * BATCHES_PER_LOAD * BATCH_SIZE / (now - epochStart));

    error = newError;

    if (epoch == 250 || epoch == 325)
      ALPHA /= 10.0f;
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

    float hiddenLosses[N_L1];
    for (int i = 0; i < N_L1; i++)
      hiddenLosses[i] = outputLoss * nn->outputWeights[i] * ReLUPrime(trace->accumulator[i]);
    // ------------------------------------------------------------------------------------------

    // OUTPUT LAYER GRADIENTS -------------------------------------------------------------------
    local[t].outputBias += outputLoss;
    for (int i = 0; i < N_L1; i++) local[t].outputWeights[i] += trace->accumulator[i] * outputLoss;
    // ------------------------------------------------------------------------------------------

    // INPUT LAYER GRADIENTS --------------------------------------------------------------------
    float lassos[N_L1];
    for (int i = 0; i < N_L1; i++) lassos[i] = LAMBDA * (trace->accumulator[i] > 0);

    float* stmLosses = hiddenLosses;
    float* xstmLosses = &hiddenLosses[N_HIDDEN];

    float* stmLassos = lassos;
    float* xstmLassos = &lassos[N_HIDDEN];

    for (int i = 0; i < N_HIDDEN; i++)
      local[t].inputBiases[i] += stmLosses[i] + xstmLosses[i] + stmLassos[i] + xstmLassos[i];

    for (int i = 0; i < f->n; i++) {
      int f1 = f->features[i][board.stm];
      int f2 = f->features[i][board.stm ^ 1];

      actives[t][f1] = actives[t][f2] = 1;

      for (int j = 0; j < N_HIDDEN; j++) {
        local[t].inputWeights[f1 * N_HIDDEN + j] += stmLosses[j] + stmLassos[j];
        local[t].inputWeights[f2 * N_HIDDEN + j] += xstmLosses[j] + xstmLassos[j];
      }
    }
    // ------------------------------------------------------------------------------------------
  }

  for (int t = 0; t < THREADS; t++)
    for (int i = 0; i < N_INPUT; i++) active[i] |= actives[t][i];

  return e / BATCH_SIZE;
}