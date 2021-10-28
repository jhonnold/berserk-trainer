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
#include "trainer.h"
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

  DataSet* data = malloc(sizeof(DataSet));
  data->n = 0;
  data->entries = malloc(sizeof(DataEntry) * MAX_POSITIONS);
  LoadEntries(entriesPath, data);

  NNGradients* gradients = malloc(sizeof(NNGradients));
  ClearGradients(gradients);

  BatchGradients* local = malloc(sizeof(BatchGradients) * THREADS);

  float error = TotalError(data, nn);
  printf("Starting Error: [%1.8f]\n", error);

  for (int epoch = 1; epoch <= 250; epoch++) {
    long epochStart = GetTimeMS();

    printf("Shuffling...\r");
    ShuffleData(data);

    int batches = data->n / BATCH_SIZE;
    for (int b = 0; b < batches; b++) {
      Train(b, data, nn, gradients, local);
      ApplyGradients(nn, gradients);

      printf("Batch: [#%5d]\r", b + 1);
    }

    char buffer[64];
    sprintf(buffer, "../nets/berserk-kq.e%d.%d.2x%d.nn", epoch, N_FEATURES, N_HIDDEN);
    SaveNN(nn, buffer);

    printf("Calculating Error...\r");
    float newError = TotalError(data, nn);

    long now = GetTimeMS();
    printf("Epoch: [#%5d], Error: [%1.8f], Delta: [%+1.8f], LR: [%.5f], Speed: [%9.0f pos/s], Time: [%lds]\n", epoch,
           newError, error - newError, ALPHA, 1000.0 * data->n / (now - epochStart), (now - epochStart) / 1000);

    error = newError;
  }
}

float TotalError(DataSet* data, NN* nn) {
  float e = 0.0f;

#pragma omp parallel for schedule(auto) num_threads(THREADS) reduction(+ : e)
  for (int i = 0; i < data->n; i++) {
    DataEntry entry = data->entries[i];

    NNAccumulators acc[1];
    NNPredict(nn, &entry.board, acc);

    e += Error(Sigmoid(acc->output), &entry);
  }

  return e / data->n;
}

void Train(int batch, DataSet* data, NN* nn, NNGradients* g, BatchGradients* local) {
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int t = 0; t < THREADS; t++)
    memset(&local[t], 0, sizeof(BatchGradients));

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int n = 0; n < BATCH_SIZE; n++) {
    const int t = omp_get_thread_num();

    DataEntry entry = data->entries[n + batch * BATCH_SIZE];
    Board board = entry.board;

    NNAccumulators acc[1];
    NNPredict(nn, &board, acc);

    float out = Sigmoid(acc->output);

    // output loss
    float outputLoss = SigmoidPrime(out) * ErrorGradient(out, &entry);

    // hidden layer losses
    float hiddenLosses[N_HIDDEN_2] = {0};
    for (int i = 0; i < N_HIDDEN_2; i++)
      hiddenLosses[i] += (acc->hidden[i] > 0.0f) * outputLoss * nn->outputWeights[i];

    // input layer losses
    float inputLosses[2][N_HIDDEN] = {0};
    for (int i = 0; i < N_HIDDEN; i++)
      for (int j = 0; j < N_HIDDEN_2; j++) {
        inputLosses[WHITE][i] += (acc->input[WHITE][i] > 0.0f && acc->input[WHITE][i] < 1.0f) * hiddenLosses[j] *
                                 nn->hiddenWeights[2 * N_HIDDEN * j + i];
        inputLosses[BLACK][i] += (acc->input[BLACK][i] > 0.0f && acc->input[BLACK][i] < 1.0f) * hiddenLosses[j] *
                                 nn->hiddenWeights[2 * N_HIDDEN * j + N_HIDDEN + i];
      }

    // input layer gradients
    for (int i = 0; i < board.n; i++) {
      for (int j = 0; j < N_HIDDEN; j++) {
        local[t].inputWeights[idx(board.pieces[i], board.wk, WHITE) * N_HIDDEN + j] += inputLosses[WHITE][j];
        local[t].inputWeights[idx(board.pieces[i], board.bk, BLACK) * N_HIDDEN + j] += inputLosses[BLACK][j];
        local[t].inputBiases[j] += inputLosses[WHITE][j] + inputLosses[BLACK][j];
      }

      // skip weights get loss from output
      local[t].skipWeights[idx(board.pieces[i], board.wk, WHITE)] += outputLoss;
    }

    // hidden layer gradients
    for (int i = 0; i < N_HIDDEN; i++)
      for (int j = 0; j < N_HIDDEN_2; j++) {
        local[t].hiddenWeights[2 * N_HIDDEN * j + i] += hiddenLosses[j] * acc->input[WHITE][i];
        local[t].hiddenWeights[2 * N_HIDDEN * j + i + N_HIDDEN] += hiddenLosses[j] * acc->input[BLACK][i];
        local[t].hiddenBiases[j] += hiddenLosses[j];
      }

    // output layer gradients
    for (int i = 0; i < N_HIDDEN_2; i++)
      local[t].outputWeights[i] += outputLoss * acc->hidden[i];
    local[t].outputBias += outputLoss;
  }

  for (int t = 0; t < THREADS; t++) {
    for (int i = 0; i < N_FEATURES; i++)
      g->skipWeightGradients[i].g += local[t].skipWeights[i];

    for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
      g->inputWeightGradients[i].g += local[t].inputWeights[i];

    for (int i = 0; i < N_HIDDEN; i++)
      g->inputBiasGradients[i].g += local[t].inputBiases[i];

    for (int i = 0; i < 2 * N_HIDDEN * N_HIDDEN_2; i++)
      g->hiddenWeightGradients[i].g += local[t].hiddenWeights[i];

    for (int i = 0; i < N_HIDDEN_2; i++)
      g->hiddenBiasGradients[i].g += local[t].hiddenBiases[i];

    for (int i = 0; i < N_HIDDEN_2; i++)
      g->outputWeightGradients[i].g += local[t].outputWeights[i];

    g->outputBiasGradient.g += local[t].outputBias;
  }
}