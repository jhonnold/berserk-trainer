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
#include "scalars.h"
#include "trainer.h"
#include "util.h"

int main(int argc, char** argv) {
  SeedRandom();

  int c, s = 0;

  char nnPath[128] = {0};
  char entriesPath[128] = {0};

  while ((c = getopt(argc, argv, "sd:n:")) != -1) {
    switch (c) {
    case 'd':
      strcpy(entriesPath, optarg);
      break;
    case 'n':
      strcpy(nnPath, optarg);
      break;
    case 's':
      s = 1;
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

  if (s) {
    PrintMinMax(data, data->n, nn);
    exit(0);
  }

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
    sprintf(buffer, "../nets/berserk-ks.e%d.2x%d.nn", epoch, N_HIDDEN);
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

    NNActivations results[1];
    NNPredict(nn, &entry.board, results);

    e += Error(Sigmoid(results->result), &entry);
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

    NNActivations results[1];
    NNPredict(nn, &board, results);

    float out = Sigmoid(results->result);
    float loss = SigmoidPrime(out) * ErrorGradient(out, &entry);

    local[t].outputBias += loss;
    for (int i = 0; i < N_HIDDEN; i++) {
      local[t].hiddenWeights[i] += results->accumulators[board.stm][i] * loss;
      local[t].hiddenWeights[i + N_HIDDEN] += results->accumulators[board.stm ^ 1][i] * loss;
    }

    for (int i = 0; i < N_HIDDEN; i++) {
      float stmLayerLoss = loss * nn->hiddenWeights[i] * (results->accumulators[board.stm][i] > 0.0f);
      float xStmLayerLoss = loss * nn->hiddenWeights[i + N_HIDDEN] * (results->accumulators[board.stm ^ 1][i] > 0.0f);

      local[t].hiddenBias[i] += stmLayerLoss + xStmLayerLoss;

      uint64_t bb = board.occupancies;
      int p = 0;
      while (bb) {
        Square sq = lsb(bb);
        Piece pc = getPiece(board.pieces, p++);

        if (stmLayerLoss)
          local[t].featureWeights[idx(pc, sq, board.kings[board.stm], board.stm) * N_HIDDEN + i] += stmLayerLoss;

        if (xStmLayerLoss)
          local[t].featureWeights[idx(pc, sq, board.kings[board.stm ^ 1], board.stm ^ 1) * N_HIDDEN + i] +=
              xStmLayerLoss;

        popLsb(bb);
      }
    }

    uint64_t bb = board.occupancies;
    int p = 0;
    while (bb) {
      Square sq = lsb(bb);
      Piece pc = getPiece(board.pieces, p++);

      local[t].skipWeights[idx(pc, sq, board.kings[board.stm], board.stm)] += loss;

      popLsb(bb);
    }
  }

  for (int t = 0; t < THREADS; t++) {
#pragma omp parallel for schedule(auto) num_threads(2)
    for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
      g->featureWeightGradients[i].g += local[t].featureWeights[i];

#pragma omp parallel for schedule(auto) num_threads(2)
    for (int i = 0; i < N_HIDDEN; i++)
      g->hiddenBiasGradients[i].g += local[t].hiddenBias[i];

#pragma omp parallel for schedule(auto) num_threads(2)
    for (int i = 0; i < N_HIDDEN * 2; i++)
      g->hiddenWeightGradients[i].g += local[t].hiddenWeights[i];

    g->outputBiasGradient.g += local[t].outputBias;

    for (int i = 0; i < N_FEATURES; i++)
      g->skipWeightGradients[i].g += local[t].skipWeights[i];
  }
}