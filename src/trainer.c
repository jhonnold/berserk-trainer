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

    NNAccumulators results[1];
    NNPredict(nn, &entry.board, results);

    e += Error(Sigmoid(results->output), &entry);
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

    NNAccumulators activations[1];
    NNPredict(nn, &board, activations);

    float out = Sigmoid(activations->output);

    // LOSS CALCULATIONS ------------------------------------------------------------------------
    float outputLoss = SigmoidPrime(out) * ErrorGradient(out, &entry);

    float hidden3Losses[N_HIDDEN_3];
    for (int i = 0; i < N_HIDDEN_3; i++)
      hidden3Losses[i] = outputLoss * nn->outputWeights[i] * ReLUPrime(activations->acc3[i]);

    float hidden2Losses[N_HIDDEN_2] = {0};
    for (int i = 0; i < N_HIDDEN_2; i++)
      for (int j = 0; j < N_HIDDEN_3; j++)
        hidden2Losses[i] += hidden3Losses[j] * nn->h3Weights[j * N_HIDDEN_2 + i] * ReLUPrime(activations->acc2[i]);

    float hiddenLosses[2][N_HIDDEN];
    for (int i = 0; i < N_HIDDEN; i++) {
      for (int j = 0; j < N_HIDDEN_2; j++) {
        hiddenLosses[board.stm][i] =
            hidden2Losses[j] * nn->h2Weights[j * 2 * N_HIDDEN + i] * ReLUPrime(activations->acc1[board.stm][i]);
        hiddenLosses[board.stm ^ 1][i] = hidden2Losses[j] * nn->h2Weights[j * 2 * N_HIDDEN + i + N_HIDDEN] *
                                         ReLUPrime(activations->acc1[board.stm ^ 1][i]);
      }
    }
    // ------------------------------------------------------------------------------------------

    // OUTPUT LAYER GRADIENTS -------------------------------------------------------------------
    local[t].outputBias += outputLoss;
    for (int i = 0; i < N_HIDDEN_3; i++)
      local[t].outputWeights[i] += activations->acc3[i] * outputLoss;
    // ------------------------------------------------------------------------------------------

    // THIRD LAYER GRADIENTS -------------------------------------------------------------------
    for (int i = 0; i < N_HIDDEN_3; i++) {
      local[t].h3Biases[i] += hidden3Losses[i];

      for (int j = 0; j < N_HIDDEN_2; j++)
        local[t].h3Weights[i * N_HIDDEN_2 + j] += activations->acc2[j] * hidden3Losses[i];
    }
    // ------------------------------------------------------------------------------------------

    // SECOND LAYER GRADIENTS -------------------------------------------------------------------
    for (int i = 0; i < N_HIDDEN_2; i++) {
      local[t].h2Biases[i] += hidden2Losses[i];

      for (int j = 0; j < N_HIDDEN; j++) {
        local[t].h2Weights[i * 2 * N_HIDDEN + j] += activations->acc1[board.stm][j] * hidden2Losses[i];
        local[t].h2Weights[i * 2 * N_HIDDEN + j + N_HIDDEN] += activations->acc1[board.stm ^ 1][j] * hidden2Losses[i];
      }
    }
    // ------------------------------------------------------------------------------------------

    // INPUT LAYER GRADIENTS --------------------------------------------------------------------
    for (int i = 0; i < N_HIDDEN; i++)
      local[t].inputBiases[i] += hiddenLosses[board.stm][i] + hiddenLosses[board.stm ^ 1][i];

    uint64_t bb = board.occupancies;
    int p = 0;
    while (bb) {
      Square sq = lsb(bb);
      Piece pc = getPiece(board.pieces, p++);

      for (int i = 0; i < N_HIDDEN; i++) {
        local[t].inputWeights[idx(pc, sq, board.kings[board.stm], board.stm) * N_HIDDEN + i] +=
            hiddenLosses[board.stm][i];
        local[t].inputWeights[idx(pc, sq, board.kings[board.stm ^ 1], board.stm ^ 1) * N_HIDDEN + i] +=
            hiddenLosses[board.stm ^ 1][i];
      }

      popLsb(bb);
    }
    // ------------------------------------------------------------------------------------------

    // SKIP CONNECTION GRADIENTS ----------------------------------------------------------------
    bb = board.occupancies;
    p = 0;
    while (bb) {
      Square sq = lsb(bb);
      Piece pc = getPiece(board.pieces, p++);

      local[t].skipWeights[idx(pc, sq, board.kings[board.stm], board.stm)] += outputLoss;

      popLsb(bb);
    }
    // ------------------------------------------------------------------------------------------
  }

  for (int t = 0; t < THREADS; t++) {
#pragma omp parallel for schedule(auto) num_threads(2)
    for (int i = 0; i < N_INPUT * N_HIDDEN; i++)
      g->inputWeights[i].g += local[t].inputWeights[i];

    for (int i = 0; i < N_HIDDEN; i++)
      g->inputBiases[i].g += local[t].inputBiases[i];

    for (int i = 0; i < 2 * N_HIDDEN * N_HIDDEN_2; i++)
      g->h2Weights[i].g += local[t].h2Weights[i];

    for (int i = 0; i < N_HIDDEN_2; i++)
      g->h2Biases[i].g += local[t].h2Biases[i];

    for (int i = 0; i < N_HIDDEN_2 * N_HIDDEN_3; i++)
      g->h3Weights[i].g += local[t].h3Weights[i];

    for (int i = 0; i < N_HIDDEN_3; i++)
      g->h3Biases[i].g += local[t].h3Biases[i];

    for (int i = 0; i < N_HIDDEN_3; i++)
      g->outputWeights[i].g += local[t].outputWeights[i];

    g->outputBias.g += local[t].outputBias;

    for (int i = 0; i < N_INPUT; i++)
      g->skipWeights[i].g += local[t].skipWeights[i];
  }
}