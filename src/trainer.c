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

  for (int epoch = 1; epoch <= 250; epoch++) {
    long epochStart = GetTimeMS();

    printf("Shuffling...\n");
    ShuffleData(data);

    int batches = data->n / BATCH_SIZE;
    for (int b = 0; b < batches; b++) {
      Train(b, data, nn, gradients, local);
      ApplyGradients(nn, gradients);

      if ((b + 1) % 1000 == 0) printf("Batch: [#%d/%d]\n", b + 1, batches);
    }

    char buffer[64];
    sprintf(buffer, "../nets/berserk-ks+p.e%d.2x%d.2x%d.nn", epoch, N_HIDDEN, N_P_HIDDEN);
    SaveNN(nn, buffer);

    printf("Calculating Validation Error...\n");
    float newError = TotalError(validation, nn);

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
    Board* board = &data->entries[i];

    NNAccumulators results[1];
    Features f[1];

    ToFeatures(board, f);
    NNPredict(nn, f, board->stm, results);

    e += Error(Sigmoid(results->output), board);
  }

  return e / data->n;
}

void Train(int batch, DataSet* data, NN* nn, NNGradients* g, BatchGradients* local) {
#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int t = 0; t < THREADS; t++) memset(&local[t], 0, sizeof(BatchGradients));

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int n = 0; n < BATCH_SIZE; n++) {
    const int t = omp_get_thread_num();

    Board board = data->entries[n + batch * BATCH_SIZE];

    NNAccumulators activations[1];
    Features f[1];

    ToFeatures(&board, f);
    NNPredict(nn, f, board.stm, activations);

    float out = Sigmoid(activations->output);

    // LOSS CALCULATIONS ------------------------------------------------------------------------
    float outputLoss = SigmoidPrime(out) * ErrorGradient(out, &board);

    float hiddenLosses[2][N_HIDDEN];
    for (int i = 0; i < N_HIDDEN; i++) {
      hiddenLosses[board.stm][i] = outputLoss * nn->outputWeights[i] * ReLUPrime(activations->acc1[board.stm][i]);
      hiddenLosses[board.stm ^ 1][i] =
          outputLoss * nn->outputWeights[i + N_HIDDEN] * ReLUPrime(activations->acc1[board.stm ^ 1][i]);
    }

    float pawnHiddenLosses[2][N_P_HIDDEN];
    for (int i = 0; i < N_P_HIDDEN; i++) {
      pawnHiddenLosses[board.stm][i] =
          outputLoss * nn->pawnOutputWeights[i] * ReLUPrime(activations->pAcc1[board.stm][i]);
      pawnHiddenLosses[board.stm ^ 1][i] =
          outputLoss * nn->pawnOutputWeights[i + N_P_HIDDEN] * ReLUPrime(activations->pAcc1[board.stm ^ 1][i]);
    }
    // ------------------------------------------------------------------------------------------

    // OUTPUT LAYER GRADIENTS -------------------------------------------------------------------
    local[t].outputBias += outputLoss;
    for (int i = 0; i < N_HIDDEN; i++) {
      local[t].outputWeights[i] += activations->acc1[board.stm][i] * outputLoss;
      local[t].outputWeights[i + N_HIDDEN] += activations->acc1[board.stm ^ 1][i] * outputLoss;
    }

    local[t].pawnOutputBias += outputLoss;
    for (int i = 0; i < N_P_HIDDEN; i++) {
      local[t].pawnOutputWeights[i] += activations->pAcc1[board.stm][i] * outputLoss;
      local[t].pawnOutputWeights[i + N_P_HIDDEN] += activations->pAcc1[board.stm ^ 1][i] * outputLoss;
    }
    // ------------------------------------------------------------------------------------------

    // INPUT LAYER GRADIENTS --------------------------------------------------------------------
    for (int i = 0; i < N_HIDDEN; i++) {
      float stmLasso = LAMBDA * (activations->acc1[board.stm][i] > 0);
      float xstmLasso = LAMBDA * (activations->acc1[board.stm ^ 1][i] > 0);

      local[t].inputBiases[i] += hiddenLosses[board.stm][i] + hiddenLosses[board.stm ^ 1][i] + stmLasso + xstmLasso;
    }

    for (int i = 0; i < f->n; i++) {
      for (int j = 0; j < N_HIDDEN; j++) {
        float stmLasso = LAMBDA * (activations->acc1[board.stm][j] > 0);
        float xstmLasso = LAMBDA * (activations->acc1[board.stm ^ 1][j] > 0);

        local[t].inputWeights[f->features[board.stm][i] * N_HIDDEN + j] += hiddenLosses[board.stm][j] + stmLasso;
        local[t].inputWeights[f->features[board.stm ^ 1][i] * N_HIDDEN + j] +=
            hiddenLosses[board.stm ^ 1][j] + xstmLasso;
      }
    }

    for (int i = 0; i < N_P_HIDDEN; i++) {
      local[t].pawnInputBiases[i] += pawnHiddenLosses[board.stm][i] + pawnHiddenLosses[board.stm ^ 1][i];
    }

    for (int i = 0; i < f->p; i++) {
      for (int j = 0; j < N_P_HIDDEN; j++) {
        local[t].pawnInputWeights[f->pawnFeatures[board.stm][i] * N_P_HIDDEN + j] += pawnHiddenLosses[board.stm][j];
        local[t].pawnInputWeights[f->pawnFeatures[board.stm ^ 1][i] * N_P_HIDDEN + j] +=
            pawnHiddenLosses[board.stm ^ 1][j];
      }
    }
    // ------------------------------------------------------------------------------------------
  }

#pragma omp parallel for schedule(auto) num_threads(4)
  for (int i = 0; i < N_INPUT * N_HIDDEN; i++)
    for (int t = 0; t < THREADS; t++) g->inputWeights[i].g += local[t].inputWeights[i];

#pragma omp parallel for schedule(auto) num_threads(2)
  for (int i = 0; i < N_HIDDEN; i++)
    for (int t = 0; t < THREADS; t++) g->inputBiases[i].g += local[t].inputBiases[i];

#pragma omp parallel for schedule(auto) num_threads(2)
  for (int i = 0; i < N_HIDDEN * 2; i++)
    for (int t = 0; t < THREADS; t++) g->outputWeights[i].g += local[t].outputWeights[i];

  for (int t = 0; t < THREADS; t++) g->outputBias.g += local[t].outputBias;

#pragma omp parallel for schedule(auto) num_threads(4)
  for (int i = 0; i < N_P_INPUT * N_P_HIDDEN; i++)
    for (int t = 0; t < THREADS; t++) g->pawnInputWeights[i].g += local[t].pawnInputWeights[i];

#pragma omp parallel for schedule(auto) num_threads(2)
  for (int i = 0; i < N_P_HIDDEN; i++)
    for (int t = 0; t < THREADS; t++) g->pawnInputBiases[i].g += local[t].pawnInputBiases[i];

#pragma omp parallel for schedule(auto) num_threads(2)
  for (int i = 0; i < N_P_HIDDEN * 2; i++)
    for (int t = 0; t < THREADS; t++) g->pawnOutputWeights[i].g += local[t].pawnOutputWeights[i];

  for (int t = 0; t < THREADS; t++) g->pawnOutputBias.g += local[t].pawnOutputBias;
}