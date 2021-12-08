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

  printf("Calculating Validation Error...\r");
  float error = TotalError(validation, nn);
  printf("Starting Error: [%1.8f]\n", error);

  for (int epoch = 1; epoch <= 20; epoch++) {
    long epochStart = GetTimeMS();

    printf("Shuffling...\r");
    ShuffleData(data);

    int batches = data->n / BATCH_SIZE;
    for (int b = 0; b < batches; b++) {
      Train(b, data, nn, gradients, local);
      ApplyGradients(nn, gradients);

      printf("Batch: [#%d/%d]\r", b + 1, batches);
    }

    char buffer[64];
    sprintf(buffer, "../nets/berserk-ks.e%d.2x%d.x%d.x%d.nn", epoch, N_HIDDEN, N_HIDDEN_2, N_HIDDEN_3);
    SaveNN(nn, buffer);

    printf("Calculating Validation Error...\r");
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
  for (int t = 0; t < THREADS; t++)
    memset(&local[t], 0, sizeof(BatchGradients));

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

    float h3losses[N_HIDDEN_3];
    for (int i = 0; i < N_HIDDEN_3; i++)
      h3losses[i] = outputLoss * nn->outputWeights[i] * ReLUPrime(activations->acc3[i]);

    float h2losses[N_HIDDEN_2] = {0};
    for (int i = 0; i < N_HIDDEN_2; i++)
      for (int j = 0; j < N_HIDDEN_3; j++)
        h2losses[i] += h3losses[j] * nn->h3Weights[j * N_HIDDEN_2 + i] * ReLUPrime(activations->acc2[i]);

    float hiddenLosses[2][N_HIDDEN] = {0};
    for (int i = 0; i < N_HIDDEN; i++) {
      for (int j = 0; j < N_HIDDEN_2; j++) {
        hiddenLosses[board.stm][i] +=
            h2losses[j] * nn->h2Weights[j * 2 * N_HIDDEN + i] * ReLUPrime(activations->acc1[board.stm][i]);
        hiddenLosses[board.stm ^ 1][i] += h2losses[j] * nn->h2Weights[j * 2 * N_HIDDEN + i + N_HIDDEN] *
                                          ReLUPrime(activations->acc1[board.stm ^ 1][i]);
      }
    }
    // ------------------------------------------------------------------------------------------

    // OUTPUT LAYER GRADIENTS -------------------------------------------------------------------
    local[t].outputBias += outputLoss;

    for (int i = 0; i < N_HIDDEN_3; i++) {
      local[t].outputWeights[i] += activations->acc3[i] * outputLoss;
    }
    // ------------------------------------------------------------------------------------------

    // THIRD LAYER GRADIENTS ---------------------------------------------------------------------
    for (int i = 0; i < N_HIDDEN_3; i++) {
      local[t].h3Biases[i] += h3losses[i];

      for (int j = 0; j < N_HIDDEN_2; j++)
        local[t].h3Weights[i * N_HIDDEN_2 + j] += activations->acc2[j] * h3losses[i];
    }
    // -------------------------------------------------------------------------------------------

    // SECOND LAYER GRADIENTS -------------------------------------------------------------------
    for (int i = 0; i < N_HIDDEN_2; i++) {
      local[t].h2Biases[i] += h2losses[i];

      for (int j = 0; j < N_HIDDEN; j++) {
        local[t].h2Weights[i * 2 * N_HIDDEN + j] += activations->acc1[board.stm][j] * h2losses[i];
        local[t].h2Weights[i * 2 * N_HIDDEN + j + N_HIDDEN] += activations->acc1[board.stm ^ 1][j] * h2losses[i];
      }
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
    // ------------------------------------------------------------------------------------------
  }

#pragma omp parallel for schedule(auto) num_threads(4)
  for (int i = 0; i < N_INPUT * N_HIDDEN; i++)
    for (int t = 0; t < THREADS; t++)
      g->inputWeights[i].g += local[t].inputWeights[i];

#pragma omp parallel for schedule(auto) num_threads(2)
  for (int i = 0; i < N_HIDDEN; i++)
    for (int t = 0; t < THREADS; t++)
      g->inputBiases[i].g += local[t].inputBiases[i];

#pragma omp parallel for schedule(auto) num_threads(2)
  for (int i = 0; i < 2 * N_HIDDEN * N_HIDDEN_2; i++)
    for (int t = 0; t < THREADS; t++)
      g->h2Weights[i].g += local[t].h2Weights[i];

#pragma omp parallel for schedule(auto) num_threads(2)
  for (int i = 0; i < N_HIDDEN_2; i++)
    for (int t = 0; t < THREADS; t++)
      g->h2Biases[i].g += local[t].h2Biases[i];

#pragma omp parallel for schedule(auto) num_threads(2)
  for (int i = 0; i < N_HIDDEN_2 * N_HIDDEN_3; i++)
    for (int t = 0; t < THREADS; t++)
      g->h3Weights[i].g += local[t].h3Weights[i];

#pragma omp parallel for schedule(auto) num_threads(2)
  for (int i = 0; i < N_HIDDEN_3; i++)
    for (int t = 0; t < THREADS; t++)
      g->h3Biases[i].g += local[t].h3Biases[i];

#pragma omp parallel for schedule(auto) num_threads(2)
  for (int i = 0; i < N_HIDDEN_3; i++)
    for (int t = 0; t < THREADS; t++)
      g->outputWeights[i].g += local[t].outputWeights[i];

  for (int t = 0; t < THREADS; t++)
    g->outputBias.g += local[t].outputBias;
}