#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "bits.h"
#include "board.h"
#include "data.h"
#include "nn.h"
#include "trainer.h"
#include "util.h"

int main(int argc, char** argv) {
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
  data->entries = malloc(sizeof(DataEntry) * 500000000);
  LoadEntries(entriesPath, data);

  NNGradients* gradients = malloc(sizeof(NNGradients));
  ClearGradients(gradients);

  BatchGradients* localGradients = malloc(sizeof(BatchGradients) * THREADS);

  float error = TotalError(data, nn);
  printf("Starting Error: [%1.8f]\n", error);

  for (int epoch = 1; epoch <= 250; epoch++) {
    long epochStart = GetTimeMS();

    printf("Shuffling...\r");
    ShuffleData(data);

    int batches = data->n / BATCH_SIZE;
    for (int b = 0; b < batches; b++) {
      Train(b, data, nn, gradients, localGradients);
      UpdateNetwork(nn, gradients);

      printf("Batch: [#%5d]\r", b + 1);
    }

    char buffer[64];
    sprintf(buffer, "../nets/berserk.hp.e%d.%d.2x%d.nn", epoch, N_FEATURES, N_HIDDEN);
    SaveNN(nn, buffer);

    printf("Calculating Error...\r");
    float newError = TotalError(data, nn);

    long now = GetTimeMS();
    printf("Epoch: [#%5d], Error: [%1.8f], Delta: [%+1.8f], LR: [%.5f], Speed: [%9.0f pos/s], Time: [%lds]\n", epoch,
           newError, error - newError, ALPHA, 1000.0 * data->n / (now - epochStart), (now - epochStart) / 1000);

    error = newError;
  }
}

float Error(float result, DataEntry* entry) {
  return powf(result - entry->wdl, 2) * 0.5f + powf(result - entry->eval, 2) * 0.5f;
}

float ErrorGradient(float result, DataEntry* entry) { return (result - entry->wdl) + (result - entry->eval); }

float TotalError(DataSet* data, NN* nn) {
  pthread_t threads[ERR_THREADS];
  CalculateErrorJob jobs[ERR_THREADS];

  int chunkSize = data->n / ERR_THREADS;

  for (int t = 0; t < ERR_THREADS; t++) {
    jobs[t].start = t * chunkSize;
    jobs[t].n = chunkSize;
    jobs[t].data = data;
    jobs[t].nn = nn;

    pthread_create(&threads[t], NULL, &CalculateError, &jobs[t]);
  }

  float e = 0.0f;

  for (int t = 0; t < ERR_THREADS; t++) {
    pthread_join(threads[t], NULL);
    e += jobs[t].error;
  }

  return e / (chunkSize * ERR_THREADS);
}

void* CalculateError(void* arg) {
  CalculateErrorJob* job = (CalculateErrorJob*)arg;
  NN* nn = job->nn;

  job->error = 0.0f;

  for (int n = job->start; n < job->start + job->n; n++) {
    DataEntry entry = job->data->entries[n];
    NNActivations results[1];
    NNPredict(nn, entry.board, results, entry.stm);
    job->error += Error(Sigmoid(results->result), &entry);
  }

  return NULL;
}

void Train(int batch, DataSet* data, NN* nn, NNGradients* g, BatchGradients* threadLocal) {
  pthread_t threads[THREADS];
  UpdateGradientsJob jobs[THREADS];

  int chunkSize = BATCH_SIZE / THREADS;

  for (int t = 0; t < THREADS; t++) {
    jobs[t].start = batch * BATCH_SIZE + t * chunkSize;
    jobs[t].n = chunkSize;
    jobs[t].data = data;
    jobs[t].nn = nn;
    jobs[t].gradients = &threadLocal[t];

    pthread_create(&threads[t], NULL, &CalculateGradients, &jobs[t]);
  }

  for (int t = 0; t < THREADS; t++)
    pthread_join(threads[t], NULL);

  for (int t = 0; t < THREADS; t++) {
    for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
      g->featureWeightGradients[i].g += threadLocal[t].featureWeights[i];

    for (int i = 0; i < N_HIDDEN; i++)
      g->hiddenBiasGradients[i].g += threadLocal[t].hiddenBias[i];

    for (int i = 0; i < N_HIDDEN * 2; i++)
      g->hiddenWeightGradients[i].g += threadLocal[t].hiddenWeights[i];

    g->outputBiasGradient.g += threadLocal[t].outputBias;
  }
}

void* CalculateGradients(void* arg) {
  UpdateGradientsJob* job = (UpdateGradientsJob*)arg;
  NN* nn = job->nn;
  BatchGradients* gradients = job->gradients;

  ClearBatchGradients(gradients);

  for (int n = job->start; n < job->start + job->n; n++) {
    DataEntry entry = job->data->entries[n];

    NNActivations results[1];
    NNPredict(nn, entry.board, results, entry.stm);

    float out = Sigmoid(results->result);
    float loss = SigmoidPrime(out) * ErrorGradient(out, &entry);

    gradients->outputBias += loss;
    for (int i = 0; i < N_HIDDEN; i++) {
      gradients->hiddenWeights[i] += results->accumulators[entry.stm][i] * loss;
      gradients->hiddenWeights[i + N_HIDDEN] += results->accumulators[entry.stm ^ 1][i] * loss;
    }

    for (int i = 0; i < N_HIDDEN; i++) {
      float stmLayerLoss = loss * nn->hiddenWeights[i] * (results->accumulators[entry.stm][i] > 0.0f);
      float xstmLayerLoss = loss * nn->hiddenWeights[i + N_HIDDEN] * (results->accumulators[entry.stm ^ 1][i] > 0.0f);

      gradients->hiddenBias[i] += stmLayerLoss + xstmLayerLoss;

      for (int j = 0; j < 32; j++) {
        if (entry.board[j].pc < 0)
          break;

        if (stmLayerLoss) {
          int stmf = feature(entry.board[j], entry.stm);
          gradients->featureWeights[stmf * N_HIDDEN + i] += stmLayerLoss;
        }

        if (xstmLayerLoss) {
          int xstmf = feature(entry.board[j], entry.stm ^ 1);
          gradients->featureWeights[xstmf * N_HIDDEN + i] += xstmLayerLoss;
        }
      }
    }
  }

  return NULL;
}

void UpdateNetwork(NN* nn, NNGradients* g) {
  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->featureWeights[i], &g->featureWeightGradients[i]);

  for (int i = 0; i < N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->hiddenBiases[i], &g->hiddenBiasGradients[i]);

  for (int i = 0; i < N_HIDDEN * 2; i++)
    UpdateAndApplyGradient(&nn->hiddenWeights[i], &g->hiddenWeightGradients[i]);

  UpdateAndApplyGradient(&nn->outputBias, &g->outputBiasGradient);
}

void UpdateAndApplyGradient(float* v, Gradient* grad) {
  if (!grad->g)
    return;

  grad->M = BETA1 * grad->M + (1.0 - BETA1) * grad->g;
  grad->V = BETA2 * grad->V + (1.0 - BETA2) * grad->g * grad->g;

  float delta = ALPHA * grad->M / (sqrtf(grad->V) + EPSILON);

  *v -= delta;

  grad->g = 0;
}

void ClearGradients(NNGradients* gradients) {
  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++) {
    gradients->featureWeightGradients[i].g = 0;
    gradients->featureWeightGradients[i].V = 0;
    gradients->featureWeightGradients[i].M = 0;
  }

  for (int i = 0; i < N_HIDDEN; i++) {
    gradients->hiddenBiasGradients[i].g = 0;
    gradients->hiddenBiasGradients[i].V = 0;
    gradients->hiddenBiasGradients[i].M = 0;
  }

  for (int i = 0; i < N_HIDDEN * 2; i++) {
    gradients->hiddenWeightGradients[i].g = 0;
    gradients->hiddenWeightGradients[i].V = 0;
    gradients->hiddenWeightGradients[i].M = 0;
  }

  gradients->outputBiasGradient.g = 0;
  gradients->outputBiasGradient.V = 0;
  gradients->outputBiasGradient.M = 0;
}

void ClearBatchGradients(BatchGradients* gradients) {
  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    gradients->featureWeights[i] = 0;

  for (int i = 0; i < N_HIDDEN; i++)
    gradients->hiddenBias[i] = 0;

  for (int i = 0; i < N_HIDDEN * 2; i++)
    gradients->hiddenWeights[i] = 0;

  gradients->outputBias = 0;
}
