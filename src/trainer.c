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

#define THREADS 28
#define BATCH_SIZE 16384

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

  printf("Loading entries from %s\n", entriesPath);

  DataSet* data = malloc(sizeof(DataSet));
  data->n = 0;
  data->entries = calloc(100000000, sizeof(DataEntry));
  LoadEntries(entriesPath, data);

  NN* nn;
  if (!nnPath[0]) {
    printf("No net specified, generating a random net.\n");
    nn = LoadRandomNN();
  } else {
    printf("Loading net from %s\n", nnPath);
    nn = LoadNN(nnPath);
  }

  NNGradients* gradients = malloc(sizeof(NNGradients));
  ClearGradients(gradients);

  float error = 0.0f;

  for (int epoch = 1; epoch <= 10000; epoch++) {
    long now;
    long epochStart = GetTimeMS();

    for (int b = 0; b < data->n / BATCH_SIZE; b++) {
      long start = GetTimeMS();
      float e = Train(b, data, nn, gradients);
      now = GetTimeMS();
      printf("Batch: [#%4d], Error: [%1.8f], Speed: [%9.0f pos/s]\r", b + 1, e, 1000.0 * BATCH_SIZE / (now - start));

      UpdateNetwork(nn, gradients);
    }

    now = GetTimeMS();
    float newError = TotalError(data, nn);
    printf("Epoch: [#%4d], Error: [%1.8f], Delta: [%+1.8f], Speed: [%9.0f pos/s]\n", epoch, newError, error - newError,
           1000.0 * data->n * epoch / (now - epochStart));

    error = newError;

    char buffer[64];
    sprintf(buffer, "../nets/768x128x1.%d.nn", epoch);
    SaveNN(nn, buffer);
  }
}

float Error(float result, DataEntry* entry) {
  return powf(result - entry->wdl, 2) * 0.5f + powf(result - entry->eval, 2) * 0.5f;
}

float ErrorGradient(float result, DataEntry* entry) { return (result - entry->wdl) + (result - entry->eval); }

float TotalError(DataSet* data, NN* nn) {
  float e = 0.0f;

#pragma omp parallel for schedule(static) num_threads(THREADS) reduction(+ : e)
  for (int i = 0; i < data->n; i++) {
    NNActivations results = {0};
    DataEntry entry = data->entries[i];

    NNPredict(nn, entry.board, &results);

    e += Error(Sigmoid(results.outputActivations[0]), &entry);
  }

  return e / data->n;
}

float Train(int batch, DataSet* data, NN* nn, NNGradients* g) {
  pthread_t threads[THREADS];
  NNGradients gradients[THREADS];
  UpdateGradientsJob jobs[THREADS];

  int chunkSize = BATCH_SIZE / THREADS;

  for (int t = 0; t < THREADS; t++) {
    jobs[t].start = batch * BATCH_SIZE + t * chunkSize;
    jobs[t].n = chunkSize;
    jobs[t].error = 0.0f;
    jobs[t].data = data;
    jobs[t].nn = nn;
    jobs[t].gradients = &gradients[t];

    pthread_create(&threads[t], NULL, &CalculateGradients, &jobs[t]);
  }

  for (int t = 0; t < THREADS; t++)
    pthread_join(threads[t], NULL);

  float e = 0.0f;
  for (int t = 0; t < THREADS; t++) {
    e += jobs[t].error;

    for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
      g->featureWeightGradients[i].g += gradients[t].featureWeightGradients[i].g;

    for (int i = 0; i < N_HIDDEN * N_OUTPUT; i++)
      g->hiddenWeightGradients[i].g += gradients[t].hiddenWeightGradients[i].g;

    for (int i = 0; i < N_HIDDEN; i++)
      g->hiddenBiasGradients[i].g += gradients[t].hiddenBiasGradients[i].g;

    for (int i = 0; i < N_OUTPUT; i++)
      g->outputBiasGradients[i].g += gradients[t].outputBiasGradients[i].g;
  }

  return e / THREADS;
}

void ClearGradients(NNGradients* gradients) {
  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++) {
    gradients->featureWeightGradients[i].g = 0;
    gradients->featureWeightGradients[i].V = 0;
    gradients->featureWeightGradients[i].M = 0;
  }

  for (int i = 0; i < N_HIDDEN * N_OUTPUT; i++) {
    gradients->hiddenWeightGradients[i].g = 0;
    gradients->hiddenWeightGradients[i].V = 0;
    gradients->hiddenWeightGradients[i].M = 0;
  }

  for (int i = 0; i < N_HIDDEN; i++) {
    gradients->hiddenBiasGradients[i].g = 0;
    gradients->hiddenBiasGradients[i].V = 0;
    gradients->hiddenBiasGradients[i].M = 0;
  }

  for (int i = 0; i < N_OUTPUT; i++) {
    gradients->outputBiasGradients[i].g = 0;
    gradients->outputBiasGradients[i].V = 0;
    gradients->outputBiasGradients[i].M = 0;
  }
}

void* CalculateGradients(void* arg) {
  UpdateGradientsJob* job = (UpdateGradientsJob*)arg;

  for (int n = job->start; n < job->start + job->n; n++) {
    DataEntry entry = job->data->entries[n];
    NN* nn = job->nn;
    NNGradients* gradients = job->gradients;

    NNActivations results[1];

    NNPredict(nn, entry.board, results);

    float out = Sigmoid(results->outputActivations[0]);
    float loss = SigmoidPrime(out) * ErrorGradient(out, &entry);
    job->error = Error(out, &entry);

    gradients->outputBiasGradients[0].g += loss;
    for (int i = 0; i < N_HIDDEN; i++)
      gradients->hiddenWeightGradients[i].g += results->hiddenActivations[i] * loss;

    for (int i = 0; i < N_HIDDEN; i++) {
      float layerLoss = loss * nn->hiddenWeights[i] * (results->hiddenActivations[i] > 0.0f);

      gradients->hiddenBiasGradients[i].g += layerLoss;

      for (int a = 0; a < 32; a++)
        if (entry.board[a])
          for (int j = 0; j < N_HIDDEN; j++)
            gradients->featureWeightGradients[entry.board[a] * N_HIDDEN + j].g += layerLoss;
        else
          break;
    }
  }

  return NULL;
}

void UpdateNetwork(NN* nn, NNGradients* g) {
  for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->featureWeights[i], &g->featureWeightGradients[i]);

  for (int i = 0; i < N_HIDDEN * N_OUTPUT; i++)
    UpdateAndApplyGradient(&nn->hiddenWeights[i], &g->hiddenWeightGradients[i]);

  for (int i = 0; i < N_HIDDEN; i++)
    UpdateAndApplyGradient(&nn->hiddenBiases[i], &g->hiddenBiasGradients[i]);

  for (int i = 0; i < N_OUTPUT; i++)
    UpdateAndApplyGradient(&nn->outputBiases[i], &g->outputBiasGradients[i]);
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
