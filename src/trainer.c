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

  float error = TotalError(data, nn);
  printf("Starting Error: [%1.8f]\n", error);

  for (int epoch = 1; epoch <= 10000; epoch++) {
    long epochStart = GetTimeMS();

    printf("Shuffling...\r");
    ShuffleData(data);

    int batches = data->n / BATCH_SIZE;
    for (int b = 0; b < batches; b++) {
      Train(b, data, nn, gradients);
      UpdateNetwork(nn, gradients);

      printf("Batch: [#%5d]\r", b + 1);
    }

    float newError = TotalError(data, nn);

    long now = GetTimeMS();
    printf("Epoch: [#%5d], Error: [%1.8f], Delta: [%+1.8f], Speed: [%9.0f pos/s]\n", epoch, newError, error - newError,
           1000.0 * data->n / (now - epochStart));

    error = newError;

    char buffer[64];
    sprintf(buffer, "../nets/berserk.e%d.%d.%d.nn", epoch, N_FEATURES, N_HIDDEN);
    SaveNN(nn, buffer);
  }
}

float Error(float result, DataEntry* entry) {
  return powf(result - entry->wdl, 2) * 0.5f + powf(result - entry->eval, 2) * 0.5f;
}

float ErrorGradient(float result, DataEntry* entry) { return (result - entry->wdl) + (result - entry->eval); }

float TotalError(DataSet* data, NN* nn) {
  pthread_t threads[THREADS];
  CalculateErrorJob jobs[THREADS];

  int chunkSize = data->n / THREADS;

  for (int t = 0; t < THREADS; t++) {
    jobs[t].start = t * chunkSize;
    jobs[t].n = chunkSize;
    jobs[t].data = data;
    jobs[t].nn = nn;

    pthread_create(&threads[t], NULL, &CalculateError, &jobs[t]);
  }

  float e = 0.0f;

  for (int t = 0; t < THREADS; t++) {
    pthread_join(threads[t], NULL);
    e += jobs[t].error;
  }

  return e / (chunkSize * THREADS);
}

void* CalculateError(void* arg) {
  CalculateErrorJob* job = (CalculateErrorJob*)arg;
  NN* nn = job->nn;

  job->error = 0.0f;

  for (int n = job->start; n < job->start + job->n; n++) {
    DataEntry entry = job->data->entries[n];
    NNActivations results[1];
    NNPredict(nn, entry.board, results);

    job->error += Error(Sigmoid(results->outputActivations[0]), &entry);
  }

  return NULL;
}

void Train(int batch, DataSet* data, NN* nn, NNGradients* g) {
  pthread_t threads[THREADS];
  NNGradients gradients[THREADS];
  UpdateGradientsJob jobs[THREADS];

  int chunkSize = BATCH_SIZE / THREADS;

  for (int t = 0; t < THREADS; t++) {
    ClearGradients(&gradients[t]);

    jobs[t].start = batch * BATCH_SIZE + t * chunkSize;
    jobs[t].n = chunkSize;
    jobs[t].data = data;
    jobs[t].nn = nn;
    jobs[t].gradients = &gradients[t];

    pthread_create(&threads[t], NULL, &CalculateGradients, &jobs[t]);
  }

  for (int t = 0; t < THREADS; t++)
    pthread_join(threads[t], NULL);

  for (int t = 0; t < THREADS; t++) {
    for (int i = 0; i < N_FEATURES * N_HIDDEN; i++)
      g->featureWeightGradients[i].g += gradients[t].featureWeightGradients[i].g;

    for (int i = 0; i < N_HIDDEN * N_OUTPUT; i++)
      g->hiddenWeightGradients[i].g += gradients[t].hiddenWeightGradients[i].g;

    for (int i = 0; i < N_HIDDEN; i++)
      g->hiddenBiasGradients[i].g += gradients[t].hiddenBiasGradients[i].g;

    for (int i = 0; i < N_OUTPUT; i++)
      g->outputBiasGradients[i].g += gradients[t].outputBiasGradients[i].g;
  }
}

void* CalculateGradients(void* arg) {
  UpdateGradientsJob* job = (UpdateGradientsJob*)arg;
  NN* nn = job->nn;
  NNGradients* gradients = job->gradients;

  for (int n = job->start; n < job->start + job->n; n++) {
    DataEntry entry = job->data->entries[n];

    NNActivations results[1];
    NNPredict(nn, entry.board, results);

    float out = Sigmoid(results->outputActivations[0]);
    float loss = SigmoidPrime(out) * ErrorGradient(out, &entry);

    gradients->outputBiasGradients[0].g += loss;
    for (int i = 0; i < N_HIDDEN; i++)
      gradients->hiddenWeightGradients[i].g += results->hiddenActivations[i] * loss;

    for (int i = 0; i < N_HIDDEN; i++) {
      float layerLoss = loss * nn->hiddenWeights[i] * (results->hiddenActivations[i] > 0.0f);
      if (!layerLoss)
        continue;

      gradients->hiddenBiasGradients[i].g += layerLoss;

      for (int a = 0; a < 32; a++)
        if (entry.board[a])
          gradients->featureWeightGradients[entry.board[a] * N_HIDDEN + i].g += layerLoss;
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
