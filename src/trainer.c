#include "trainer.h"

#include <immintrin.h>
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

  char nnPath[128] = {0};

  char samplesPath[128] = {0};
  int numberOfFiles = 1;
  int batchesPerFile = 10000;

  char validationPath[128] = {0};
  int numberOfValidations = 1000000;

  int c;
  while ((c = getopt(argc, argv, "n:s:c:b:v:z:")) != -1) {
    switch (c) {
      case 'h':
        printf("\nSYNOPSIS");
        printf("\n\t./trainer -n <network_file>       (default random net)");
        printf("\n\t          -s <samples_file>       (required)");
        printf("\n\t          -c [number_of_files]    (default 1)");
        printf("\n\t          -b [batches_per_file]   (default 10000)\n");
        printf("\nOPTIONS");
        printf("\n\t-n network_file");
        printf("\n\t\tNetwork file to load and start with for training. ");
        printf("If not specified, a random one will be generated\n");
        printf("\n\t-s samples_file");
        printf("\n\t\tFile(s) for training samples. They will be loaded as <file>.#.binpack\n");
        printf("\n\t-c number_of_files");
        printf("\n\t\tThe number of files to load.\n");
        printf("\n\t-b batches_per_file");
        printf("\n\t\tThe number of batches in each file.\n");
        exit(EXIT_SUCCESS);
      case 'n':
        strcpy(nnPath, optarg);
        break;
      case 's':
        strcpy(samplesPath, optarg);
        break;
      case 'b':
        batchesPerFile = atoi(optarg);
        break;
      case 'c':
        numberOfFiles = atoi(optarg);
        break;
      case 'v':
        strcpy(validationPath, optarg);
        break;
      case 'z':
        numberOfValidations = atoi(optarg);
        break;
      case '?':
        exit(EXIT_FAILURE);
    }
  }

  if (!samplesPath[0]) {
    printf("No samples file specified!\n");
    return 1;
  } else if (!validationPath[0]) {
    printf("No validation path specified!\n");
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

  Optimizer* optimizer = malloc(sizeof(Optimizer));
  memset(optimizer, 0, sizeof(Optimizer));

  Gradients* gradients = malloc(sizeof(Gradients) * THREADS);

  printf("Loading validation from %s\n", validationPath);
  DataSet* validation = malloc(sizeof(DataSet));
  validation->n = 0;
  validation->entries = malloc(sizeof(Board) * numberOfValidations);
  LoadBinpack(validation, validationPath, numberOfValidations);

  printf("Calculating Validation Error...\n");
  float error = TotalError(validation, nn);
  printf("Starting Error: [%1.8f]\n", error);

  for (int epoch = 1; epoch <= 22; epoch++) {
    long epochStart = GetTimeMS();

    for (int f = 1; f <= numberOfFiles; f++) {
      DataSet* data = malloc(sizeof(DataSet));
      data->n = 0;
      data->entries = malloc(sizeof(Board) * batchesPerFile * BATCH_SIZE);

      char fileToLoad[128];
      sprintf(fileToLoad, "%s.%d.binpack", samplesPath, f);
      printf("Loading entries from %s\n", fileToLoad);
      LoadBinpack(data, fileToLoad, batchesPerFile * BATCH_SIZE);

      printf("Shuffling...\n");
      ShuffleData(data);
      
      uint32_t bb = (f - 1) * batchesPerFile;
      uint32_t batches = data->n / BATCH_SIZE;
      for (uint32_t b = 0; b < batches; b++) {
        Train(b, data, nn, gradients);
        ApplyGradients(nn, optimizer, gradients);

        printf("Batch: [#%d/%d]\n", bb + b + 1, numberOfFiles * batchesPerFile);
      }

      free(data->entries), free(data);
    }

    char buffer[64];
    sprintf(buffer, "../nets/berserk-kq.e%d.2x%d.nn", epoch, N_HIDDEN);
    SaveNN(nn, buffer);

    printf("Calculating Validation Error...\n");
    float newError = TotalError(validation, nn);

    long now = GetTimeMS();
    printf("Epoch: [#%5d], Error: [%1.8f], Delta: [%+1.8f], LR: [%.5f], Speed: [%9.0f pos/s], Time: [%lds]\n", epoch,
           newError, error - newError, ALPHA,
           1000.0 * (numberOfFiles * batchesPerFile * BATCH_SIZE) / (now - epochStart), (now - epochStart) / 1000);

    error = newError;

    // LR DROP
    if (epoch == 20) ALPHA = 0.001;

    if (epoch == 21) ALPHA = 0.0001;
  }
}

float TotalError(DataSet* data, NN* nn) {
  float e = 0.0;

#pragma omp parallel for schedule(auto) num_threads(THREADS) reduction(+ : e)
  for (uint32_t i = 0; i < data->n; i++) {
    Board* board = &data->entries[i];

    NNAccumulators results[1];
    Features f[1];

    ToFeatures(board, f);
    NNPredict(nn, f, board->stm, results);

    e += Error(Sigmoid(results->output), board);
  }

  return e / data->n;
}

void Train(int batch, DataSet* data, NN* nn, Gradients* gradients) {
  for (int t = 0; t < THREADS; t++) memset(&gradients[t], 0, sizeof(Gradients));

#pragma omp parallel for schedule(auto) num_threads(THREADS)
  for (int n = 0; n < BATCH_SIZE; n++) {
    const int t = omp_get_thread_num();

    Board board = data->entries[n + batch * BATCH_SIZE];

    NNAccumulators activations[1];
    Features f[1];

    ToFeatures(&board, f);
    NNPredict(nn, f, board.stm, activations);

    float out = Sigmoid(activations->output);
    float delta = SigmoidPrime(out) * ErrorGradient(out, &board);

    // LOSS CALCULATIONS ------------------------------------------------------------------------
    const size_t WIDTH = sizeof(__m256) / sizeof(float);
    const size_t CHUNKS = N_HIDDEN / WIDTH;

    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0);
    const __m256 lambda = _mm256_set1_ps(LAMBDA);

    __m256 outputLoss = _mm256_set1_ps(delta);

    __m256 hiddenLosses[2 * CHUNKS] ALIGN64;

    __m256* acc = (__m256*)activations->acc1;
    __m256* nnOutputWeights = (__m256*)nn->outputWeights;

    for (size_t i = 0; i < 2 * CHUNKS; i++) {
      __m256 reluPrime = _mm256_blendv_ps(zero, one, _mm256_cmp_ps(acc[i], zero, 30));

      hiddenLosses[i] = _mm256_mul_ps(outputLoss, _mm256_mul_ps(nnOutputWeights[i], reluPrime));
    }
    // ------------------------------------------------------------------------------------------

    // OUTPUT LAYER GRADIENTS -------------------------------------------------------------------
    gradients[t].outputBias += delta;

    __m256* outputWeights = (__m256*)gradients[t].outputWeights;
    for (size_t i = 0; i < 2 * CHUNKS; i++)
      outputWeights[i] = _mm256_add_ps(outputWeights[i], _mm256_mul_ps(acc[i], outputLoss));
    // ------------------------------------------------------------------------------------------

    // INPUT LAYER GRADIENTS --------------------------------------------------------------------
    __m256* stmActivations = &acc[0];
    __m256* xstmActivations = &acc[CHUNKS];

    __m256* stmLosses = &hiddenLosses[0];
    __m256* xstmLosses = &hiddenLosses[CHUNKS];

    __m256* biasGradients = (__m256*)gradients[t].inputBiases;
    __m256* weightGradients = (__m256*)gradients[t].inputWeights;

    for (size_t i = 0; i < CHUNKS; i++) {
      __m256 stmLasso = _mm256_blendv_ps(zero, lambda, _mm256_cmp_ps(stmActivations[i], zero, 30));
      __m256 xstmLasso = _mm256_blendv_ps(zero, lambda, _mm256_cmp_ps(xstmActivations[i], zero, 30));

      __m256 lassos = _mm256_add_ps(stmLasso, xstmLasso);
      __m256 losses = _mm256_add_ps(stmLosses[i], xstmLosses[i]);

      biasGradients[i] = _mm256_add_ps(biasGradients[i], _mm256_add_ps(lassos, losses));
    }

    for (int i = 0; i < f->n; i++) {
      for (size_t j = 0; j < CHUNKS; j++) {
        __m256 stmLasso = _mm256_blendv_ps(zero, lambda, _mm256_cmp_ps(stmActivations[j], zero, 30));
        __m256 xstmLasso = _mm256_blendv_ps(zero, lambda, _mm256_cmp_ps(xstmActivations[j], zero, 30));

        weightGradients[f->features[i][board.stm] * CHUNKS + j] = _mm256_add_ps(
            weightGradients[f->features[i][board.stm] * CHUNKS + j], _mm256_add_ps(stmLosses[j], stmLasso));
        weightGradients[f->features[i][board.stm ^ 1] * CHUNKS + j] = _mm256_add_ps(
            weightGradients[f->features[i][board.stm ^ 1] * CHUNKS + j], _mm256_add_ps(xstmLosses[j], xstmLasso));
      }
    }
    // ------------------------------------------------------------------------------------------
  }
}