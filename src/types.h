#ifndef TYPES_H
#define TYPES_H

#include <inttypes.h>
#include <stdbool.h>

#define N_INPUT 768
#define N_HIDDEN 512
#define N_OUTPUT 1

#define N_P_INPUT 6144
#define N_P_HIDDEN 128
#define N_P_OUTPUT 1

#define THREADS 16
#define BATCH_SIZE 16384

#define ALPHA 0.01f
#define BETA1 0.95f
#define BETA2 0.999f
#define EPSILON 1e-8f

#define LAMBDA (1.0 / (1024 * 1024))

#define MAX_POSITIONS 1750000000
#define VALIDATION_POSITIONS 10000000

#define CRELU_MAX 256

enum {
  WHITE_PAWN,
  WHITE_KNIGHT,
  WHITE_BISHOP,
  WHITE_ROOK,
  WHITE_QUEEN,
  WHITE_KING,
  BLACK_PAWN,
  BLACK_KNIGHT,
  BLACK_BISHOP,
  BLACK_ROOK,
  BLACK_QUEEN,
  BLACK_KING,
};

enum { WHITE, BLACK };

typedef uint8_t Color;
typedef uint8_t Square;
typedef uint8_t Piece;
typedef uint16_t Feature;

typedef struct {
  Color stm, wdl;
  Square kings[2];
  uint64_t occupancies;
  uint8_t pieces[16];
} Board;

typedef struct {
  int8_t n, p;
  Feature features[2][32];
  Feature pawnFeatures[2][16];
} Features;

typedef struct {
  int n;
  Board* entries;
} DataSet;

typedef struct {
  float outputBias;
  float outputWeights[2 * N_HIDDEN] __attribute__((aligned(64)));

  float inputBiases[N_HIDDEN] __attribute__((aligned(64)));
  float inputWeights[N_INPUT * N_HIDDEN] __attribute__((aligned(64)));

  float pawnOutputBias;
  float pawnOutputWeights[2 * N_P_HIDDEN] __attribute__((aligned(64)));

  float pawnInputBiases[N_P_HIDDEN] __attribute__((aligned(64)));
  float pawnInputWeights[N_P_INPUT * N_P_HIDDEN] __attribute__((aligned(64)));
} NN;

typedef struct {
  float output;
  float acc1[2][N_HIDDEN] __attribute__((aligned(64)));
  float pAcc1[2][N_P_HIDDEN] __attribute__((aligned(64)));
} __attribute__((aligned(64))) NNAccumulators;

typedef struct {
  float g, M, V;
} Gradient;

typedef struct {
  Gradient outputBias;
  Gradient outputWeights[2 * N_HIDDEN];

  Gradient inputBiases[N_HIDDEN];
  Gradient inputWeights[N_INPUT * N_HIDDEN];

  Gradient pawnOutputBias;
  Gradient pawnOutputWeights[2 * N_P_HIDDEN];

  Gradient pawnInputBiases[N_P_HIDDEN];
  Gradient pawnInputWeights[N_P_INPUT * N_P_HIDDEN];
} NNGradients;

typedef struct {
  float outputBias;
  float outputWeights[2 * N_HIDDEN];

  float inputBiases[N_HIDDEN];
  float inputWeights[N_INPUT * N_HIDDEN];

  float pawnOutputBias;
  float pawnOutputWeights[2 * N_P_HIDDEN];

  float pawnInputBiases[N_P_HIDDEN];
  float pawnInputWeights[N_P_INPUT * N_P_HIDDEN];
} BatchGradients;

extern const Square psqt[];
extern const Piece charToPiece[];
extern const Piece opposite[];
extern const int8_t scalar[];
extern const float SS;

#endif