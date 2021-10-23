#ifndef TYPES_H
#define TYPES_H

#include <inttypes.h>
#include <stdbool.h>

#define N_FEATURES 1536
#define N_HIDDEN 256
#define N_OUTPUT 1

#define THREADS 24
#define BATCH_SIZE 16384

#define ALPHA 0.01f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f

#define MAX_POSITIONS 1000000000

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
  N_PIECES
};

enum { WHITE, BLACK };

typedef uint8_t Square;
typedef uint8_t Piece;
typedef uint16_t Feature;

typedef struct {
  Square sq;
  Piece pc;
} OccupiedSquare;

typedef struct {
  int8_t n;
  Square wk, bk;
  OccupiedSquare* pieces;
} Board;

typedef struct {
  int8_t stm;
  float wdl, eval;
  Board board;
} DataEntry;

typedef struct {
  int n;
  DataEntry* entries;
} DataSet;

typedef struct {
  float outputBias;
  float featureWeights[N_FEATURES * N_HIDDEN] __attribute__((aligned(64)));
  float hiddenBiases[N_HIDDEN] __attribute__((aligned(64)));
  float hiddenWeights[N_HIDDEN * 2] __attribute__((aligned(64)));
} NN;

typedef struct {
  float result;
  float accumulators[2][N_HIDDEN] __attribute__((aligned(64)));
} NNActivations;

typedef struct {
  float g, M, V;
} Gradient;

typedef struct {
  Gradient outputBiasGradient;
  Gradient featureWeightGradients[N_FEATURES * N_HIDDEN];
  Gradient hiddenBiasGradients[N_HIDDEN];
  Gradient hiddenWeightGradients[N_HIDDEN * 2];
} NNGradients;

typedef struct {
  float outputBias;
  float featureWeights[N_FEATURES * N_HIDDEN];
  float hiddenBias[N_HIDDEN];
  float hiddenWeights[N_HIDDEN * 2];
} BatchGradients;

extern const Piece charToPiece[];
extern const Piece opposite[];
extern const float SS;

#endif