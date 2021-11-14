#ifndef TYPES_H
#define TYPES_H

#include <inttypes.h>
#include <stdbool.h>

#define N_INPUT 768
#define N_HIDDEN 256
#define N_HIDDEN_2 32
#define N_OUTPUT 1

#define THREADS 16
#define BATCH_SIZE 16384

#define ALPHA 0.01f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f

#define MAX_POSITIONS 1500000000
#define VALIDATION_POSITIONS 2500000

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
  Color stm;
  Square kings[2];
  uint64_t occupancies;
  uint8_t pieces[16];
} Board;

typedef struct {
  int8_t n;
  Feature features[2][32];
} Features;

typedef struct {
  int8_t wdl;
  float eval;
  Board board;
} DataEntry;

typedef struct {
  int n;
  DataEntry* entries;
} DataSet;

typedef struct {
  float outputBias;
  float outputWeights[N_HIDDEN_2] __attribute__((aligned(64)));

  float h2Biases[N_HIDDEN_2] __attribute__((aligned(64)));
  float h2Weights[2 * N_HIDDEN * N_HIDDEN_2] __attribute__((aligned(64)));

  float inputBiases[N_HIDDEN] __attribute__((aligned(64)));
  float inputWeights[N_INPUT * N_HIDDEN] __attribute__((aligned(64)));

  float psqtWeights[N_INPUT] __attribute__((aligned(64)));
} NN;

typedef struct {
  float output;
  float acc2[N_HIDDEN_2] __attribute__((aligned(64)));
  float acc1[2][N_HIDDEN] __attribute__((aligned(64)));
} NNAccumulators;

typedef struct {
  float g, M, V;
} Gradient;

typedef struct {
  Gradient outputBias;
  Gradient outputWeights[N_HIDDEN_2];

  Gradient h2Biases[N_HIDDEN_2];
  Gradient h2Weights[2 * N_HIDDEN * N_HIDDEN_2];

  Gradient inputBiases[N_HIDDEN];
  Gradient inputWeights[N_INPUT * N_HIDDEN];

  Gradient psqtWeights[N_INPUT];
} NNGradients;

typedef struct {
  float outputBias;
  float outputWeights[N_HIDDEN_2];

  float h2Biases[N_HIDDEN_2];
  float h2Weights[2 * N_HIDDEN * N_HIDDEN_2];

  float inputBiases[N_HIDDEN];
  float inputWeights[N_INPUT * N_HIDDEN];

  float psqtWeights[N_INPUT];
} BatchGradients;

extern const float psqtInitValues[];
extern const Square psqt[];
extern const Piece charToPiece[];
extern const Piece opposite[];
extern const int8_t scalar[];
extern const float SS;

#endif