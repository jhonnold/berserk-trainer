#ifndef TYPES_H
#define TYPES_H

#include <inttypes.h>
#include <stdbool.h>

#define N_INPUT 768
#define N_HIDDEN 512

#define N_PAWN_INPUT 224
#define N_PAWN_HIDDEN 64

#define N_OUTPUT 1


#define THREADS 8
#define BATCH_SIZE 16384

#define ALPHA 0.01f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f

#define MAX_POSITIONS 1500000000
#define VALIDATION_POSITIONS 1000000

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
  int8_t n, p;
  Feature features[2][32];
  Features pawnFeatures[2][18];
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
  float outputWeights[2 * N_HIDDEN] __attribute__((aligned(64)));

  float inputBiases[N_HIDDEN] __attribute__((aligned(64)));
  float inputWeights[N_INPUT * N_HIDDEN] __attribute__((aligned(64)));

  // uses same output bias
  float pawnOutputWeights[2 * N_PAWN_HIDDEN] __attribute__((aligned(64)));

  float pawnInputBiases[N_PAWN_HIDDEN] __attribute__((aligned(64)));
  float pawnInputWeights[N_PAWN_INPUT * N_PAWN_HIDDEN] __attribute__((aligned(64)));
} NN;

typedef struct {
  float output;
  float acc1[2][N_HIDDEN] __attribute__((aligned(64)));
  float pawnAcc1[2][N_PAWN_HIDDEN] __attribute__((aligned(64)));
} NNAccumulators;

typedef struct {
  float g, M, V;
} Gradient;

typedef struct {
  Gradient outputBias;
  Gradient outputWeights[2 * N_HIDDEN];

  Gradient inputBiases[N_HIDDEN];
  Gradient inputWeights[N_INPUT * N_HIDDEN];

  Gradient pawnOutputWeights[2 * N_PAWN_HIDDEN];

  Gradient pawnInputBiases[N_PAWN_HIDDEN];
  Gradient pawnInputWeights[N_PAWN_INPUT * N_PAWN_HIDDEN];
} NNGradients;

typedef struct {
  float outputBias;
  float outputWeights[2 * N_HIDDEN];

  float inputBiases[N_HIDDEN];
  float inputWeights[N_INPUT * N_HIDDEN];

  float pawnOutputWeights[2 * N_PAWN_HIDDEN];

  float pawnInputBiases[N_PAWN_HIDDEN];
  float pawnInputWeights[N_PAWN_INPUT * N_PAWN_HIDDEN];
} BatchGradients;

extern const Feature pawnIdxOffset[];
extern const Square psqt[];
extern const Piece charToPiece[];
extern const Piece opposite[];
extern const int8_t scalar[];
extern const float SS;

#endif