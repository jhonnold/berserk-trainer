#ifndef TYPES_H
#define TYPES_H

#include <inttypes.h>
#include <stdbool.h>

#define N_INPUT 1536
#define N_HIDDEN 384
#define N_HIDDEN_2 16
#define N_HIDDEN_3 32
#define N_OUTPUT 1

#define THREADS 4
#define BATCH_SIZE 16384
#define BATCHES_PER_LOAD 32768

extern float ALPHA;
#define BETA1 0.95
#define BETA2 0.999
#define EPSILON 1e-8

#define WDL 0.5
#define EVAL 0.5

#define LAMBDA (1.0f / (1024 * 1024))

#define CRELU_MAX 256

#define ALIGN64 __attribute__((aligned(64)))

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
  float eval;
  uint64_t occupancies;
  uint8_t pieces[16];
} Board;

typedef struct {
  int8_t n;
  Feature features[32][2];
} Features;

typedef struct {
  uint64_t n;
  Board* entries;
} DataSet;

typedef struct {
  float outputBias;
  float outputWeights[N_HIDDEN_3] ALIGN64;

  float l2Biases[N_HIDDEN_3] ALIGN64;
  float l2Weights[N_HIDDEN_2 * N_HIDDEN_3] ALIGN64;

  float l1Biases[N_HIDDEN_2] ALIGN64;
  float l1Weights[2 * N_HIDDEN * N_HIDDEN_2] ALIGN64;

  float inputBiases[N_HIDDEN] ALIGN64;
  float inputWeights[N_INPUT * N_HIDDEN] ALIGN64;
} NN;

typedef struct {
  float output;
  float l2Accumulator[N_HIDDEN_3] ALIGN64;
  float l1Accumulator[N_HIDDEN_2] ALIGN64;
  float accumulator[2 * N_HIDDEN] ALIGN64;
} ALIGN64 NetworkTrace;

typedef struct {
  float M, V;
} Gradient;

typedef struct {
  Gradient outputBias;
  Gradient outputWeights[N_HIDDEN_3] ALIGN64;

  Gradient l2Biases[N_HIDDEN_3] ALIGN64;
  Gradient l2Weights[N_HIDDEN_2 * N_HIDDEN_3] ALIGN64;

  Gradient l1Biases[N_HIDDEN_2] ALIGN64;
  Gradient l1Weights[2 * N_HIDDEN * N_HIDDEN_2] ALIGN64;

  Gradient inputBiases[N_HIDDEN] ALIGN64;
  Gradient inputWeights[N_INPUT * N_HIDDEN] ALIGN64;
} NNGradients;

typedef struct {
  float outputBias;
  float outputWeights[N_HIDDEN_3] ALIGN64;

  float l2Biases[N_HIDDEN_3] ALIGN64;
  float l2Weights[N_HIDDEN_2 * N_HIDDEN_3] ALIGN64;

  float l1Biases[N_HIDDEN_2] ALIGN64;
  float l1Weights[2 * N_HIDDEN * N_HIDDEN_2] ALIGN64;

  float inputBiases[N_HIDDEN] ALIGN64;
  float inputWeights[N_INPUT * N_HIDDEN] ALIGN64;
} BatchGradients;

extern const Square psqt[];
extern const Piece charToPiece[];
extern const Piece opposite[];
extern const int8_t scalar[];
extern const float SS;

#endif