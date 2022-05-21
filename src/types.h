#ifndef TYPES_H
#define TYPES_H

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>

#define N_INPUT (12 * 2 * 64)
#define N_HIDDEN 512
#define N_L1 (2 * N_HIDDEN)
#define N_L2 8
#define N_OUTPUT 1

#define THREADS 16

// total fens in berserk9dev2.d9.bin - 2098790400
#define BATCH_SIZE 16384
#define BATCHES_PER_LOAD 6100

extern float ALPHA;
#define BETA1 0.95
#define BETA2 0.999
#define EPSILON 1e-8

#define STEP_RATE 100
#define GAMMA 0.1f

#define WDL 0.5
#define EVAL 0.5

#define LAMBDA (1.0 / (1024 * 1024))

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
  FILE* fin;
  uint64_t entriesCount;

  DataSet* nextData;
} CyclicalLoadArgs;

typedef struct {
  float outputBias;
  float outputWeights[N_L2] ALIGN64;

  float l2Biases[N_L2] ALIGN64;
  float l2Weights[N_L1 * N_L2] ALIGN64;

  float inputBiases[N_HIDDEN] ALIGN64;
  float inputWeights[N_INPUT * N_HIDDEN] ALIGN64;
} NN;

typedef struct {
  float output;
  float l2Acc[N_L2] ALIGN64;
  float accumulator[N_L1] ALIGN64;
} ALIGN64 NetworkTrace;

typedef struct {
  float M, V;
} Gradient;

typedef struct {
  Gradient outputBias;
  Gradient outputWeights[N_L2] ALIGN64;

  Gradient l2Biases[N_L2] ALIGN64;
  Gradient l2Weights[N_L1 * N_L2] ALIGN64;

  Gradient inputBiases[N_HIDDEN] ALIGN64;
  Gradient inputWeights[N_INPUT * N_HIDDEN] ALIGN64;
} NNGradients;

typedef struct {
  float outputBias;
  float outputWeights[N_L2] ALIGN64;

  float l2Biases[N_L2] ALIGN64;
  float l2Weights[N_L1 * N_L2] ALIGN64;

  float inputBiases[N_HIDDEN] ALIGN64;
  float inputWeights[N_INPUT * N_HIDDEN] ALIGN64;
} BatchGradients;

extern int ITERATION;
extern int LAST_SEEN[N_INPUT];
extern const Piece OPPOSITE[12];
extern const Feature KING_BUCKETS[64];
extern const Square PSQT64_TO_32[64];
extern const Piece CHAR_TO_PIECE[];
extern const float SS;

#endif