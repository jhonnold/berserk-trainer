#ifndef TYPES_H
#define TYPES_H

#include <inttypes.h>
#include <stdbool.h>

#define N_INPUT (16 * 12 * 32)
#define N_HIDDEN 512
#define N_OUTPUT 1

#define THREADS 16
#define BATCH_SIZE 16384

extern float ALPHA;
#define BETA1 0.95
#define BETA2 0.999
#define EPSILON 1e-8

#define WDL 0.5
#define EVAL 0.5

#define LAMBDA (1.0 / (1024 * 1024))

#define MAX_POSITIONS 1800000000
#define VALIDATION_POSITIONS 15000000

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
  uint32_t n;
  Board* entries;
} DataSet;

typedef struct {
  float outputBias;
  float outputWeights[2 * N_HIDDEN] ALIGN64;

  float inputBiases[N_HIDDEN] ALIGN64;
  float inputWeights[N_INPUT * N_HIDDEN] ALIGN64;
} NN;

typedef struct {
  float output;
  float accumulator[2 * N_HIDDEN] ALIGN64;
} ALIGN64 NetworkTrace;

typedef struct {
  float g, M, V;
} Gradient;

typedef struct {
  Gradient outputBias;
  Gradient outputWeights[2 * N_HIDDEN];

  Gradient inputBiases[N_HIDDEN];
  Gradient inputWeights[N_INPUT * N_HIDDEN];
} NNGradients;

typedef struct {
  float outputBias;
  float outputWeights[2 * N_HIDDEN];

  float inputBiases[N_HIDDEN];
  float inputWeights[N_INPUT * N_HIDDEN];
} BatchGradients;

extern int ITERATION;
extern int LAST_SEEN[N_INPUT];

extern const Square psqt[];
extern const Piece charToPiece[];
extern const Piece opposite[];
extern const int8_t scalar[];
extern const float SS;
extern const Feature kingBucket[64];

#endif