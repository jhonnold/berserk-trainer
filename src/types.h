#ifndef TYPES_H
#define TYPES_H

#include <inttypes.h>
#include <stdbool.h>

#define N_INPUT 1536
#define N_HIDDEN 512
#define N_OUTPUT 1

#define THREADS 8
#define BATCH_SIZE 16384

extern float ALPHA;
#define BETA1 0.95
#define BETA2 0.999
#define EPSILON 1e-8

#define LAMBDA (1.0 / (1024 * 1024))

#define MAX_POSITIONS INT32_MAX
#define VALIDATION_POSITIONS (15 * 1000 * 1000)

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
  uint64_t occupancies;
  uint8_t pieces[16];
} __attribute__((packed, aligned(4))) Board;

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
  float acc1[2 * N_HIDDEN] ALIGN64;
} ALIGN64 NNAccumulators;

typedef struct {
  float mOutputBias, vOutputBias;
  float mOutputWeights[2 * N_HIDDEN] ALIGN64, vOutputWeights[2 * N_HIDDEN] ALIGN64;

  float mInputBiases[N_HIDDEN] ALIGN64, vInputBiases[N_HIDDEN] ALIGN64;
  float mInputWeights[N_INPUT * N_HIDDEN] ALIGN64, vInputWeights[N_INPUT * N_HIDDEN] ALIGN64;
} ALIGN64 Optimizer;

typedef struct {
  float outputBias;
  float outputWeights[2 * N_HIDDEN] ALIGN64;

  float inputBiases[N_HIDDEN] ALIGN64;
  float inputWeights[N_INPUT * N_HIDDEN] ALIGN64;
} ALIGN64 Gradients;

extern const Square psqt[];
extern const Piece charToPiece[];
extern const Piece opposite[];
extern const int8_t scalar[];
extern const float SS;

#endif