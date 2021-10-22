#ifndef BOARD_H
#define BOARD_H

#include <inttypes.h>

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

typedef struct {
  int8_t pc, sq;
} Piece;

typedef struct {
  int8_t wkingSq;
  int8_t bkingSq;
  Piece pieces[32];
} Board;

int8_t rank(int8_t sq);
int8_t file(int8_t sq);
int8_t mirror(int8_t s);
int8_t invertPiece(int8_t pc);
int8_t sameSideKing(int8_t sq, int8_t ksq);
int16_t feature(Board* board, int i, const int perspective);

void ParseFen(char* fen, Board* board);

#endif