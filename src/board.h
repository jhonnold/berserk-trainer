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

enum {
  WHITE,
  BLACK
};

typedef struct {
  int8_t pc, sq;
} Piece;
typedef Piece Board[32];

int8_t mirror(int8_t s);
int8_t invertPiece(int8_t pc);
int16_t feature(Piece p, const int side);

void ParseFen(char* fen, Board board);

#endif