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

typedef uint64_t BitBoard;
typedef int16_t Board[32];

void ParseFen(char* fen, Board board);

#endif