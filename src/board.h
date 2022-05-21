#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "util.h"

INLINE Piece Invert(Piece p) { return (p + 6) % 12; }

INLINE Feature idx(Piece pc, Square sq, Square king, const Color view) {
  int oP = view == WHITE ? pc : Invert(pc);
  int oSq = (7 * !(king & 4)) ^ (56 * view) ^ sq;

  return oP * 64 //
    + oSq;
}

INLINE Piece getPiece(uint8_t pieces[16], int n) { return (pieces[n / 2] >> ((n & 1) * 4)) & 0xF; }

void ToFeatures(Board* board, Features* f);
void ParseFen(char* fen, Board* board);

#endif