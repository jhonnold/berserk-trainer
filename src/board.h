#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "util.h"

INLINE Square mirror(Square s) { return s ^ 56; }

INLINE int8_t kIdx(Square k) { return !!(k & 4); }

INLINE Piece inv(Piece p) { return opposite[p]; }

INLINE Feature idx(Piece pc, Square sq, Square king, const Color view) {
  if (view == WHITE)
    return pc * 128 + kIdx(mirror(king)) * 64 + mirror(sq);
  else
    return inv(pc) * 128 + kIdx(king) * 64 + sq;
}

INLINE Piece getPiece(uint8_t pieces[16], int n) { return (pieces[n / 2] >> ((n & 1) * 4)) & 0xF; }

INLINE int bucket(int pieceCount) { return (pieceCount - 1) / 16; }

void ParseFen(char* fen, Board* board);

#endif