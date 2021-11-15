#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "util.h"

INLINE Square mirror(Square s) { return s ^ 56; }

INLINE int8_t kIdx(Square k, Square s) { return (k & 4) == (s & 4); }

INLINE Piece inv(Piece p) { return opposite[p]; }

INLINE Feature idx(Piece pc, Square sq, Square king, const Color view) {
  if (view == WHITE)
    return (pc < WHITE_KNIGHT) * (pc * 64 * 48 + mirror(king) * 48 + mirror(sq) - 8) + //
           (pc >= WHITE_KNIGHT) * (2 * 64 * 48 + (pc - 2) * 2 * 32 + kIdx(king, sq) * 32 + psqt[mirror(sq)]);
  else
    return (pc < WHITE_KNIGHT) * (inv(pc) * 64 * 48 + king * 48 + sq - 8) + //
           (pc >= WHITE_KNIGHT) * (2 * 64 * 48 + (inv(pc) - 2) * 2 * 32 + kIdx(king, sq) * 32 + psqt[sq]);
}

INLINE Piece getPiece(uint8_t pieces[16], int n) { return (pieces[n / 2] >> ((n & 1) * 4)) & 0xF; }

void ToFeatures(Board* board, Features* f);
void ParseFen(char* fen, Board* board);

#endif