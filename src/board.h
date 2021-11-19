#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "util.h"

INLINE Square mirror(Square s) { return s ^ 56; }

INLINE int8_t kIdx(Square k, Square s) { return 2 * ((k & 4) == (s & 4)) + !!(k & 32); }

INLINE Piece inv(Piece p) { return opposite[p]; }

INLINE Feature idx(Piece pc, Square sq, Square king, const Color view) {
  if (view == WHITE)
    return pc * 128 + kIdx(mirror(king), mirror(sq)) * 32 + psqt[mirror(sq)];
  else
    return inv(pc) * 128 + kIdx(king, sq) * 32 + psqt[sq];
}

INLINE Piece getPiece(uint8_t pieces[16], int n) { return (pieces[n / 2] >> ((n & 1) * 4)) & 0xF; }

void ToFeatures(Board* board, Features* f);
void ParseFen(char* fen, Board* board);

#endif