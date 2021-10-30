#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "util.h"

INLINE Square mirror(Square s) { return s ^ 56; }

INLINE int8_t kIdx(Square k) { return 2 * !!(k & 4) + !!(k & 32); }

INLINE Piece inv(Piece p) { return opposite[p]; }

INLINE Feature idx(OccupiedSquare occ, Square king, const Color view) {
  if (view == WHITE)
    return occ.pc * 256 + kIdx(king) * 64 + occ.sq;
  else
    return inv(occ.pc) * 256 + kIdx(mirror(king)) * 64 + mirror(occ.sq);
}

INLINE Feature kpIdx(OccupiedSquare occ, const Color view) {
  if (view == WHITE)
    return pieceToKP[WHITE][occ.pc] * 64 + occ.sq;
  else
    return pieceToKP[BLACK][occ.pc] * 64 + mirror(occ.sq);
}

void ParseFen(char* fen, Board* board, Color stm);

#endif