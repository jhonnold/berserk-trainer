#ifndef BOARD_H
#define BOARD_H

#include "util.h"
#include "types.h"

INLINE Square mirror(Square s) { return s ^ 56; }

INLINE bool ss(Square s1, Square s2) { return (s1 & 4) == (s2 & 4); }

INLINE Piece inv(Piece p) { return opposite[p]; }

INLINE Feature idx(OccupiedSquare occ, Square king, const Color view) {
  if (view == WHITE)
    return occ.pc * 128 + ss(occ.sq, king) * 64 + occ.sq;
  else
    return inv(occ.pc) * 128 + ss(occ.sq, king) * 64 + mirror(occ.sq);
}

void ParseFen(char* fen, Board* board, Color stm);

#endif