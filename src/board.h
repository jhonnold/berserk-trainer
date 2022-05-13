#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "util.h"

INLINE int8_t Bucket(int pieceCount) { return (pieceCount - 1) / 4; }

INLINE int8_t KingIndex(Square k, Square s) { return 2 * ((k & 4) == (s & 4)) + ((k & 32) == (s & 32)); }

INLINE Piece Invert(Piece p) { return OPPOSITE[p]; }

INLINE Feature idx(Piece pc, Square sq, Square king, const Color view) {
  int pieceIdx = view == WHITE ? pc : Invert(pc);
  int kingIdx = KingIndex(king, sq);
  int sqIdx = PSQT64_TO_32[((view == WHITE) * 56) ^ sq];

  return pieceIdx * 4 * 32  //
         + kingIdx * 32     //
         + sqIdx;
}

INLINE Piece getPiece(uint8_t pieces[16], int n) { return (pieces[n / 2] >> ((n & 1) * 4)) & 0xF; }

void ToFeatures(Board* board, Features* f);
void ParseFen(char* fen, Board* board);

#endif