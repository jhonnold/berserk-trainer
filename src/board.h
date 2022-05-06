#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "util.h"

INLINE Square orient(Square s, Square king, const Color view) { return (7 * !(king & 4)) ^ (56 * view) ^ s; }

INLINE Feature idx(Piece pc, Square sq, Square king, const Color view) {
  int pieceColor = pc < BLACK_PAWN ? WHITE : BLACK;
  int pieceType = pc % 6;
  int pieceIdx = min(10, pieceType * 2 + (pieceColor != view));

  int kingSq = orient(king, king, view);
  int pieceSq = orient(sq, king, view);

  return KING_BUCKETS[kingSq] * 11 * 64  //
         + pieceIdx * 64                 //
         + pieceSq;
}

INLINE Piece getPiece(uint8_t pieces[16], int n) { return (pieces[n / 2] >> ((n & 1) * 4)) & 0xF; }

void ToFeatures(Board* board, Features* f);
void ParseFen(char* fen, Board* board);

#endif