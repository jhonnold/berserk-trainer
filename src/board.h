#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "util.h"

INLINE Square mirror(Square s) { return s ^ 56; }

INLINE uint8_t kingQuadrantIndex(Square k, Square s) {
  uint8_t fileBit = (~(k ^ s) & 4u) >> 2;   // 1 or 0
  uint8_t rankBit = (~(k ^ s) & 32u) >> 4;  // 2 or 0

  // 0 if opposite quadrant
  // 1 if same file half, but different rank
  // 2 if same rank half, but different file
  // 3 if same quadrant
  return rankBit | fileBit;
}

INLINE uint8_t kingSideIndex(Square k, Square s) {
  uint8_t fileBit = (~(k ^ s) & 4u) >> 2;  // 1 or 0

  // 0 if opposite file half
  // 1 if same file half
  return fileBit;
}

INLINE Piece inv(Piece p) { return opposite[p]; }

INLINE Feature idx(Piece pc, Square sq, Square king, const Color view) {
  if (view == WHITE)
    king = mirror(king), sq = mirror(sq);
  else
    pc = inv(pc);

  int pieceIdx = pc * 64;
  int kingIdx = pc == WHITE_KING || pc == BLACK_KING ? 0 :  // king gets all 64 dedicated to square
                    pc == WHITE_PAWN || pc == BLACK_PAWN
                        ? kingSideIndex(king, sq) * 32       // pawns get king side relation
                        : kingQuadrantIndex(king, sq) * 16;  // pieces get quadrant relation
  int sqIdx =
      pc == WHITE_KING || pc == BLACK_KING ? sq : pc == WHITE_PAWN || pc == BLACK_PAWN ? psqt32[sq] : psqt16[sq];

  return pieceIdx + kingIdx + sqIdx;
}

INLINE Piece getPiece(uint8_t pieces[16], int n) { return (pieces[n / 2] >> ((n & 1) * 4)) & 0xF; }

void ToFeatures(Board* board, Features* f);
void ParseFen(char* fen, Board* board);

#endif