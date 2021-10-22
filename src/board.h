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

enum { WHITE, BLACK };

typedef struct {
  int8_t pc, sq;
} Piece;

typedef struct {
  int8_t wkingSq;
  int8_t bkingSq;
  Piece pieces[32];
  int8_t passers[2][8];
} Board;

typedef uint64_t BitBoard;

int8_t rank(int8_t sq);
int8_t file(int8_t sq);
int8_t mirror(int8_t s);
int8_t invertPiece(int8_t pc);
int8_t sameSideKing(int8_t sq, int8_t ksq);
int16_t feature(Piece p, int8_t kingSq, const int perspective);
int16_t passerFeature(int8_t sq, int color, const int perspective);

BitBoard fillN(BitBoard b);
BitBoard fillS(BitBoard b);
BitBoard passers(BitBoard w, BitBoard b, const int s);

void ParseFen(char* fen, Board* board);

#endif