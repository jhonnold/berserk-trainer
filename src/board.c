#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "bits.h"
#include "board.h"

const int8_t charToPiece[] = {['P'] = WHITE_PAWN,   ['N'] = WHITE_KNIGHT, ['B'] = WHITE_BISHOP, ['R'] = WHITE_ROOK,
                              ['Q'] = WHITE_QUEEN,  ['K'] = WHITE_KING,   ['p'] = BLACK_PAWN,   ['n'] = BLACK_KNIGHT,
                              ['b'] = BLACK_BISHOP, ['r'] = BLACK_ROOK,   ['q'] = BLACK_QUEEN,  ['k'] = BLACK_KING};

const int8_t invertMap[] = {BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING,
                            WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING};

inline int8_t rank(int8_t sq) { return sq >> 3; }

inline int8_t file(int8_t sq) { return sq & 7; }

inline int8_t mirror(int8_t s) { return s ^ 56; }

inline int8_t invertPiece(int8_t pc) { return invertMap[pc]; }

inline int8_t inKingArea(int8_t sq, int8_t ksq) {
  return abs(rank(sq) - rank(ksq)) < 3 && abs(file(sq) - file(ksq)) < 3;
}

inline int16_t feature(Piece p, int8_t kingSq, const int perspective) {
  if (perspective == WHITE)
    return p.pc * 128 + inKingArea(p.sq, kingSq) * 64 + p.sq;
  else
    return invertPiece(p.pc) * 128 + inKingArea(p.sq, kingSq) * 64 + mirror(p.sq);
}

void ParseFen(char* fen, Board* board) {
  int n = 0;
  char* _fen = fen;

  for (int8_t sq = 0; sq < 64; sq++) {
    char c = *fen;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      int pc = charToPiece[(int)*fen];

      if (c == 'k')
        board->bkingSq = mirror(sq);
      else if (c == 'K')
        board->wkingSq = mirror(sq);

      board->pieces[n].pc = pc;
      board->pieces[n].sq = mirror(sq);

      n++;
    } else if (c >= '1' && c <= '8')
      sq += (c - '1');
    else if (c == '/')
      sq--;
    else {
      printf("Unable to parse FEN: %s!\n", _fen);
      exit(1);
    }

    fen++;
  }

  if (n < 32) {
    board->pieces[n].pc = -1;
    board->pieces[n].sq = -1;
  }

  if (board->wkingSq < 0 || board->bkingSq < 0) {
    printf("Unable to locate kings in FEN: %s!\n", _fen);
    exit(1);
  }
}