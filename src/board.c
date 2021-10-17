#include <stdio.h>
#include <string.h>

#include "bits.h"
#include "board.h"

const int8_t charToPiece[] = {['P'] = WHITE_PAWN,   ['N'] = WHITE_KNIGHT, ['B'] = WHITE_BISHOP, ['R'] = WHITE_ROOK,
                              ['Q'] = WHITE_QUEEN,  ['K'] = WHITE_KING,   ['p'] = BLACK_PAWN,   ['n'] = BLACK_KNIGHT,
                              ['b'] = BLACK_BISHOP, ['r'] = BLACK_ROOK,   ['q'] = BLACK_QUEEN,  ['k'] = BLACK_KING};

const int8_t invertMap[] = {BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING,
                            WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING};

inline int8_t mirror(int8_t s) { return s ^ 56; }

inline int8_t invertPiece(int8_t pc) { return invertMap[pc]; }

inline int16_t feature(Piece p, const int side) {
  if (side == WHITE)
    return p.pc * 64 + p.sq;
  else
    return invertPiece(p.pc) * 64 + mirror(p.sq);
}

void ParseFen(char* fen, Board board) {
  int n = 0;

  for (int8_t sq = 0; sq < 64; sq++) {
    char c = *fen;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      int pc = charToPiece[(int)*fen];

      board[n].pc = pc;
      board[n].sq = mirror(sq);

      n++;
    } else if (c >= '1' && c <= '8')
      sq += (c - '1');
    else if (c == '/')
      sq--;
    else {
      printf("Unable to parse FEN: %s!\n", fen);
      return;
    }

    fen++;
  }

  if (n < 32) {
    board[n].pc = -1;
    board[n].sq = -1;
  }
}