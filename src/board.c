#include <stdio.h>
#include <string.h>

#include "bits.h"
#include "board.h"

const int charToPiece[] = {['P'] = WHITE_PAWN,   ['N'] = WHITE_KNIGHT, ['B'] = WHITE_BISHOP, ['R'] = WHITE_ROOK,
                           ['Q'] = WHITE_QUEEN,  ['K'] = WHITE_KING,   ['p'] = BLACK_PAWN,   ['n'] = BLACK_KNIGHT,
                           ['b'] = BLACK_BISHOP, ['r'] = BLACK_ROOK,   ['q'] = BLACK_QUEEN,  ['k'] = BLACK_KING};

inline int mirror(int s) {
  return s ^ 56;
}

void ParseFen(char* fen, Board board, int stm) {
  int n = 0;

  for (int sq = 0; sq < 64; sq++) {
    char c = *fen;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      int pc = charToPiece[(int)*fen];
      int pSq = stm == BLACK ? sq : mirror(sq);
      if (stm == BLACK)
        pc = pc >= BLACK_PAWN ? pc - 6 : pc + 6;

      board[n++] = pc * 64 + pSq;
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
}