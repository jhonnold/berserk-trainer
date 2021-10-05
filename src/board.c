#include <stdio.h>

#include "bits.h"
#include "board.h"

const int charToPiece[] = {['P'] = WHITE_PAWN,   ['N'] = WHITE_KNIGHT, ['B'] = WHITE_BISHOP, ['R'] = WHITE_ROOK,
                           ['Q'] = WHITE_QUEEN,  ['K'] = WHITE_KING,   ['p'] = BLACK_PAWN,   ['n'] = BLACK_KNIGHT,
                           ['b'] = BLACK_BISHOP, ['r'] = BLACK_ROOK,   ['q'] = BLACK_QUEEN,  ['k'] = BLACK_KING};

int m(int s) {
  return s ^ 56;
}

void ParseFen(char* fen, Board board) {
  int n = 0;

  for (int sq = 0; sq < 64; sq++) {
    char c = *fen;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
      board[n++] = charToPiece[(int)*fen] * 64 + m(sq);
    else if (c >= '1' && c <= '8')
      sq += (c - '1');
    else if (c == '/')
      sq--;
    else {
      printf("Unable to parse FEN: %s!\n", fen);
      return;
    }

    fen++;
  }

  // printf("[");
  // int i;
  // for (i = 0; i < 32; i++)
  //   if (board[i])
  //     printf("%d,", board[i]);
  //   else
  //     break;

  // printf("] (%d)\n", i);
}