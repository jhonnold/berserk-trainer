#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bits.h"
#include "board.h"

void ParseFen(char* fen, Board* board, Color stm) {
  char* _fen = fen;
  int n = 0;

  // Make sure the board is empty
  for (int i = 0; i < 16; i++)
    board->pieces[i] = 0;

  board->wk = INT8_MAX;
  board->bk = INT8_MAX;

  for (Square sq = 0; sq < 64; sq++) {
    char c = *fen;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      Piece pc = charToPiece[(int)*fen];

      if (c == 'k') {
        if (stm == WHITE)
          board->bk = mirror(sq);
        else
          board->wk = sq;
      } else if (c == 'K') {
        if (stm == WHITE)
          board->wk = mirror(sq);
        else
          board->bk = sq;
      }

      setBit(board->occupancies, stm == WHITE ? mirror(sq) : sq);
      board->pieces[n / 2] |= (stm == WHITE ? pc : inv(pc)) << ((n & 1) * 4);

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

  if (board->wk == INT8_MAX || board->bk == INT8_MAX) {
    printf("Unable to locate kings in FEN: %s!\n", _fen);
    exit(1);
  }
}