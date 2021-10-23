#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "board.h"

void ParseFen(char* fen, Board* board) {
  char* _fen = fen;

  board->pieces = malloc(sizeof(OccupiedSquare) * 32);
  board->n = 0;
  board->wk = INT8_MAX;
  board->bk = INT8_MAX;

  for (Square sq = 0; sq < 64; sq++) {
    char c = *fen;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      Piece pc = charToPiece[(int)*fen];

      if (c == 'k')
        board->bk = mirror(sq);
      else if (c == 'K')
        board->wk = mirror(sq);

      board->pieces[board->n].pc = pc;
      board->pieces[board->n].sq = mirror(sq);

      board->n++;
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

  board->pieces = realloc(board->pieces, sizeof(OccupiedSquare) * board->n);
}