#include "board.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bits.h"

void ToFeatures(Board* board, Features* f) {
  f->n = 0;

  uint64_t bb = board->occupancies;

  while (bb) {
    Square sq = popLsb(&bb);
    Piece pc = getPiece(board->pieces, f->n);

    f->features[WHITE][f->n] = idx(pc, sq, board->kings[WHITE], WHITE);
    f->features[BLACK][f->n] = idx(pc, sq, board->kings[BLACK], BLACK);

    f->n++;
  }
}

void ParseFen(char* fen, Board* board) {
  char* _fen = fen;
  int n = 0;

  // Make sure the board is empty
  board->occupancies = 0;
  for (int i = 0; i < 16; i++) board->pieces[i] = 0;

  board->kings[WHITE] = INT8_MAX;
  board->kings[BLACK] = INT8_MAX;

  for (Square sq = 0; sq < 64; sq++) {
    char c = *fen;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      Piece pc = charToPiece[(int)*fen];

      if (c == 'K')
        board->kings[WHITE] = sq;
      else if (c == 'k')
        board->kings[BLACK] = sq;

      setBit(board->occupancies, sq);
      board->pieces[n / 2] |= pc << ((n & 1) * 4);

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

  if (board->kings[WHITE] == INT8_MAX || board->kings[BLACK] == INT8_MAX) {
    printf("Unable to locate kings in FEN: %s!\n", _fen);
    exit(1);
  }
}