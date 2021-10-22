#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

inline int8_t sameSideKing(int8_t sq, int8_t ksq) { return (sq & 0x04) == (ksq & 0x04); }

inline int16_t feature(Piece p, int8_t kingSq, const int perspective) {
  if (perspective == WHITE)
    return p.pc * 128 + sameSideKing(p.sq, kingSq) * 64 + p.sq;
  else
    return invertPiece(p.pc) * 128 + sameSideKing(p.sq, kingSq) * 64 + mirror(p.sq);
}

inline int16_t passerFeature(int8_t sq, int color, const int perspective) {
  if (perspective == WHITE)
    return 1536 + color * 48 + sq - 8;
  else
    return 1536 + (color ^ 1) * 48 + mirror(sq) - 8;
}

inline BitBoard fillN(BitBoard b) {
  b |= b << 32;
  b |= b << 16;
  b |= b << 8;
  return b;
}

inline BitBoard fillS(BitBoard b) {
  b |= b >> 32;
  b |= b >> 16;
  b |= b >> 8;
  return b;
}

inline BitBoard passers(BitBoard w, BitBoard b, const int s) {
  if (s == WHITE) {
    b = ShiftS(b);
    b |= ShiftE(b) | ShiftW(b);
    return w & ~fillS(b);
  } else {
    w = ShiftN(w);
    w |= ShiftE(w) | ShiftW(w);
    return b & ~fillN(w);
  }
}

void ParseFen(char* fen, Board* board) {
  int n = 0;
  char* _fen = fen;

  BitBoard whitePawns = 0ULL;
  BitBoard blackPawns = 0ULL;

  for (int8_t sq = 0; sq < 64; sq++) {
    char c = *fen;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      int pc = charToPiece[(int)*fen];
      int s = mirror(sq);

      switch (pc) {
      case WHITE_PAWN:
        setBit(whitePawns, s);
        break;
      case BLACK_PAWN:
        setBit(blackPawns, s);
        break;
      default:
        break;
      }

      if (c == 'k')
        board->bkingSq = s;
      else if (c == 'K')
        board->wkingSq = s;

      board->pieces[n].pc = pc;
      board->pieces[n].sq = s;

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

  BitBoard whitePassers = passers(whitePawns, blackPawns, WHITE);
  BitBoard blackPassers = passers(whitePawns, blackPawns, BLACK);

  int p = 0;
  while (whitePassers) {
    int sq = lsb(whitePassers);
    board->passers[WHITE][p++] = sq;
    popLsb(whitePassers);
  }

  if (p < 8)
    board->passers[WHITE][p] = 0;

  p = 0;
  while (blackPassers) {
    int sq = lsb(blackPassers);
    board->passers[BLACK][p++] = sq;
    popLsb(blackPassers);
  }

  if (p < 8)
    board->passers[BLACK][p] = 0;
}