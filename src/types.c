#include "types.h"

const float SS = 3.68415f / 512;

const Piece charToPiece[] = {['P'] = WHITE_PAWN,   ['N'] = WHITE_KNIGHT, ['B'] = WHITE_BISHOP, ['R'] = WHITE_ROOK,
                             ['Q'] = WHITE_QUEEN,  ['K'] = WHITE_KING,   ['p'] = BLACK_PAWN,   ['n'] = BLACK_KNIGHT,
                             ['b'] = BLACK_BISHOP, ['r'] = BLACK_ROOK,   ['q'] = BLACK_QUEEN,  ['k'] = BLACK_KING};
const Piece opposite[] = {BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING, 0, 0,
                          WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING, 0, 0};

const int8_t scalar[] = {0, 1, 2, 3, 4, 5, -1, -1, 6, 7, 8, 9, 10, 11, -1, -1};