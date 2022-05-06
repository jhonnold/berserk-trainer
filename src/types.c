#include "types.h"

float ALPHA = 0.01f;

const float SS = 3.68415f / 512;

const Piece CHAR_TO_PIECE[] = {
    ['P'] = WHITE_PAWN,    //
    ['N'] = WHITE_KNIGHT,  //
    ['B'] = WHITE_BISHOP,  //
    ['R'] = WHITE_ROOK,    //
    ['Q'] = WHITE_QUEEN,   //
    ['K'] = WHITE_KING,    //
    ['p'] = BLACK_PAWN,    //
    ['n'] = BLACK_KNIGHT,  //
    ['b'] = BLACK_BISHOP,  //
    ['r'] = BLACK_ROOK,    //
    ['q'] = BLACK_QUEEN,   //
    ['k'] = BLACK_KING     //
};

const int8_t KING_BUCKETS[64] = {
    -1, -1, -1, -1, 31, 30, 29, 28,  //
    -1, -1, -1, -1, 27, 26, 25, 24,  //
    -1, -1, -1, -1, 23, 22, 21, 20,  //
    -1, -1, -1, -1, 19, 18, 17, 16,  //
    -1, -1, -1, -1, 15, 14, 13, 12,  //
    -1, -1, -1, -1, 11, 10, 9,  8,   //
    -1, -1, -1, -1, 7,  6,  5,  4,   //
    -1, -1, -1, -1, 3,  2,  1,  0    //
};
