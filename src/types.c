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
    -1, -1, -1, -1, 1, 1, 1, 1,  //
    -1, -1, -1, -1, 1, 1, 1, 1,  //
    -1, -1, -1, -1, 1, 1, 1, 1,  //
    -1, -1, -1, -1, 1, 1, 1, 1,  //
    -1, -1, -1, -1, 1, 1, 1, 1,  //
    -1, -1, -1, -1, 1, 1, 1, 1,  //
    -1, -1, -1, -1, 0, 0, 0, 0,  //
    -1, -1, -1, -1, 0, 0, 0, 0   //
};
