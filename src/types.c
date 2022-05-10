#include "types.h"

int ITERATION = 0;
int LAST_SEEN[N_INPUT] = {0};

float ALPHA = 0.000875f;

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

const Piece OPPOSITE[12] = {
    BLACK_PAWN,    //
    BLACK_KNIGHT,  //
    BLACK_BISHOP,  //
    BLACK_ROOK,    //
    BLACK_QUEEN,   //
    BLACK_KING,    //
    WHITE_PAWN,    //
    WHITE_KNIGHT,  //
    WHITE_BISHOP,  //
    WHITE_ROOK,    //
    WHITE_QUEEN,   //
    WHITE_KING     //
};

const Square PSQT64_TO_32[64] = {
    0,  1,  2,  3,  3,  2,  1,  0,   //
    4,  5,  6,  7,  7,  6,  5,  4,   //
    8,  9,  10, 11, 11, 10, 9,  8,   //
    12, 13, 14, 15, 15, 14, 13, 12,  //
    16, 17, 18, 19, 19, 18, 17, 16,  //
    20, 21, 22, 23, 23, 22, 21, 20,  //
    24, 25, 26, 27, 27, 26, 25, 24,  //
    28, 29, 30, 31, 31, 30, 29, 28   //
};
