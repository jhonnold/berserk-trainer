#ifndef BITS_H
#define BITS_H

#define bit(sq) (1ULL << (sq))
#define bits(bb) (__builtin_popcountll(bb))
#define setBit(bb, sq) ((bb) |= bit(sq))
#define popLsb(bb) ((bb) &= (bb) - 1)
#define lsb(bb) (__builtin_ctzll(bb))

#endif