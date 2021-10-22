#ifndef BITS_H
#define BITS_H

#define bit(sq) (1ULL << (sq))
#define setBit(bb, sq) ((bb) |= bit(sq))
#define popLsb(bb) ((bb) &= (bb) - 1)
#define lsb(bb) (__builtin_ctzll(bb))

#define A_FILE 0x0101010101010101ULL
#define B_FILE 0x0202020202020202ULL
#define C_FILE 0x0404040404040404ULL
#define D_FILE 0x0808080808080808ULL
#define E_FILE 0x1010101010101010ULL
#define F_FILE 0x2020202020202020ULL
#define G_FILE 0x4040404040404040ULL
#define H_FILE 0x8080808080808080ULL

#define ShiftN(bb) ((bb) << 8)
#define ShiftS(bb) ((bb) >> 8)
#define ShiftNN(bb) ((bb) << 16)
#define ShiftSS(bb) ((bb) >> 16)
#define ShiftW(bb) (((bb) & ~A_FILE) << 1)
#define ShiftE(bb) (((bb) & ~H_FILE) >> 1)
#define ShiftNE(bb) (((bb) & ~H_FILE) << 7)
#define ShiftSW(bb) (((bb) & ~A_FILE) >> 7)
#define ShiftNW(bb) (((bb) & ~A_FILE) << 9)
#define ShiftSE(bb) (((bb) & ~H_FILE) >> 9)

#endif