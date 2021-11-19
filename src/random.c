#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "random.h"

// I dunno anything about random number generators, and had a bad one for a while
// Thanks to Martin Sedlák (author of Cheng) this one is really cool and works :)
// http://www.vlasak.biz/cheng/

uint64_t keys[2];

inline uint64_t rotate(uint64_t v, uint8_t s) { return (v >> s) | (v << (64 - s)); }

inline uint64_t RandomUInt64() {
  uint64_t tmp = keys[0];
  keys[0] += rotate(keys[1] ^ 0xc5462216u ^ ((uint64_t)0xcf14f4ebu << 32), 1);
  return keys[1] += rotate(tmp ^ 0x75ecfc58u ^ ((uint64_t)0x9576080cu << 32), 9);
}

void SeedRandom() {
  keys[0] = keys[1] = time(NULL);

  for (int i = 0; i < 64; i++)
    RandomUInt64();
}

// https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
double RandomGaussian(double mu, double sigma) {
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;

  if (call == 1) {
    call = !call;
    return (mu + sigma * (double)X2);
  }

  do {
    U1 = -1 + ((double)rand() / RAND_MAX) * 2;
    U2 = -1 + ((double)rand() / RAND_MAX) * 2;
    W = pow(U1, 2) + pow(U2, 2);
  } while (W >= 1 || W == 0);

  mult = sqrt((-2 * log(W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (double)X1);
}
