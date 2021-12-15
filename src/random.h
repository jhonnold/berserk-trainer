#ifndef RANDOM_H
#define RANDOM_H

#include <inttypes.h>

uint64_t rotate(uint64_t v, uint8_t s);
uint64_t RandomUInt64();
void SeedRandom();
float RandomGaussian(float mu, float sigma);

#endif