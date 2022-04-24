#ifndef PACKER_H
#define PACKER_H

#include <stdint.h>

typedef struct {
  uint8_t stm, wdl;
  uint8_t kings[2];
  uint64_t occupancies;
  uint8_t pieces[16];
} __attribute__((packed, aligned(4))) Sample;

void parse_line(Sample* dest, char* line);
float sigmoid(int score);

#endif