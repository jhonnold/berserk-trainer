#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "board.h"
#include "random.h"
#include "util.h"

void LoadEntries(char* path, DataSet* data, uint32_t n, uint32_t offset) {
  FILE* fp = fopen(path, "r");
  if (fp == NULL) {
    printf("Cannot open file: %s!\n", path);
    exit(1);
  }

  char in[128];
  uint32_t p = 0;

  while (offset-- > 0) fgets(in, 128, fp);

  while (p < n && fgets(in, 128, fp)) {
    LoadDataEntry(in, &data->entries[p++]);

    if (p % 1000000 == 0) printf("\nLoaded positions: [%10d]", p);
  }

  data->n = p;
  printf("\nLoaded positions: [%10d]\n", p);
}

void LoadDataEntry(char* buffer, Board* result) {
  result->stm = strstr(buffer, "w ") ? WHITE : BLACK;
  ParseFen(buffer, result);

  if (strstr(buffer, "[1.0]"))
    result->wdl = 2;
  else if (strstr(buffer, "[0.5]"))
    result->wdl = 1;
  else if (strstr(buffer, "[0.0]"))
    result->wdl = 0;
  else {
    printf("Cannot parse entry: %s!\n", buffer);
    exit(1);
  }

  // Invert for black to move
  if (result->stm == BLACK) {
    result->wdl = 2 - result->wdl;
  }
}

void ShuffleData(DataSet* data) {
  Board temp;

  for (uint32_t i = 0; i < data->n; i++) {
    uint32_t j = RandomUInt64() % data->n;
    temp = data->entries[i];
    data->entries[i] = data->entries[j];
    data->entries[j] = temp;
  }
}