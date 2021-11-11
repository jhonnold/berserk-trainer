#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "board.h"
#include "data.h"
#include "random.h"
#include "util.h"


void LoadEntries(char* path, DataSet* data, int n) {
  FILE* fp = fopen(path, "r");
  if (fp == NULL) {
    printf("Cannot open file: %s!\n", path);
    exit(1);
  }

  int p = 0;
  char in[128];
  while (p < n && fgets(in, 128, fp)) {
    LoadDataEntry(in, &data->entries[p++]);

    if (p % 16384 == 0)
      printf("\rLoaded positions: [%9d]", p);
  }

  data->n = p;
  printf("\rLoaded positions: [%9d]\n", p);
}

void LoadDataEntry(char* buffer, DataEntry* result) {
  result->board.stm = strstr(buffer, "w ") ? WHITE : BLACK;
  ParseFen(buffer, &result->board);

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

  int eval = atoi(strstr(buffer, "] ") + 2);
  result->eval = Sigmoid(eval);

  // Invert for black to move
  if (result->board.stm == BLACK) {
    result->wdl = 2 - result->wdl;
    result->eval = 1 - result->eval;
  }
}

void ShuffleData(DataSet* data) {
  DataEntry temp;

  for (int i = 0; i < data->n; i++) {
    int j = RandomUInt64() % data->n;
    temp = data->entries[i];
    data->entries[i] = data->entries[j];
    data->entries[j] = temp;
  }
}