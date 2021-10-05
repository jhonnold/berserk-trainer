#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "board.h"
#include "data.h"
#include "util.h"

void LoadEntries(char* path, DataSet* data) {
  FILE* fp = fopen(path, "r");
  if (fp == NULL) {
    printf("Cannot open file: %s!\n", path);
    exit(1);
  }

  int p = 0;
  char in[128];
  while (fgets(in, 128, fp)) {
    LoadDataEntry(in, &data->entries[p++]);

    if (!(p & 4095))
      printf("\rLoaded positions: [%9d]", p);
  }

  data->n = p;
  printf("\rLoaded positions: [%9d]\n", p);
}

void LoadDataEntry(char* buffer, DataEntry* result) {
  ParseFen(buffer, result->board);

  if (strstr(buffer, "[1.0]"))
    result->wdl = 1.0;
  else if (strstr(buffer, "[0.5]"))
    result->wdl = 0.5;
  else if (strstr(buffer, "[0.0]"))
    result->wdl = 0.0;
  else {
    printf("Cannot parse entry: %s!\n", buffer);
    exit(1);
  }

  int eval = atoi(strstr(buffer, "] ") + 2);
  result->eval = Sigmoid(eval);
}