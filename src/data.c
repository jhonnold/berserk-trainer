#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

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

    if (p % 16384 == 0)
      printf("\rLoaded positions: [%9d]", p);
  }

  data->n = p;
  printf("\rLoaded positions: [%9d]\n", p);
}

void LoadDataEntry(char* buffer, DataEntry* result) {
  result->stm = strstr(buffer, "w ") ? WHITE : BLACK;
  ParseFen(buffer, result->board, result->stm);

  if (strstr(buffer, "[1.0]"))
    result->wdl = result->stm == BLACK ? 0.0 : 1.0;
  else if (strstr(buffer, "[0.5]"))
    result->wdl = 0.5;
  else if (strstr(buffer, "[0.0]"))
    result->wdl = result->stm == BLACK ? 1.0 : 0.0;
  else {
    printf("Cannot parse entry: %s!\n", buffer);
    exit(1);
  }

  int eval = atoi(strstr(buffer, "] ") + 2);
  if (result->stm == BLACK)
    eval = -eval;

  result->eval = Sigmoid(eval);
}

void ShuffleData(DataSet* data) {
  srand(time(NULL));
  DataEntry temp;

  for (int i = 0; i < data->n; i++) {
    int j = rand() % data->n;
    temp = data->entries[i];
    data->entries[i] = data->entries[j];
    data->entries[j] = temp;
  }
}