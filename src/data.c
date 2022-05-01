#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "board.h"
#include "random.h"
#include "util.h"

void WriteToFile(char* dest, char* src) {
  FILE* fp = fopen(src, "r");
  if (fp == NULL) {
    printf("Cannot open file: %s!\n", src);
    exit(1);
  }

  FILE* fout = fopen(dest, "wb");
  if (fout == NULL) {
    printf("Cannot open file: %s!\n", dest);
    exit(1);
  }

  uint64_t count = 0;
  char line[128];

  Board board[1];

  while (fgets(line, 128, fp)) {
    LoadDataEntry(line, board);
    fwrite(board, sizeof(Board), 1, fout);

    count++;

    if (count % 1000000 == 0) printf("Wrote positions: [%10ld]\n", count);
  }

  fclose(fp);
  fclose(fout);

  printf("Wrote positions: [%10ld]\n", count);
}

void LoadEntriesBinary(char* path, DataSet* data, uint64_t n, uint64_t offset) {
  FILE* fp = fopen(path, "rb");
  if (fp == NULL) {
    printf("Cannot open file: %s!\n", path);
    exit(1);
  }

  if (data->entries == NULL) data->entries = malloc(sizeof(Board) * n);

  fseek(fp, sizeof(Board) * offset, SEEK_SET);
  
  size_t x;
  if ((x = fread(data->entries, sizeof(Board), n, fp)) != n) {
    printf("Failed to read %ld files from %s with offset %ld - %ld\n", n, path, offset, x);
    exit(1);
  }

  fclose(fp);

  data->n = n;
}

void LoadEntries(char* path, DataSet* data, uint32_t n, uint32_t offset) {
  FILE* fp = fopen(path, "r");
  if (fp == NULL) {
    printf("Cannot open file: %s!\n", path);
    exit(1);
  }

  data->n = 0;
  if (data->entries == NULL) data->entries = malloc(sizeof(Board) * n);

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

  int eval = atoi(strstr(buffer, "] ") + 2);
  result->eval = Sigmoid(eval);

  // Invert for black to move
  if (result->stm == BLACK) {
    result->wdl = 2 - result->wdl;
    result->eval = 1.0 - result->eval;
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