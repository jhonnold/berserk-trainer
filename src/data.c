#include "data.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "board.h"
#include "random.h"
#include "util.h"

volatile int DATA_LOADED = 0;
volatile int COMPLETE = 0;

void WriteToFile(char* dest, char* src, uint64_t entries) {
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

  while (count < entries && fgets(line, 128, fp)) {
    LoadDataEntry(line, board);
    fwrite(board, sizeof(Board), 1, fout);

    count++;
    if (!(count % 10000000)) printf("\rWrote positions: [%10" PRId64 "]", count);
  }

  fclose(fp);
  fclose(fout);

  printf("\rWrote positions: [%10" PRId64 "]\n", count);
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
    printf("Failed to read %" PRId64 " files from %s with offset %" PRId64 " - %" PRId64 "\n", n, path, offset, x);
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

void* CyclicalLoader(void* args) {
  CyclicalLoadArgs* loader = (CyclicalLoadArgs*)args;

  size_t readsize = BATCH_SIZE * BATCHES_PER_LOAD;
  size_t location = 0;

  while (!COMPLETE) {
    // back to the start
    if (location + readsize > loader->entriesCount) {
      fseek(loader->fin, 0, SEEK_SET);
      location = 0;
    }

    size_t x;
    if ((x = fread(loader->nextData->entries, sizeof(Board), readsize, loader->fin)) != readsize)
      printf("Failed to read entries from file!\n"), exit(1);

    loader->nextData->n = readsize;
    location += readsize;

    ShuffleData(loader->nextData);

    DATA_LOADED = 1;
    while (DATA_LOADED)
      ;
  }

  return NULL;
}