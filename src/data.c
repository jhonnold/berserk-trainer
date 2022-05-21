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
    if (!(count % 1000000)) printf("\rWrote positions: [%10" PRId64 "]", count);
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

static char* RandomString(char* str, size_t size) {
  const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  if (size) {
    --size;
    for (size_t n = 0; n < size; n++) {
      int key = RandomUInt64() % (int)(sizeof charset - 1);
      str[n] = charset[key];
    }
    str[size] = '\0';
  }
  return str;
}

void ShuffleBinpack(uint64_t n, char* in, char* out) {
  FILE* fin = fopen(in, "rb");
  if (fin == NULL) printf("Cannot read file %s\n!", in), exit(1);
  printf("Reading from binary packed file: %s\n", in);

  const size_t MAX_READ_SIZE = 16384 * 16384;
  Board* inMemoryBoards = malloc(sizeof(Board) * MAX_READ_SIZE);

  size_t nTempFiles = ceil((double) n / MAX_READ_SIZE);
  char** tempFileNames = malloc(sizeof(char*) * nTempFiles);
  size_t* tempFileBoardCounts = calloc(nTempFiles, sizeof(size_t));

  for (size_t f = 0; f < nTempFiles; f++) {
    // Build temp file name
    char buffer[128];
    tempFileNames[f] = malloc(sizeof(char) * 16);
    RandomString(tempFileNames[f], 16);
    sprintf(buffer, "E:/tmp/%s", tempFileNames[f]);

    FILE* fout = fopen(buffer, "wb");
    if (fout == NULL) printf("Failed to open temporary file #%lld -- %s!\n", f + 1, buffer), exit(1);

    // readsize has a max, otherwise grab as much as possible
    size_t readsize = n - f * MAX_READ_SIZE;
    if (readsize > MAX_READ_SIZE) readsize = MAX_READ_SIZE;

    size_t x;
    if ((x = fread(inMemoryBoards, sizeof(Board), readsize, fin)) != readsize) printf("Failed to read!\n"), exit(1);

    tempFileBoardCounts[f] = readsize;
    printf("Starting to write a total of %lld boards to temporary file #%lld: %s\n", tempFileBoardCounts[f], f + 1,
           buffer);

    // write to the temp file in a random order
    // grab a random idx, write to the file
    // then remove the old board from memory (overwrite with last element)
    for (size_t c = readsize; c > 0; c--) {
      uint64_t idx = RandomUInt64() % c;
      fwrite(&inMemoryBoards[idx], sizeof(Board), 1, fout);

      if (idx != c - 1) inMemoryBoards[idx] = inMemoryBoards[c - 1];

      if (!(c % 1000000)) printf("%10lld remaining writes\r", c);
    }
    printf("\n");

    fclose(fout);
  }

  // no longer need to store in memory
  free(inMemoryBoards);

  // Create file opens with all temp files
  FILE** tempFiles = malloc(sizeof(FILE*) * nTempFiles);
  for (size_t f = 0; f < nTempFiles; f++) {
    char buffer[128];
    sprintf(buffer, "E:/tmp/%s", tempFileNames[f]);

    tempFiles[f] = fopen(buffer, "rb");
    if (tempFiles[f] == NULL) printf("Failed to reopen temporary file #%lld -- %s\n", f + 1, buffer), exit(1);
  }

  // Create write out file
  FILE* fout = fopen(out, "wb");
  if (fout == NULL) printf("Unable to open output file: %s!\n", out), exit(1);
  printf("Starting to write a total of %lld boards to output file: %s\n", n, out);

  Board board[1];
  for (uint64_t i = 0; i < n; i++) {
    if (!((i + 1) % 1000000)) printf("Writing [%10lld of %10lld], Files Remaining %3lld\r", i + 1, n, nTempFiles);

    // Pick a random file and read from it then write out
    size_t fileIdx = nTempFiles > 1 ? RandomUInt64() % nTempFiles : 0;
    fread(board, sizeof(Board), 1, tempFiles[fileIdx]);
    fwrite(board, sizeof(Board), 1, fout);

    tempFileBoardCounts[fileIdx]--;

    // if no more boards remain in this file, then don't try and
    // read from it anymore
    if (tempFileBoardCounts[fileIdx] <= 0) {
      fclose(tempFiles[fileIdx]);
      
      char buffer[128];
      sprintf(buffer, "E:/tmp/%s", tempFileNames[fileIdx]);
      if (remove(buffer))
        printf("\nFailed to remove %s!\n", buffer);

      if (fileIdx != nTempFiles - 1) {
        tempFiles[fileIdx] = tempFiles[nTempFiles - 1];
        tempFileBoardCounts[fileIdx] = tempFileBoardCounts[nTempFiles - 1];
        tempFileNames[fileIdx] = tempFileNames[nTempFiles - 1];
      }

      nTempFiles--;
    }
  }
  printf("Writing [%10lld of %10lld], Files Remaining %3d\n", n, n, 0);

  fclose(fout);
  free(tempFiles);
  free(tempFileNames);
  free(tempFileBoardCounts);
}