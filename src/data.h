#ifndef DATA_H
#define DATA_H

#include "board.h"

#define MAX_POSITIONS 1000000000

typedef struct {
    int8_t stm;
    float wdl, eval;
    Board board;
} DataEntry;

typedef struct {
    int n;
    DataEntry* entries;
} DataSet;

void LoadEntries(char* path, DataSet* data);
void LoadDataEntry(char* buffer, DataEntry* result);
void ShuffleData(DataSet* data);

#endif