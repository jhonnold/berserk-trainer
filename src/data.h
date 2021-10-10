#ifndef DATA_H
#define DATA_H

#include "board.h"

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