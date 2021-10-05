#ifndef DATA_H
#define DATA_H

#include "board.h"

typedef struct {
    float wdl, eval;
    Board board;
} DataEntry;

typedef struct {
    int n;
    DataEntry* entries;
} DataSet;

void LoadEntries(char* path, DataSet* data);
void LoadDataEntry(char* buffer, DataEntry* result);

#endif