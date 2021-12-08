#ifndef DATA_H
#define DATA_H

#include "types.h"

void LoadEntries(char* path, DataSet* data, int n, int offset);
void LoadDataEntry(char* buffer, Board* result);
void ShuffleData(DataSet* data);

#endif