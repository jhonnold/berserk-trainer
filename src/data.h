#ifndef DATA_H
#define DATA_H

#include "types.h"

void LoadEntries(char* path, DataSet* data);
void LoadDataEntry(char* buffer, DataEntry* result);
void ShuffleData(DataSet* data);

#endif