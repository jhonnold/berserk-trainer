#ifndef DATA_H
#define DATA_H

#include "types.h"

void LoadBinpack(DataSet* dest, char* path, uint32_t n);
void LoadEntries(char* path, DataSet* data, uint32_t n, uint32_t offset);
void LoadDataEntry(char* buffer, Board* result);
void ShuffleData(DataSet* data);

#endif