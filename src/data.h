#ifndef DATA_H
#define DATA_H

#include "types.h"

void WriteToFile(char* dest, char* src);
void LoadEntriesBinary(char* path, DataSet* data, uint64_t n, uint64_t offset);
void LoadEntries(char* path, DataSet* data, uint32_t n, uint32_t offset);
void LoadDataEntry(char* buffer, Board* result);
void ShuffleData(DataSet* data);

#endif