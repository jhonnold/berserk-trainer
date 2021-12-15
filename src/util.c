#ifdef WIN32
#include <windows.h>
#else
#include <stddef.h>
#include <sys/time.h>
#endif

#include "util.h"

#ifdef WIN32
long GetTimeMS() { return GetTickCount(); }
#else
long GetTimeMS() {
  struct timeval time;
  gettimeofday(&time, NULL);

  return time.tv_sec * 1000 + time.tv_usec / 1000;
}
#endif

void* AlignedMalloc(int size) {
  void* mem = malloc(size + 64 + sizeof(void*));
  void** ptr = (void**)((uintptr_t)(mem + 64 + sizeof(void*)) & ~(64 - 1));
  ptr[-1] = mem;
  return ptr;
}

void AlignedFree(void* ptr) { free(((void**)ptr)[-1]); }