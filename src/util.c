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
