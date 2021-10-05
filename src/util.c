#ifdef WIN32
#include <windows.h>
#else
#include <stddef.h>
#include <sys/time.h>
#endif
#include <math.h>

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

const float SS = 3.5f / 1024;

float Sigmoid(float s) { return 1.0f / (1.0f + expf(-s * SS)); }

float SigmoidPrime(float s) { return s * (1.0 - s) * SS; }
