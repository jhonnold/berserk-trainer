CC = gcc
SRC = *.c
EXE = trainer

LIBS = -lm -lpthread
WFLAGS = -std=gnu11 -Wall -Wextra -Wshadow
CFLAGS = -O3 $(WFLAGS) -flto -fopenmp -ffast-math -march=native -DNDEBUG -g -Wl,--stack,268435456

POPCOUNT = -DPOPCOUNT -msse -msse3 -mpopcnt
AVX2 = $(POPCOUNT) -mavx2 -msse4.1 -mssse3 -msse2

all:
	$(CC) $(CFLAGS) $(SRC) $(LIBS) $(AVX2) -o $(EXE)