CC = gcc
SRC = src/*.c
EXE = trainer

LIBS = -lm -lpthread
DEFS = -DNDEBUG
WFLAGS = -std=gnu17 -Wall -Wextra -Wshadow
CFLAGS = -O3 $(WFLAGS) -flto -ffast-math -fopenmp -march=native -mtune=native -g

all:
	$(CC) $(CFLAGS) $(SRC) $(DEFS) $(LIBS) -o $(EXE)