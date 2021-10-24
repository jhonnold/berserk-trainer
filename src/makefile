CC = gcc
SRC = *.c
EXE = trainer

LIBS = -lm
DEFS = -DNDEBUG
WFLAGS = -std=gnu17 -Wall -Wextra -Wshadow -Wl,--stack,128000000
CFLAGS = -Ofast $(WFLAGS) -flto -ffast-math -fopenmp -march=native -mtune=native -g

all:
	$(CC) $(CFLAGS) $(SRC) $(DEFS) $(LIBS) -o $(EXE)