CC = gcc
SRC = *.c
EXE = testing

LIBS = -lm
DEFS = -DNDEBUG
WFLAGS = -std=gnu17 -Wall -Wextra -Wshadow
CFLAGS = -O3 $(WFLAGS) -flto -fopenmp -march=native -mtune=native -g

all:
	$(CC) $(CFLAGS) $(SRC) $(DEFS) $(LIBS) -o $(EXE)