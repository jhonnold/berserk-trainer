CC = gcc
SRC = *.c
EXE = binpack

LIBS = -lm
DEFS = -DNDEBUG
WFLAGS = -std=gnu17 -Wall -Wextra -Wshadow
CFLAGS = -O3 $(WFLAGS) -flto -march=native -g

all:
	$(CC) $(CFLAGS) $(SRC) $(DEFS) $(LIBS) -o $(EXE)