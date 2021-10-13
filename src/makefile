CC = gcc
SRC = *.c
EXE = trainer

LIBS = -lm -lpthread
WFLAGS = -std=gnu17 -Wall -Wextra -Wshadow
CFLAGS = -O3 $(WFLAGS) -flto -march=native -mtune=native -DNDEBUG -g

all:
	$(CC) $(CFLAGS) $(SRC) $(LIBS) -o $(EXE)