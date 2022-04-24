#include "packer.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const int BATCH_SIZE = 16384;
const uint8_t WHITE = 0;
const uint8_t BLACK = 1;
const uint8_t KING_LOOKUP_FAILED = UINT8_MAX;
const float SIGMOID_SCALAR = 3.68415 / 512;

const uint8_t char_to_piece[] = {
    ['P'] = 0,   // White pawn
    ['N'] = 1,   // White knight
    ['B'] = 2,   // White bishop
    ['R'] = 3,   // White rook
    ['Q'] = 4,   // White queen
    ['K'] = 5,   // White king
    ['p'] = 6,   // Black pawn
    ['n'] = 7,   // Black knight
    ['b'] = 8,   // Black bishop
    ['r'] = 9,   // Black rook
    ['q'] = 10,  // Black queen
    ['k'] = 11,  // Black king
};

int main(int argc, char** argv) {
  int batches_per_file = 10000;
  uint64_t n_fens = batches_per_file * BATCH_SIZE;

  char in_filepath[128] = {0};
  char out_filepath[128] = {0};

  int c;
  while ((c = getopt(argc, argv, "hi:o:b:n:")) != -1) {
    switch (c) {
      case 'h':
        printf("\nSYNOPSIS");
        printf("\n\t./binpack -i <input_file>       (required)");
        printf("\n\t          -o <output_file>      (required)");
        printf("\n\t          -b [batches_per_file] (default 10000)");
        printf("\n\t          -n [number_of_fens]   (default 163840000)\n");
        printf("\nOPTIONS");
        printf("\n\t-i input_file");
        printf("\n\t\tInput file to read fens from.\n");
        printf("\n\t-o output_file");
        printf("\n\t\tOutput file to write binpacked fens to.");
        printf(" Will auto-generate a suffix of <outputfile>.#.binpack.\n");
        printf("\n\t-b batches_per_file");
        printf("\n\t\tThe number of batches to save in each file.\n");
        printf("\n\t-n number_of_fens");
        printf("\n\t\tThe number of fens to read from the input file.\n");
        exit(EXIT_SUCCESS);
      case 'i':
        strcpy(in_filepath, optarg);
        break;
      case 'o':
        strcpy(out_filepath, optarg);
        break;
      case 'b':
        batches_per_file = atoi(optarg);
        break;
      case 'n':
        n_fens = atoll(optarg);
        break;
      case '?':
        exit(EXIT_FAILURE);
    }
  }

  if (!(*in_filepath)) {
    printf("An input file is required!\n");
    exit(EXIT_FAILURE);
  } else if (!(*out_filepath)) {
    printf("An output file is required!\n");
    exit(EXIT_FAILURE);
  }

  FILE* fin = fopen(in_filepath, "r");
  if (fin == NULL) {
    printf("Unable to read from %s\n!", in_filepath);
    exit(EXIT_FAILURE);
  }

  char output_file[128];
  char line[128];
  Sample sample[1];

  int iterations = n_fens / (batches_per_file * BATCH_SIZE);
  for (int i = 1; i <= iterations; i++) {
    sprintf(output_file, "%s.%d.binpack", out_filepath, i);

    FILE* fout = fopen(output_file, "wb");
    if (fout == NULL) {
      printf("Unable to write to %s\n!", output_file);
      exit(EXIT_FAILURE);
    }

    printf("Writing to file %s\n", output_file);

    for (int n = 0; n < batches_per_file * BATCH_SIZE; n++) {
      fgets(line, 128, fin);

      parse_line(sample, line);
      fwrite(sample, sizeof(Sample), 1, fout);
    }

    fclose(fout);
    printf("Successfully wrote to file %s\n", output_file);
  }
}

void parse_line(Sample* dest, char* line) {
  // Reset
  dest->occupancies = 0;
  dest->kings[0] = dest->kings[1] = KING_LOOKUP_FAILED;
  memset(dest->pieces, 0, sizeof(uint8_t) * 16);

  // Read the FEN one char at a time
  char* fen = line;
  for (int sq = 0, piece_count = 0; sq < 64; sq++, fen++) {
    char c = *fen;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      // Cache king squares for faster lookup
      if (c == 'K')
        dest->kings[WHITE] = sq;
      else if (c == 'k')
        dest->kings[BLACK] = sq;

      // Store piece 4 bits at a time
      uint8_t piece = char_to_piece[(int)c];
      dest->occupancies |= (1ull << sq);
      dest->pieces[piece_count / 2] |= piece << (piece_count % 2 ? 4 : 0);
      piece_count++;
    } else if (c >= '1' && c <= '8')
      sq += (c - '1');
    else if (c == '/')
      sq--;
    else {
      printf("Unable to parse line (invalid FEN char): %s\n", line);
      exit(EXIT_FAILURE);
    }
  }

  if (dest->kings[WHITE] == KING_LOOKUP_FAILED || dest->kings[BLACK] == KING_LOOKUP_FAILED) {
    printf("Unable to parse line (kings not found): %s\n", line);
    exit(EXIT_FAILURE);
  }

  dest->stm = strstr(line, "w ") ? WHITE : BLACK;

  if (strstr(line, "[1.0]"))
    dest->wdl = 2;
  else if (strstr(line, "[0.5]"))
    dest->wdl = 1;
  else if (strstr(line, "[0.0]"))
    dest->wdl = 0;
  else {
    printf("Unable to parse line (unknown result): %s\n", line);
    exit(EXIT_FAILURE);
  }

  // int score = atoi(strstr(line, "] ") + 2);
  // dest->eval = sigmoid(score);

  if (dest->stm == BLACK) {
    dest->wdl = 2 - dest->wdl;
    // dest->eval = 1 - dest->eval;
  }
}

float sigmoid(int score) { return 1.0 / (1.0 + expf(-SIGMOID_SCALAR * score)); }