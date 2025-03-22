#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>

#define USECPSEC 1000000ULL
#define MSECPSEC 1000ULL

#define MAX_STRING_LENGTH 30
#define BATCH_NUM_LINES 1000

#define BATCH_SIZE BATCH_NUM_LINES * MAX_STRING_LENGTH

typedef struct {
    FILE *file;
    size_t buffer_pos;
    size_t total_items;
    size_t batch_read_items;
} batch_reader_t;

ssize_t read_strings_from_file(const char *filename, char ***strings, size_t **lengths);

batch_reader_t* batch_reader_init(const char *filename);

ssize_t batch_read_strings_from_file(batch_reader_t *batch_reader, char ***strings, size_t **lengths);

void batch_reader_close(batch_reader_t *batch_reader);

size_t elapsed_time_usec(size_t start);