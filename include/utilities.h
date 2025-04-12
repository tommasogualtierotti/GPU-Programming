#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>

#ifndef UTILITIES_H
    #define UTILITIES_H
#endif

#ifndef CONFIG_H
    #include "config.h"
#endif

#define USECPSEC 1000000ULL
#define MSECPSEC 1000ULL

#define USECPSEC_DIV (float)USECPSEC
#define MSECPSEC_DIV (float)MSECPSEC

#define SHA1_HASH_LENGTH 5ULL // the actual length is 20 bytes but hashes array is declared as uint32_t array

#define KiB_MEMORY_VALUE 10ULL
#define MiB_MEMORY_VALUE 20ULL
#define GiB_MEMORY_VALUE 30ULL

#define BATCH_SIZE BATCH_NUM_LINES * MAX_STRING_LENGTH

#define LEFTROTATE(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits))))

#define CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Error: NULL pointer detected at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

typedef struct {
    FILE *file;
    size_t buffer_pos;
    size_t total_items;
    size_t batch_read_items;
    size_t iteration;
} batch_reader_t;

size_t line_count(const char *filename);

size_t read_strings_from_file(const char *filename, char ***strings, size_t **lengths);

batch_reader_t* batch_reader_init(const char *filename);

size_t batch_read_strings_from_file(batch_reader_t *batch_reader, char ***strings, size_t **lengths);

void batch_reader_close(batch_reader_t *batch_reader);

void batch_reader_rewind_file(batch_reader_t* batch_reader);

size_t elapsed_time_usec(size_t start);

void print_sha1_hashes_gpu(uint32_t *hashes, char **strings, size_t num_lines);

void print_sha1_hashes_cpu(uint8_t *hashes, char **strings, size_t num_lines);