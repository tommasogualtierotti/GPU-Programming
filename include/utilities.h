#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define USECPSEC 1000000ULL

#define MAX_STRING_LENGTH 30

size_t read_strings_from_file(const char *filename, char ***strings, size_t **lengths);

size_t elapsed_time_usec(size_t start);
size_t elapsed_time_msec(size_t start);