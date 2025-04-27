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

#ifndef CUDA_UTILITIES_CUH
    #include "cuda_utilities.cuh"
#endif

#define USECPSEC 1000000ULL
#define MSECPSEC 1000ULL

#define USECPSEC_DIV (float)USECPSEC
#define MSECPSEC_DIV (float)MSECPSEC

#define SHA1_HASH_LENGTH 5ULL /**< Number of 32‑bit words in a SHA‑1 hash (20 bytes) */
#define SHA1_HASH_LENGTH_BYTES 20ULL /**< Number of 32‑bit words in a SHA‑1 hash (20 bytes) */
#define MD5_HASH_LENGTH 4ULL /**< Number of 32‑bit words in a MD5 hash (16 bytes) */
#define MD5_HASH_LENGTH_BYTES 16ULL
#define SHA256_HASH_LENGTH_BYTES 32ULL /**< Number of bytes in a SHA-256 hash (32 bytes) */

#define BATCH_SIZE (BATCH_NUM_LINES * MAX_STRING_LENGTH) /**< Total bytes per batch */

#define LEFTROTATE(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits)))) /**< Circular left rotation */

/**
 * @brief Check for NULL pointer and exit on error.
 *
 * If ptr is NULL, prints an error message with file and line,
 * then exits the program.
 *
 * @param ptr Pointer to check.
 */
#define CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Error: NULL pointer detected at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * @brief Structure for maintaining batch‐reading state.
 *
 *   - file: pointer to the open FILE  
 *   - buffer_pos: how many lines have been read  
 *   - total_items: total lines in the file  
 *   - batch_read_items: lines read in the current batch  
 *   - iteration: number of batches read so far
 */
typedef struct {
    FILE *file;
    size_t buffer_pos;
    size_t total_items;
    size_t batch_read_items;
    size_t iteration;
} batch_reader_t;

/**
 * @brief Count the number of lines in a file.
 *
 * Opens the specified file, reads through each line to count them,
 * then closes the file.
 *
 * @param filename Path to the input file.
 * @return Number of lines in the file.
 */
size_t line_count(const char *filename);

/**
 * @brief Read all strings from a file into dynamically allocated arrays.
 *
 * Allocates arrays for the strings and their lengths, reads each line
 * into its own buffer, strips the newline, and records its length.
 *
 * @param filename Path to the input file.
 * @param strings Output pointer to array of string pointers.
 * @param lengths Output pointer to array of string lengths.
 * @return Number of strings read.
 */
size_t read_strings_from_file(const char *filename, char ***strings, size_t **lengths);

/**
 * @brief Initialize a batch reader for chunked file reading.
 *
 * Allocates and initializes a batch_reader_t structure, opens the file,
 * counts total lines, and rewinds the file.
 *
 * @param filename Path to the input file.
 * @return Pointer to an initialized batch_reader_t, or NULL on failure.
 */
batch_reader_t* batch_reader_init(const char *filename);

/**
 * @brief Read a batch of strings from a file using a batch reader.
 *
 * Reads up to BATCH_NUM_LINES lines into preallocated buffers,
 * zeroing or allocating as needed on each call.
 *
 * @param batch_reader Pointer to an initialized batch_reader_t.
 * @param strings In/out pointer to array of string buffers.
 * @param lengths In/out pointer to array of string lengths.
 * @return Number of strings read in this batch.
 */
size_t batch_read_strings_from_file(batch_reader_t *batch_reader, char ***strings, size_t **lengths);

/**
 * @brief Close a batch reader and free its resources.
 *
 * Closes the file and frees the batch_reader_t struct.
 *
 * @param batch_reader Pointer to the batch_reader_t to close.
 */
void batch_reader_close(batch_reader_t *batch_reader);

/**
 * @brief Rewind the batch reader's file to the beginning.
 *
 * @param batch_reader Pointer to the batch_reader_t to rewind.
 */
void batch_reader_rewind_file(batch_reader_t* batch_reader);

/**
 * @brief Compute elapsed time in microseconds.
 *
 * Uses gettimeofday() to compute the difference from the given start time.
 *
 * @param start Start time in microseconds.
 * @return Elapsed time in microseconds.
 */
size_t elapsed_time_usec(size_t start);

/**
 * @brief Print SHA‑1 hashes computed on the GPU.
 *
 * For each string and its hash, prints the string followed by its
 * 5‑word (20‑byte) hash in hexadecimal.
 *
 * @param hashes Array of uint32_t hashes (5 words per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_sha1_hashes_gpu(uint32_t *hashes, char **strings, size_t num_lines);

/**
 * @brief Print SHA‑1 hashes computed on the CPU.
 *
 * For each string and its hash, prints the string followed by its
 * 20‑byte hash in two‑digit hexadecimal.
 *
 * @param hashes Array of uint8_t hashes (20 bytes per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_sha1_hashes_cpu(uint8_t *hashes, char **strings, size_t num_lines);

/**
 * @brief Print MD5 hashes computed on the GPU.
 *
 * For each string and its hash, prints the string followed by its
 * 4‑word (16‑byte) hash in hexadecimal.
 *
 * @param hashes Array of uint8_t hashes (16 bytes per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_md5_hashes_gpu(uint8_t *hashes, char **strings, size_t num_lines);

/**
 * @brief Print MD5 hashes computed on the CPU.
 *
 * For each string and its hash, prints the string followed by its
 * 16‑byte hash in two‑digit hexadecimal.
 *
 * @param hashes Array of uint8_t hashes (16 bytes per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_md5_hashes_cpu(uint8_t *hashes, char **strings, size_t num_lines);

/**
 * @brief Print SHA-256 hashes computed on the GPU.
 *
 * For each string and its hash, prints the string followed by its
 * 8‑word (32‑byte) hash in hexadecimal.
 *
 * @param hashes Array of uint8_t hashes (32 bytes per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_sha256_hashes_gpu(uint8_t *hashes, char **strings, size_t num_lines);

/**
 * @brief Print SHA‑256 hashes computed on the CPU.
 *
 * For each string and its hash, prints the string followed by its
 * 32‑byte hash in two‑digit hexadecimal.
 *
 * @param hashes Array of uint8_t hashes (32 bytes per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_sha256_hashes_cpu(uint8_t *hashes, char **strings, size_t num_lines);