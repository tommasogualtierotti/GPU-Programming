#ifndef MD5_PARALLEL_CUH
#define MD5_PARALLEL_CUH

#ifndef MD5_COMMON_H
    #include "md5_common.h"
#endif

/**
 * @brief Compute MD5 hashes for multiple strings on the GPU.
 *
 * Launches a CUDA kernel to process each input string in parallel, producing
 * a 128-bit (16 × 8-bit) MD5 hash for each.
 *
 * @param data Array of pointers to input strings.
 * @param lengths Array of lengths for each input string.
 * @param num_lines Number of input strings to hash.
 * @param hashes Output buffer for MD5 hashes; must have space for num_lines × 16 uint8_t entries.
 */
void parallel_md5(const char **data, const size_t *lengths, size_t num_lines, uint8_t *hashes);

/**
 * @brief Read strings in batches and compute their MD5 hashes on the GPU.
 *
 * Uses batch_reader to read the input file in chunks of BATCH_NUM_LINES,
 * transfers each batch to the GPU (optionally using streams), computes
 * MD5 hashes, and stores results in the provided hashes buffer.
 *
 * @param hashes Output buffer for all computed MD5 hashes; must be large enough to hold
 *               total_file_lines × 16 uint8_t entries.
 */
void parallel_md5_batch_reading(uint8_t *hashes);

#endif