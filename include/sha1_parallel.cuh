#ifndef SHA1_PARALLEL_CUH
#define SHA1_PARALLEL_CUH

#ifndef UTILITIES_H
    #include "utilities.h"
#endif

#ifndef CUDA_UTILITIES_CUH
    #include "cuda_utilities.cuh"
#endif

#ifndef CONFIG_H
    #include "config.h"
#endif

/**
 * @brief Compute SHA-1 hashes for multiple strings on the GPU.
 *
 * Launches a CUDA kernel to process each input string in parallel, producing
 * a 160-bit (5 × 32-bit) SHA-1 hash for each.
 *
 * @param data Array of pointers to input strings.
 * @param lengths Array of lengths for each input string.
 * @param num_lines Number of input strings to hash.
 * @param hashes Output buffer for SHA-1 hashes; must have space for num_lines × 5 uint32_t entries.
 */
void parallel_sha1(const char **data, const size_t *lengths, size_t num_lines, uint32_t *hashes);

/**
 * @brief Read strings in batches and compute their SHA-1 hashes on the GPU.
 *
 * Uses batch_reader to read the input file in chunks of BATCH_NUM_LINES,
 * transfers each batch to the GPU (optionally using streams), computes
 * SHA-1 hashes, and stores results in the provided hashes buffer.
 *
 * @param hashes Output buffer for all computed SHA-1 hashes; must be large enough to hold
 *               total_file_lines × 5 uint32_t entries.
 */
void parallel_sha1_batch_reading(uint32_t *hashes);

#endif