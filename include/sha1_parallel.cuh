#ifndef SHA1_PARALLEL_CUH
    #define SHA1_PARALLEL_CUH
#endif

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
 * @brief Computes SHA-1 hashes in parallel for multiple input strings using CUDA.
 *
 * This function processes multiple input strings using a parallelized SHA-1 implementation on the GPU.
 *
 * @param data Pointer to an array of input strings.
 * @param lengths Pointer to an array of lengths corresponding to each input string.
 * @param num_lines The number of input strings to process.
 * @param hashes Pointer to the output buffer where the computed 160-bit (5 x 32-bit) SHA-1 hashes will be stored.
 */
void parallel_sha1(const char **data, const size_t *lengths, size_t num_lines, uint32_t *hashes);

void parallel_sha1_batch_reading(uint32_t *hashes);