#ifndef MD5_COMMON_H
#define MD5_COMMON_H

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
* @struct MD5_CTX_t
* @brief Structure to store the context for MD5 hashing.
*
* This structure maintains the state of an MD5 hash computation, including:
* - A buffer (`data`) to store the current 512-bit chunk of input.
* - The current data length (`datalen`) in bytes.
* - The total length of input processed (`bitlen`) in bits.
* - The hash state (`state`), which consists of four 32-bit registers (A, B, C, D).
*/
typedef struct {
    uint8_t data[64];   /**< Buffer to store the current chunk of input data. */
    uint32_t datalen;   /**< Length of data currently stored in the buffer. */
    uint64_t bitlen;    /**< Total length of input processed so far. */
    uint32_t state[4];  /**< Hash state registers (A, B, C, D). */
 } MD5_CTX_t;

#endif