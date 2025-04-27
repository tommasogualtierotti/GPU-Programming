#ifndef SHA256_PARALLEL_CUH
#define SHA256_PARALLEL_CUH

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
* @brief Performs a right rotate (circular shift) on a 32-bit integer.
*
* @param x The 32-bit integer to be rotated.
* @param n The number of positions to rotate.
* @return The rotated 32-bit integer.
*/
#define ROTR(x, n) ((x >> n) | (x << (32 - n)))

/**
* @brief SHA-256 choice function.
*
* Selects bits from `y` or `z` based on the bits of `x`. 
* If a bit in `x` is 1, the corresponding bit from `y` is selected;
* otherwise, the bit from `z` is selected.
*
* @param x The control input.
* @param y The first choice.
* @param z The second choice.
* @return The result of the choice operation.
*/
#define CH(x, y, z) ((x & y) ^ (~x & z))

/**
* @brief SHA-256 majority function.
*
* Computes the majority function, which returns the value that occurs most
* frequently among the bits of `x`, `y`, and `z`.
*
* @param x First input.
* @param y Second input.
* @param z Third input.
* @return The majority result.
*/
#define MAJ(x, y, z) ((x & y) ^ (x & z) ^ (y & z))

/**
* @brief SHA-256 upper-case Sigma0 function.
*
* Applies a bitwise transformation consisting of right rotations.
*
* @param x Input word.
* @return The transformed word.
*/
#define SIGMA0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))

/**
* @brief SHA-256 upper-case Sigma1 function.
*
* Applies a bitwise transformation consisting of right rotations.
*
* @param x Input word.
* @return The transformed word.
*/
#define SIGMA1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))

/**
* @brief SHA-256 lower-case sigma0 function.
*
* Applies a bitwise transformation consisting of right rotations and shifts.
*
* @param x Input word.
* @return The transformed word.
*/
#define sigma0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))

/**
* @brief SHA-256 lower-case sigma1 function.
*
* Applies a bitwise transformation consisting of right rotations and shifts.
*
* @param x Input word.
* @return The transformed word.
*/
#define sigma1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))

/**
 * @brief Compute SHA-256 hashes for multiple strings on the GPU.
 *
 * Launches a CUDA kernel to process each input string in parallel, producing
 * a 256-bit (32 × 8-bit) SHA-256 hash for each.
 *
 * @param data Array of pointers to input strings.
 * @param lengths Array of lengths for each input string.
 * @param num_lines Number of input strings to hash.
 * @param hashes Output buffer for SHA-256 hashes; must have space for num_lines × 32-bytes entries.
 */
void parallel_sha256(const char **data, const size_t *lengths, size_t num_lines, uint8_t *hashes);

/**
 * @brief Read strings in batches and compute their SHA-256 hashes on the GPU.
 *
 * Uses batch_reader to read the input file in chunks of BATCH_NUM_LINES,
 * transfers each batch to the GPU (optionally using streams), computes
 * SHA-256 hashes, and stores results in the provided hashes buffer.
 *
 * @param hashes Output buffer for all computed SHA-256 hashes; must be large enough to hold
 *               total_file_lines × 32-uint8_t entries.
 */
void parallel_sha256_batch_reading(uint8_t *hashes);

#endif