#ifndef SHA256_H
#define SHA256_H

#ifndef UTILITIES_H
    #include "utilities.h"
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
 * @brief Computes the SHA-256 hash of a given message.
 *
 * This function processes an input message using the SHA-256 hashing algorithm
 * and produces a 256-bit (32-byte) hash output.
 *
 * @param message Pointer to the input message data.
 * @param length Length of the input message in bytes.
 * @param hash Pointer to the output buffer where the 32-byte hash will be stored.
 */
void sha256(const uint8_t *message, size_t length, uint8_t hash[32]);

#endif