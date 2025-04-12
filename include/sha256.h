#ifndef SHA256_H
    #define SHA256_H
#endif

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
 * @brief SHA-256 round constants.
 *
 * These constants are derived from the fractional parts of the cube roots of 
 * the first 64 prime numbers and are used in the SHA-256 compression function.
 */
const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

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
