#ifndef SHA1_H
    #define SHA1_H
#endif

#ifndef UTILITIES_H
    #include "utilities.h"
#endif

#ifndef CONFIG_H
    #include "config.h"
#endif

/**
 * @brief Computes the SHA-1 hash of a given message.
 *
 * This function processes an input message using the SHA-1 hashing algorithm
 * and produces a 160-bit (20-byte) hash output.
 *
 * @param message Pointer to the input message data.
 * @param length Length of the input message in bytes.
 * @param hash Pointer to the output buffer where the 20-byte (160-bit) SHA-1 hash will be stored.
 */
void sha1(const char *message, size_t length, uint8_t hash[20]);