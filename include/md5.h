#ifndef MD5_H
	#define MD5_H
#endif

#ifndef MD5_COMMON_H
    #include "md5_common.h"
#endif

/**
 * @brief Computes the MD5 hash of a given message.
 *
 * This function processes the input message using the MD5 hashing algorithm
 * and stores the resulting 128-bit hash in the provided output buffer.
 *
 * @param ctx Pointer to the MD5 context structure.
 * @param message Pointer to the input message data.
 * @param msg_length Length of the input message in bytes.
 * @param hash Pointer to the output buffer where the computed 16-byte (128-bit) MD5 hash will be stored.
 */
void md5 ( MD5_CTX_t *ctx, const uint8_t* message, uint64_t msg_length, uint8_t* hash );