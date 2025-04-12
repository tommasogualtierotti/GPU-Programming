#ifndef MD5_H
	#define MD5_H
#endif

#ifndef UTILITIES_H
    #include "utilities.h"
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

/**
 * @brief MD5 constants derived from the sine function.
 *
 * These constants are the integer part of the sines of integers (in radians) 
 * multiplied by 2^32. They are used in the MD5 algorithm's main transformation.
 */
const uint32_t K[64] = {
	0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
	0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
	0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
	0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
	0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
	0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
	0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
	0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
	0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
	0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
	0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
	0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
	0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
	0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
	0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
	0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391 
};

/**
 * @brief MD5 per-round shift amounts.
 *
 * The `s` array specifies the number of left bit rotations applied to
 * different parts of the MD5 hashing process. Each round uses a different
 * set of shifts to ensure diffusion of input bits.
 */
const uint32_t s[] = {
	7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
	5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
	4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
	6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

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