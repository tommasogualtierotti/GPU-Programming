#ifndef MD5_H
    #include "../include/md5.h"
#endif

/**
 * @brief MD5 constants derived from the sine function.
 *
 * These constants are the integer part of the sines of integers (in radians) 
 * multiplied by 2^32. They are used in the MD5 algorithm's main transformation.
 */
static const uint32_t K[64] = {
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
static const uint32_t s[] = {
	7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
	5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
	4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
	6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

/**
 * @brief Initializes the MD5 context.
 * 
 * This function sets the initial values for the MD5 context, including the state variables and lengths.
 * 
 * @param ctx A pointer to the MD5 context to be initialized.
 */
void md5_init(MD5_CTX_t *ctx) {
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x67452301; // A
    ctx->state[1] = 0xefcdab89; // B
    ctx->state[2] = 0x98badcfe; // C
    ctx->state[3] = 0x10325476; // D
}

/**
 * @brief Performs one round of MD5 transformation on the context using the given 64-byte data block.
 * 
 * This function processes a 64-byte block of data and updates the MD5 state. It applies the MD5 transformation 
 * logic as specified in the MD5 algorithm.
 * 
 * @param ctx  A pointer to the MD5 context that holds the state to be updated.
 * @param data A 64-byte chunk of message data to be processed.
 */
void md5_transform(MD5_CTX_t *ctx, const uint8_t data[]) {
    uint32_t a, b, c, d, m[16], i, j;

    for (i = 0, j = 0; j < 64; i++, j += 4)
        m[i] = (data[j]) | (data[j+1] << 8) | (data[j+2] << 16) | (data[j+3] << 24);

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];

    for (i = 0; i < 64; i++) {
        uint32_t f, g;
        if (i < 16) {
            f = (b & c) | ((~b) & d);
            g = i;
        } else if (i < 32) {
            f = (d & b) | ((~d) & c);
            g = (5 * i + 1) % 16;
        } else if (i < 48) {
            f = b ^ c ^ d;
            g = (3 * i + 5) % 16;
        } else {
            f = c ^ (b | (~d));
            g = (7 * i) % 16;
        }
        uint32_t temp = d;
        d = c;
        c = b;
        b = b + LEFTROTATE((a + f + K[i] + m[g]), s[i]);
        a = temp;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
}

/**
 * @brief Updates the MD5 context with new data.
 * 
 * This function processes the input data, updates the context with each chunk of data, and performs the necessary 
 * transformations when a 64-byte block is filled.
 * 
 * @param ctx  A pointer to the MD5 context that holds the state.
 * @param data A pointer to the data to be hashed.
 * @param len  The length of the data to be processed.
 */
void md5_update(MD5_CTX_t *ctx, const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        ctx->bitlen += 8;
        if (ctx->datalen == 64) {
            md5_transform(ctx, ctx->data);
            // ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

/**
 * @brief Finalizes the MD5 computation and produces the hash.
 * 
 * This function applies padding and appends the length of the input message, then performs the final transformation 
 * to generate the hash value.
 * 
 * @param ctx  A pointer to the MD5 context containing the current state.
 * @param hash The output buffer to store the 128-bit (16-byte) hash value.
 */
void md5_final(MD5_CTX_t *ctx, uint8_t hash[]) {
    size_t i;

    ctx->data[ctx->datalen++] = 0x80;
    if (ctx->datalen > 56) {
        while (ctx->datalen < 64)
            ctx->data[ctx->datalen++] = 0x00;
        md5_transform(ctx, ctx->data);
        ctx->datalen = 0;
    }
    while (ctx->datalen < 56)
        ctx->data[ctx->datalen++] = 0x00;

    for (i = 0; i < 8; i++)
        ctx->data[56 + i] = (ctx->bitlen >> (i * 8)) & 0xFF;
    md5_transform(ctx, ctx->data);

    for (i = 0; i < 4; i++) {
        hash[i]      = (ctx->state[0] >> (i * 8)) & 0xFF;
        hash[i + 4]  = (ctx->state[1] >> (i * 8)) & 0xFF;
        hash[i + 8]  = (ctx->state[2] >> (i * 8)) & 0xFF;
        hash[i + 12] = (ctx->state[3] >> (i * 8)) & 0xFF;
    }
    
}


/**
 * @brief Computes the MD5 hash of a message.
 * 
 * This function initializes the MD5 context, processes the message in chunks, and finalizes the hash computation.
 * 
 * @param ctx        A pointer to the MD5 context.
 * @param message    A pointer to the message to be hashed.
 * @param msg_length The length of the message in bytes.
 * @param hash       The output buffer to store the MD5 hash value.
 */
void md5 ( MD5_CTX_t *ctx, const uint8_t* message, uint64_t msg_length, uint8_t* hash )
{
    md5_init ( ctx );
    md5_update ( ctx, (const uint8_t*)message, msg_length );
    md5_final ( ctx, hash );
}