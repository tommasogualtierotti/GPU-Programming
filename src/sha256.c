#ifndef SHA256_H
    #include "../include/sha256.h"
#endif

/**
 * @brief SHA-256 round constants.
 *
 * These constants are derived from the fractional parts of the cube roots of 
 * the first 64 prime numbers and are used in the SHA-256 compression function.
 */
static const uint32_t K[64] = {
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
 * @brief Performs one round of SHA-256 transformation on the state using the given 64-byte data block.
 * 
 * This function processes a 64-byte block of data and updates the SHA-256 state. It uses the constants and
 * functions defined in the SHA-256 specification to compute the hash transformation.
 * 
 * @param state The current state of the SHA-256 hash.
 * @param data  A 64-byte chunk of message data to be processed.
 */
void sha256_transform(uint32_t state[8], const uint8_t data[64]) {
    uint32_t W[64], a, b, c, d, e, f, g, h, T1, T2;

    for (int i = 0; i < 16; ++i)
        W[i] = (data[i * 4] << 24) | (data[i * 4 + 1] << 16) | (data[i * 4 + 2] << 8) | data[i * 4 + 3];
    for (int i = 16; i < 64; ++i)
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    for (int i = 0; i < 64; ++i) {
        T1 = h + SIGMA1(e) + CH(e, f, g) + K[i] + W[i];
        T2 = SIGMA0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}


/**
 * @brief Computes the SHA-256 hash of a message.
 * 
 * This function processes the input message in blocks, applies padding, and computes the SHA-256 hash. The
 * final hash value is stored in the provided hash output buffer.
 * 
 * @param message The input message to be hashed.
 * @param length  The length of the input message in bytes.
 * @param hash    The output buffer to store the 256-bit (32-byte) hash value.
 */
void sha256(const uint8_t *message, size_t length, uint8_t hash[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint8_t buffer[64] = {0};
    size_t totalLength = length;
    size_t i;

    while (length >= 64) {
        sha256_transform(state, message);
        message += 64;
        length -= 64;
    }

    memcpy(buffer, message, length);
    buffer[length] = 0x80;
    if (length >= 56) {
        sha256_transform(state, buffer);
        memset(buffer, 0, 64);
    }
    uint64_t totalBits = totalLength * 8;
    for (i = 0; i < 8; ++i)
        buffer[63 - i] = (totalBits >> (i * 8)) & 0xff;
    sha256_transform(state, buffer);

    for (i = 0; i < 8; ++i) {
        hash[i * 4] = (state[i] >> 24) & 0xff;
        hash[i * 4 + 1] = (state[i] >> 16) & 0xff;
        hash[i * 4 + 2] = (state[i] >> 8) & 0xff;
        hash[i * 4 + 3] = state[i] & 0xff;
    }
}
