#ifndef SHA1_H
    #include "../include/sha1.h"
#endif

/**
 * @brief Compute SHA-1 hash for a message.
 *
 * This function implements the SHA-1 algorithm: it pads the input message,
 * processes it in 64-byte chunks, and outputs a 20-byte (160-bit) hash.
 *
 * @param message Pointer to the input data.
 * @param length Length of the input data in bytes.
 * @param hash Output array of 20 bytes to receive the SHA-1 digest.
 */
void sha1(const char *message, size_t length, uint8_t hash[20]) {
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    uint64_t total_bits = length * 8;

    size_t padded_length = ((length + 9) + 63) & ~63;
    uint8_t *padded = (uint8_t *)malloc(padded_length);
    memcpy(padded, message, length);
    padded[length] = 0x80;
    memset(padded + length + 1, 0, padded_length - length - 9);
    for (int i = 0; i < 8; ++i)
        padded[padded_length - 8 + i] = (total_bits >> (56 - 8 * i)) & 0xFF;

    for (size_t chunk = 0; chunk < padded_length; chunk += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = (padded[chunk + i * 4] << 24) | (padded[chunk + i * 4 + 1] << 16) |
                   (padded[chunk + i * 4 + 2] << 8) | padded[chunk + i * 4 + 3];
        for (int t = 16; t < 80; ++t)
            w[t] = LEFTROTATE(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1);

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int t = 0; t < 80; ++t) {
            uint32_t f, k;
            if (t < 20) {
                f = (b & c) | (~b & d);
                k = 0x5A827999;
            } else if (t < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (t < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }
            uint32_t temp = LEFTROTATE(a, 5) + f + e + k + w[t];
            e = d;
            d = c;
            c = LEFTROTATE(b, 30);
            b = a;
            a = temp;
        }

        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }

    free(padded);

    for (int i = 0; i < 5; ++i) {
        hash[i * 4] = (h[i] >> 24) & 0xFF;
        hash[i * 4 + 1] = (h[i] >> 16) & 0xFF;
        hash[i * 4 + 2] = (h[i] >> 8) & 0xFF;
        hash[i * 4 + 3] = h[i] & 0xFF;
    }
}