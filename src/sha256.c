#include "../include/sha256.h"

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
