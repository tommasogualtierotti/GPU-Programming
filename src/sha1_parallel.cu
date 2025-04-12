#include "../include/sha1_parallel.cuh"

__device__ void sha1_device(const char *data, size_t len, uint32_t *hash) 
{
    uint32_t state[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    uint8_t padded_data[64];
    int i;

    for (i = 0; i < len; i++) 
    {
        padded_data[i] = data[i];
    }
    padded_data[len] = 0x80; // Append the bit '1' to the message

    for (i = len + 1; i < 56; i++) 
    {
        padded_data[i] = 0;
    }

    uint64_t bit_len = len * 8;
    for (i = 0; i < 8; i++) 
    {
        padded_data[56 + i] = (bit_len >> (56 - 8 * i)) & 0xFF;
    }

    uint32_t w[80];
    for (i = 0; i < 16; i++) 
    {
        w[i] = (padded_data[i * 4] << 24) | (padded_data[i * 4 + 1] << 16) | (padded_data[i * 4 + 2] << 8) | padded_data[i * 4 + 3];
    }
    for (i = 16; i < 80; i++) 
    {
        w[i] = ROTLEFT(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4];
    uint32_t f, k, temp;

    for (i = 0; i < 80; i++) 
    {
        if (i < 20) 
        {
            f = (b & c) | ((~b) & d);
            k = 0x5A827999;
        } 
        else if (i < 40) 
        {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;
        } 
        else if (i < 60) 
        {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;
        } 
        else 
        {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;
        }

        temp = ROTLEFT(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = ROTLEFT(b, 30);
        b = a;
        a = temp;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;

    for (i = 0; i < 5; i++) 
    {
        hash[i] = state[i];
    }
}

__global__ void sha1_kernel(const char *data, uint32_t *hashes, size_t *lengths, size_t num_lines) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_lines) {
        sha1_device(data + idx * MAX_STRING_LENGTH, lengths[idx], hashes + idx * SHA1_HASH_LENGTH);
    }
}

