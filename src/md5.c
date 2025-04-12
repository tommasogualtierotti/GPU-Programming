#include "../include/md5.h"

void md5_init(MD5_CTX_t *ctx) {
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x67452301; // A
    ctx->state[1] = 0xefcdab89; // B
    ctx->state[2] = 0x98badcfe; // C
    ctx->state[3] = 0x10325476; // D
}

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

void md5 ( MD5_CTX_t *ctx, const uint8_t* message, uint64_t msg_length, uint8_t* hash )
{
    md5_init ( ctx );
    md5_update ( ctx, (const uint8_t*)message, msg_length );
    md5_final ( ctx, hash );
}