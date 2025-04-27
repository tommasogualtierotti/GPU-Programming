#ifndef MD5_PARALLEL_CUH
#include "../include/md5_parallel.cuh"
#endif

/**
 * @brief MD5 constants derived from the sine function.
 *
 * These constants are the integer part of the sines of integers (in radians) 
 * multiplied by 2^32. They are used in the MD5 algorithm's_device main transformation.
 */
__constant__ uint32_t K_device[64] = {
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
 * The `s_device` array specifies the number of left bit rotations applied to
 * different parts of the MD5 hashing process. Each round uses a different
 * set of shifts to ensure diffusion of input bits.
 */
__constant__ uint32_t s_device[] = {
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
__device__ void md5_init_device(MD5_CTX_t *ctx) {
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
__device__ void md5_transform_device(MD5_CTX_t *ctx, const uint8_t data[]) {
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
        b = b + LEFTROTATE((a + f + K_device[i] + m[g]), s_device[i]);
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
__device__ void md5_update_device(MD5_CTX_t *ctx, const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        ctx->bitlen += 8;
        if (ctx->datalen == 64) {
            md5_transform_device(ctx, ctx->data);
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
__device__ void md5_final_device(MD5_CTX_t *ctx, uint8_t hash[]) {
    size_t i;

    ctx->data[ctx->datalen++] = 0x80;
    if (ctx->datalen > 56) {
        while (ctx->datalen < 64)
            ctx->data[ctx->datalen++] = 0x00;
        md5_transform_device(ctx, ctx->data);
        ctx->datalen = 0;
    }
    while (ctx->datalen < 56)
        ctx->data[ctx->datalen++] = 0x00;

    for (i = 0; i < 8; i++)
        ctx->data[56 + i] = (ctx->bitlen >> (i * 8)) & 0xFF;
    md5_transform_device(ctx, ctx->data);

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
__device__ void md5_device ( MD5_CTX_t *ctx, const uint8_t* message, uint64_t msg_length, uint8_t* hash )
{
    md5_init_device ( ctx );
    md5_update_device ( ctx, (const uint8_t*)message, msg_length );
    md5_final_device ( ctx, hash );
}

/**
 * @brief CUDA kernel to compute MD5 for multiple strings.
 *
 * Each thread handles one string of fixed MAX_STRING_LENGTH bytes.
 *
 * @param data Device buffer containing all strings concatenated.
 * @param hashes Device output buffer for 16-byte hashes.
 * @param lengths Device buffer of per‑string lengths.
 * @param num_lines Number of strings to process.
 */
__global__ void md5_kernel(const char *data,
                            uint8_t *hashes,
                            size_t *lengths,
                            size_t num_lines) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_lines) {
        MD5_CTX_t ctx = {0};
        md5_device(&ctx, (const uint8_t *)data + idx * MAX_STRING_LENGTH,
                    lengths[idx],
                    hashes + idx * MD5_HASH_LENGTH_BYTES);
    }
}

#ifdef USE_STREAMS
/**
 * @brief Create CUDA streams for asynchronous work.
 *
 * @param streams Array of NUM_STREAMS cudaStream_t handles.
 */
static void cuda_streams_create(cudaStream_t *streams)
{
    for (size_t i = 0; i < NUM_STREAMS; i++) 
    {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }
}

/**
 * @brief Destroy CUDA streams after use.
 *
 * Synchronizes each stream, then destroys it.
 *
 * @param streams Array of NUM_STREAMS cudaStream_t handles.
 */
static void cuda_streams_destroy(cudaStream_t *streams)
{
    for (size_t i = 0; i < NUM_STREAMS; i++) 
    {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
}
#endif

/**
 * @brief Compute MD5 hashes for multiple strings on the GPU.
 *
 * Launches a CUDA kernel to process each input string in parallel, producing
 * a 128-bit (16 × 8-bit) MD5 hash for each.
 *
 * @param data Array of pointers to input strings.
 * @param lengths Array of lengths for each input string.
 * @param num_lines Number of input strings to hash.
 * @param hashes Output buffer for MD5 hashes; must have space for num_lines × 16-bytes entries.
 */
void parallel_md5(const char **data, const size_t *lengths, size_t num_lines, uint8_t *hashes)
{
    char *d_data;
    uint8_t *d_hashes;
    size_t *d_lengths;
    cudaEvent_t start, stop;
#ifdef USE_STREAMS
    static float elapsed_time_stream_gpu = 0, total_time_stream_gpu = 0;
#else
    static float elapsed_time_nostream_gpu = 0, total_time_nostream_gpu = 0;
#endif
    size_t total_size = num_lines * MAX_STRING_LENGTH;
    size_t total_host_time = elapsed_time_usec(0);
#ifdef USE_STREAMS
    cudaStream_t streams[NUM_STREAMS];
#endif

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaMalloc(&d_data,    total_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hashes,  num_lines * MD5_HASH_LENGTH_BYTES * sizeof(hashes[0])));
    CHECK_CUDA_ERROR(cudaMalloc(&d_lengths, num_lines * sizeof(size_t)));

#ifdef USE_STREAMS
    char *h_data;
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_data, total_size, cudaHostAllocDefault));
#else
    char *h_data = (char*)malloc(total_size);
#endif
    CHECK_NULL(h_data);
    memset(h_data, 0, total_size);
    for (size_t i = 0; i < num_lines; i++) 
    {
        memcpy(h_data + i * MAX_STRING_LENGTH, data[i], lengths[i]);
    }

#ifdef USE_STREAMS
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    cuda_streams_create(streams);

    size_t base = num_lines / CHUNKS, rem = num_lines % CHUNKS;
    size_t off_len = 0, off_data = 0;

    for (size_t i = 0; i < CHUNKS; i++) 
    {
        size_t lines = base + (i < rem ? 1 : 0);
        size_t bytes = lines * MAX_STRING_LENGTH;
        int s = i % NUM_STREAMS;

        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data + off_data, h_data + off_data, bytes, cudaMemcpyHostToDevice, streams[s]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_lengths + off_len, lengths + off_len, lines * sizeof(size_t), cudaMemcpyHostToDevice, streams[s]));

        size_t blocks = (lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        md5_kernel<<<blocks, THREADS_PER_BLOCK, 0, streams[s]>>>
            (d_data + off_data, d_hashes + off_len * MD5_HASH_LENGTH_BYTES, d_lengths + off_len, lines);
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaMemcpyAsync(hashes + off_len * MD5_HASH_LENGTH_BYTES, d_hashes + off_len * MD5_HASH_LENGTH_BYTES,
                                         lines * MD5_HASH_LENGTH_BYTES * sizeof(hashes[0]),
                                         cudaMemcpyDeviceToHost, streams[s]));

        off_len  += lines;
        off_data += bytes;
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cuda_streams_destroy(streams);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_stream_gpu, start, stop));
    total_time_stream_gpu += elapsed_time_stream_gpu;
    
#else
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_lengths, lengths, num_lines * sizeof(size_t), cudaMemcpyHostToDevice));

    size_t blocks = (num_lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    md5_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, d_hashes, d_lengths, num_lines);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(hashes, d_hashes, num_lines * MD5_HASH_LENGTH_BYTES * sizeof(hashes[0]), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_nostream_gpu, start, stop));
    total_time_nostream_gpu += elapsed_time_nostream_gpu;
#endif

    total_host_time = elapsed_time_usec(total_host_time);
#ifdef USE_STREAMS
    print_execution_time(total_time_stream_gpu, total_host_time);
    cudaFreeHost(h_data);
#else
    print_execution_time(total_time_nostream_gpu, total_host_time);
    free(h_data);
#endif

    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_hashes));
    CHECK_CUDA_ERROR(cudaFree(d_lengths));
}

/**
 * @brief Read strings in batches and compute their MD5 hashes on the GPU.
 *
 * Uses batch_reader to read the input file in chunks of BATCH_NUM_LINES,
 * transfers each batch to the GPU (optionally using streams), computes
 * MD5 hashes, and stores results in the provided hashes buffer.
 *
 * @param hashes Output buffer for all computed MD5 hashes; must be large enough to hold
 *               total_file_lines × 16 uint8_t entries.
 */
void parallel_md5_batch_reading(uint8_t *hashes)
{
    char **input_strings = NULL;
    size_t *data_length = NULL;
    char *d_data; uint8_t *d_hashes; size_t *d_lengths;
    cudaEvent_t start, stop;
    size_t total_host_time = 0;
#ifdef USE_STREAMS
    static float elapsed_time_stream_gpu = 0, total_time_stream_gpu = 0;
    size_t active_chunks = 0;
    cudaStream_t streams[NUM_STREAMS] = {0};
#else
    static float elapsed_time_nostream_gpu = 0, total_time_nostream_gpu = 0;
    size_t total_size = 0;
#endif
    size_t num_lines = 0, batch_iteration = 0;

    total_host_time = elapsed_time_usec(0);
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    batch_reader_t *reader = batch_reader_init(FILENAME_PATH);
    CHECK_NULL(reader);

#ifdef USE_STREAMS
    char *h_data; 
    CHECK_CUDA_ERROR(cudaHostAlloc((void**)&h_data, BATCH_SIZE, cudaHostAllocDefault));
    cuda_streams_create(streams);
#else
    char *h_data = (char*)malloc(BATCH_SIZE);
#endif
    CHECK_NULL(h_data);
    CHECK_CUDA_ERROR(cudaMalloc(&d_data,   BATCH_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hashes, BATCH_NUM_LINES * MD5_HASH_LENGTH_BYTES * sizeof(hashes[0])));
    CHECK_CUDA_ERROR(cudaMalloc(&d_lengths,BATCH_NUM_LINES * sizeof(size_t)));

    do 
    {
        batch_read_strings_from_file(reader, &input_strings, &data_length);
        num_lines = reader->batch_read_items;
        if (num_lines == 0) break;

        memset(h_data, 0, BATCH_SIZE);
        for (size_t i = 0; i < num_lines; i++) 
        {
            memcpy(h_data + i * MAX_STRING_LENGTH,
                input_strings[i],
                data_length[i]);
        }

#ifdef USE_STREAMS
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        active_chunks = (num_lines < CHUNKS) ? num_lines : CHUNKS;
        size_t base_lines = num_lines / active_chunks, extra = num_lines % active_chunks;
        size_t off_len = 0, off_data = 0, off_hash = 0;

        CHECK_CUDA_ERROR(cudaMemset(d_data,   0, BATCH_SIZE));
        CHECK_CUDA_ERROR(cudaMemset(d_lengths,0, BATCH_NUM_LINES * sizeof(size_t)));
        CHECK_CUDA_ERROR(cudaMemset(d_hashes,0, BATCH_NUM_LINES * MD5_HASH_LENGTH_BYTES * sizeof(hashes[0])));

        for (size_t i = 0; i < active_chunks; i++)
        {
            size_t lines = base_lines + (i < extra ? 1 : 0);
            size_t bytes = lines * MAX_STRING_LENGTH;
            size_t s = i % NUM_STREAMS;

            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data   + off_data, h_data   + off_data, bytes, cudaMemcpyHostToDevice, streams[s]));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_lengths+ off_len,  data_length + off_len, lines * sizeof(size_t), cudaMemcpyHostToDevice, streams[s]));

            md5_kernel<<<(lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, streams[s]>>>
                (d_data + off_data, d_hashes + off_hash, d_lengths + off_len, lines);
            CHECK_CUDA_ERROR(cudaGetLastError());

            CHECK_CUDA_ERROR(cudaMemcpyAsync(hashes + ((batch_iteration * BATCH_NUM_LINES + off_len) * MD5_HASH_LENGTH_BYTES),
                                            d_hashes + off_hash,
                                            lines * MD5_HASH_LENGTH_BYTES * sizeof(hashes[0]),
                                            cudaMemcpyDeviceToHost,
                                            streams[s]));

            off_len  += lines;
            off_data += bytes;
            off_hash += lines * MD5_HASH_LENGTH_BYTES;
        }

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_stream_gpu, start, stop));
        total_time_stream_gpu += elapsed_time_stream_gpu;

#else
        total_size = num_lines * MAX_STRING_LENGTH;
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        CHECK_CUDA_ERROR(cudaMemset(d_data,   0, BATCH_SIZE));
        CHECK_CUDA_ERROR(cudaMemset(d_lengths,0, BATCH_NUM_LINES * sizeof(size_t)));
        CHECK_CUDA_ERROR(cudaMemset(d_hashes,0, BATCH_NUM_LINES * MD5_HASH_LENGTH_BYTES * sizeof(hashes[0])));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(d_data,    h_data,   total_size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_lengths, data_length, num_lines * sizeof(size_t), cudaMemcpyHostToDevice));
        size_t blocks = (num_lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        md5_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, d_hashes, d_lengths, num_lines);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(hashes + (batch_iteration * BATCH_NUM_LINES * MD5_HASH_LENGTH_BYTES),
                                    d_hashes,
                                    num_lines * MD5_HASH_LENGTH_BYTES * sizeof(hashes[0]),
                                    cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_nostream_gpu, start, stop));
        total_time_nostream_gpu += elapsed_time_nostream_gpu;
#endif
 
        // for (size_t i = 0; i < num_lines; i++)
        // {
        //     printf("String: %s\tHash: ", input_strings[i]);
        //     for (size_t j = 0; j < MD5_HASH_LENGTH_BYTES; ++j) 
        //     {
        //         printf("%02x", hashes[(batch_iteration * BATCH_NUM_LINES + i) * MD5_HASH_LENGTH_BYTES + j]);
        //     }
        //     printf("\n");
        // }

        batch_iteration++;
    } while (reader->batch_read_items == BATCH_NUM_LINES);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef USE_STREAMS
    cuda_streams_destroy(streams);
#endif
    batch_reader_close(reader);

    total_host_time = elapsed_time_usec(total_host_time);
#ifdef USE_STREAMS
    print_execution_time(total_time_stream_gpu, total_host_time);
#else
    print_execution_time(total_time_nostream_gpu, total_host_time);
#endif

#ifdef USE_STREAMS
    cudaFreeHost(h_data);
#else
    free(h_data);
#endif

    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_hashes));
    CHECK_CUDA_ERROR(cudaFree(d_lengths));
}