#ifndef SHA1_PARALLEL_CUH
    #include "../include/sha1_parallel.cuh"
#endif

/**
 * @brief SHA-1 compression function for a single message block.
 *
 * Implements the SHA-1 padding and main loop for one 64‑byte block.
 *
 * @param data Pointer to the input message bytes (up to 64 bytes).
 * @param len Length of the input message in bytes (≤56).
 * @param hash Output buffer (5 × uint32_t) to receive the SHA‑1 state.
 */
__device__ void sha1_device(const char *data, size_t len, uint32_t *hash) 
{
    uint32_t state[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    uint8_t padded_data[64];
    int i;

    for (i = 0; i < len; i++) 
    {
        padded_data[i] = data[i];
    }
    padded_data[len] = 0x80;
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
        w[i] = (padded_data[i * 4] << 24)
             | (padded_data[i * 4 + 1] << 16)
             | (padded_data[i * 4 + 2] << 8)
             | padded_data[i * 4 + 3];
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
        e = d; d = c; c = ROTLEFT(b, 30); b = a; a = temp;
    }

    state[0] += a; state[1] += b; state[2] += c;
    state[3] += d; state[4] += e;

    for (i = 0; i < 5; i++) 
    {
        hash[i] = state[i];
    }
}

/**
 * @brief CUDA kernel to compute SHA‑1 for multiple strings.
 *
 * Each thread handles one string of fixed MAX_STRING_LENGTH bytes.
 *
 * @param data Device buffer containing all strings concatenated.
 * @param hashes Device output buffer for 5‑word hashes.
 * @param lengths Device buffer of per‑string lengths.
 * @param num_lines Number of strings to process.
 */
__global__ void sha1_kernel(const char *data,
                            uint32_t *hashes,
                            size_t *lengths,
                            size_t num_lines) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_lines) {
        sha1_device(data + idx * MAX_STRING_LENGTH,
                    lengths[idx],
                    hashes + idx * SHA1_HASH_LENGTH);
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
 * @brief Read file in batches and compute SHA‑1 on the GPU.
 *
 * Uses batch_reader to read chunks of BATCH_NUM_LINES lines, then
 * transfers to device (optionally with streams), launches sha1_kernel,
 * and retrieves results into hashes[]. Measures and prints timing.
 *
 * @param hashes Host array to store all computed hashes.
 */
void parallel_sha1_batch_reading(uint32_t *hashes)
{
    char **input_strings = NULL;
    size_t *data_length = NULL;
    char *d_data; uint32_t *d_hashes; size_t *d_lengths;
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
    CHECK_CUDA_ERROR(cudaMalloc(&d_hashes, BATCH_NUM_LINES * SHA1_HASH_LENGTH * sizeof(hashes[0])));
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

        // CHECK_CUDA_ERROR(cudaMemset(d_data,   0, BATCH_SIZE));
        // CHECK_CUDA_ERROR(cudaMemset(d_lengths,0, BATCH_NUM_LINES * sizeof(size_t)));
        // CHECK_CUDA_ERROR(cudaMemset(d_hashes,0, BATCH_NUM_LINES * SHA1_HASH_LENGTH * sizeof(hashes[0])));

        for (size_t i = 0; i < active_chunks; i++)
        {
            size_t lines = base_lines + (i < extra ? 1 : 0);
            size_t bytes = lines * MAX_STRING_LENGTH;
            size_t s = i % NUM_STREAMS;

            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data   + off_data, h_data   + off_data, bytes, cudaMemcpyHostToDevice, streams[s]));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_lengths+ off_len,  data_length + off_len, lines * sizeof(size_t), cudaMemcpyHostToDevice, streams[s]));

            sha1_kernel<<<(lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, streams[s]>>>
                (d_data + off_data, d_hashes + off_hash, d_lengths + off_len, lines);
            CHECK_CUDA_ERROR(cudaGetLastError());

            CHECK_CUDA_ERROR(cudaMemcpyAsync(hashes + ((batch_iteration * BATCH_NUM_LINES + off_len) * SHA1_HASH_LENGTH),
                                             d_hashes + off_hash,
                                             lines * SHA1_HASH_LENGTH * sizeof(uint32_t),
                                             cudaMemcpyDeviceToHost,
                                             streams[s]));

            off_len  += lines;
            off_data += bytes;
            off_hash += lines * SHA1_HASH_LENGTH;
        }

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_stream_gpu, start, stop));
        total_time_stream_gpu += elapsed_time_stream_gpu;

#else
        total_size = num_lines * MAX_STRING_LENGTH;
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        // CHECK_CUDA_ERROR(cudaMemset(d_data,   0, BATCH_SIZE));
        // CHECK_CUDA_ERROR(cudaMemset(d_lengths,0, BATCH_NUM_LINES * sizeof(size_t)));
        // CHECK_CUDA_ERROR(cudaMemset(d_hashes,0, BATCH_NUM_LINES * SHA1_HASH_LENGTH * sizeof(hashes[0])));
        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(d_data,    h_data,   total_size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_lengths, data_length, num_lines * sizeof(size_t), cudaMemcpyHostToDevice));
        size_t blocks = (num_lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sha1_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, d_hashes, d_lengths, num_lines);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(hashes + (batch_iteration * BATCH_NUM_LINES * SHA1_HASH_LENGTH),
                                    d_hashes,
                                    num_lines * SHA1_HASH_LENGTH * sizeof(hashes[0]),
                                    cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_nostream_gpu, start, stop));
        total_time_nostream_gpu += elapsed_time_nostream_gpu;
#endif

#ifdef DEBUG_PRINT_HASHES
        for (size_t i = 0; i < num_lines; i++)
        {
            printf("String: %s\tHash: ", input_strings[i]);
            for (size_t j = 0; j < SHA1_HASH_LENGTH_BYTES / sizeof(hashes[0]); ++j) 
            {
                printf("%02x", hashes[(batch_iteration * BATCH_NUM_LINES + i) * (SHA1_HASH_LENGTH_BYTES / sizeof(hashes[0])) + j]);
            }
            printf("\n");
        }
#endif

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

/**
 * @brief Compute SHA-1 hashes for an array of strings.
 *
 * Allocates device buffers, copies data, launches sha1_kernel,
 * retrieves results, and measures execution time.
 *
 * @param data Host array of input strings.
 * @param lengths Host array of string lengths.
 * @param num_lines Number of strings to process.
 * @param hashes Host output buffer for 5-word SHA-1 hashes.
 */
void parallel_sha1(const char **data, const size_t *lengths, size_t num_lines, uint32_t *hashes)
{
    char *d_data;
    uint32_t *d_hashes;
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
    CHECK_CUDA_ERROR(cudaMalloc(&d_hashes,  num_lines * SHA1_HASH_LENGTH * sizeof(hashes[0])));
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

        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data + off_data, 
                                         h_data + off_data, 
                                         bytes, 
                                         cudaMemcpyHostToDevice, 
                                         streams[s]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_lengths + off_len, 
                                         lengths + off_len, 
                                         lines * sizeof(size_t), 
                                         cudaMemcpyHostToDevice, 
                                         streams[s]));

        size_t blocks = (lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sha1_kernel<<<blocks, THREADS_PER_BLOCK, 0, streams[s]>>>
                (d_data + off_data, 
                 d_hashes + off_len * SHA1_HASH_LENGTH, 
                 d_lengths + off_len, 
                 lines);
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaMemcpyAsync(hashes + off_len * SHA1_HASH_LENGTH,
                                         d_hashes + off_len * SHA1_HASH_LENGTH,
                                         lines * SHA1_HASH_LENGTH * sizeof(hashes[0]),
                                         cudaMemcpyDeviceToHost, 
                                         streams[s]));

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
    sha1_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, d_hashes, d_lengths, num_lines);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(hashes, d_hashes, num_lines * SHA1_HASH_LENGTH * sizeof(hashes[0]), cudaMemcpyDeviceToHost));
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
