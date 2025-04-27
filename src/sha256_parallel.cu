#ifndef SHA256_PARALLEL_CUH
    #include "../include/sha256_parallel.cuh"
#endif

/**
 * @brief SHA-256 round constants.
 *
 * These constants are derived from the fractional parts of the cube roots of 
 * the first 64 prime numbers and are used in the SHA-256 compression function.
 */
__constant__ uint32_t K[64] = {
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
__device__ void sha256_transform_device(uint32_t state[8], const uint8_t data[64]) {
    uint32_t W[64] = {0};
    uint32_t a, b, c, d, e, f, g, h, T1, T2;

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
__device__ void sha256_device(const uint8_t *message, size_t length, uint8_t hash[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint8_t buffer[64] = {0};
    size_t totalLength = length;
    size_t i;

    while (length >= 64) {
        sha256_transform_device(state, message);
        message += 64;
        length -= 64;
    }

    memcpy(buffer, message, length);
    buffer[length] = 0x80;
    if (length >= 56) {
        sha256_transform_device(state, buffer);
        memset(buffer, 0, 64);
    }
    uint64_t totalBits = totalLength * 8;
    for (i = 0; i < 8; ++i)
        buffer[63 - i] = (totalBits >> (i * 8)) & 0xff;
    sha256_transform_device(state, buffer);

    for (i = 0; i < 8; ++i) {
        hash[i * 4] = (state[i] >> 24) & 0xff;
        hash[i * 4 + 1] = (state[i] >> 16) & 0xff;
        hash[i * 4 + 2] = (state[i] >> 8) & 0xff;
        hash[i * 4 + 3] = state[i] & 0xff;
    }
}

/**
 * @brief CUDA kernel to compute SHA‑256 for multiple strings.
 *
 * Each thread handles one string of fixed MAX_STRING_LENGTH bytes.
 *
 * @param data Device buffer containing all strings concatenated.
 * @param hashes Device output buffer for 32-byte hashes.
 * @param lengths Device buffer of per‑string lengths.
 * @param num_lines Number of strings to process.
 */
__global__ void sha256_kernel(const char *data,
                            uint8_t *hashes,
                            size_t *lengths,
                            size_t num_lines) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_lines) {
        sha256_device((const uint8_t *)data + idx * MAX_STRING_LENGTH,
                    lengths[idx],
                    hashes + idx * SHA256_HASH_LENGTH_BYTES);
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
 * @brief Compute SHA-256 hashes for multiple strings on the GPU.
 *
 * Launches a CUDA kernel to process each input string in parallel, producing
 * a 256-bit (32 × 8-bit) SHA-256 hash for each.
 *
 * @param data Array of pointers to input strings.
 * @param lengths Array of lengths for each input string.
 * @param num_lines Number of input strings to hash.
 * @param hashes Output buffer for SHA-256 hashes; must have space for num_lines × 32-bytes entries.
 */
void parallel_sha256(const char **data, const size_t *lengths, size_t num_lines, uint8_t *hashes)
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
    CHECK_CUDA_ERROR(cudaMalloc(&d_hashes,  num_lines * SHA256_HASH_LENGTH_BYTES * sizeof(hashes[0])));
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
        sha256_kernel<<<blocks, THREADS_PER_BLOCK, 0, streams[s]>>>
            (d_data + off_data, d_hashes + off_len * SHA256_HASH_LENGTH_BYTES, d_lengths + off_len, lines);
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaMemcpyAsync(hashes + off_len * SHA256_HASH_LENGTH_BYTES, d_hashes + off_len * SHA256_HASH_LENGTH_BYTES,
                                         lines * SHA256_HASH_LENGTH_BYTES * sizeof(hashes[0]),
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
    sha256_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, d_hashes, d_lengths, num_lines);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(hashes, d_hashes, num_lines * SHA256_HASH_LENGTH_BYTES * sizeof(hashes[0]), cudaMemcpyDeviceToHost));
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
 * @brief Read strings in batches and compute their SHA-256 hashes on the GPU.
 *
 * Uses batch_reader to read the input file in chunks of BATCH_NUM_LINES,
 * transfers each batch to the GPU (optionally using streams), computes
 * SHA-256 hashes, and stores results in the provided hashes buffer.
 *
 * @param hashes Output buffer for all computed SHA-256 hashes; must be large enough to hold
 *               total_file_lines × 32-uint8_t entries.
 */
void parallel_sha256_batch_reading(uint8_t *hashes)
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
    CHECK_CUDA_ERROR(cudaMalloc(&d_hashes, BATCH_NUM_LINES * SHA256_HASH_LENGTH_BYTES * sizeof(hashes[0])));
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
        CHECK_CUDA_ERROR(cudaMemset(d_hashes,0, BATCH_NUM_LINES * SHA256_HASH_LENGTH_BYTES * sizeof(hashes[0])));

        for (size_t i = 0; i < active_chunks; i++)
        {
            size_t lines = base_lines + (i < extra ? 1 : 0);
            size_t bytes = lines * MAX_STRING_LENGTH;
            size_t s = i % NUM_STREAMS;

            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data   + off_data, h_data   + off_data, bytes, cudaMemcpyHostToDevice, streams[s]));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_lengths+ off_len,  data_length + off_len, lines * sizeof(size_t), cudaMemcpyHostToDevice, streams[s]));

            sha256_kernel<<<(lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, streams[s]>>>
                (d_data + off_data, d_hashes + off_hash, d_lengths + off_len, lines);
            CHECK_CUDA_ERROR(cudaGetLastError());

            CHECK_CUDA_ERROR(cudaMemcpyAsync(hashes + ((batch_iteration * BATCH_NUM_LINES + off_len) * SHA256_HASH_LENGTH_BYTES),
                                            d_hashes + off_hash,
                                            lines * SHA256_HASH_LENGTH_BYTES * sizeof(hashes[0]),
                                            cudaMemcpyDeviceToHost,
                                            streams[s]));

            off_len  += lines;
            off_data += bytes;
            off_hash += lines * SHA256_HASH_LENGTH_BYTES;
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
        CHECK_CUDA_ERROR(cudaMemset(d_hashes,0, BATCH_NUM_LINES * SHA256_HASH_LENGTH_BYTES * sizeof(hashes[0])));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(d_data,    h_data,   total_size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_lengths, data_length, num_lines * sizeof(size_t), cudaMemcpyHostToDevice));
        size_t blocks = (num_lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sha256_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, d_hashes, d_lengths, num_lines);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(hashes + (batch_iteration * BATCH_NUM_LINES * SHA256_HASH_LENGTH_BYTES),
                                    d_hashes,
                                    num_lines * SHA256_HASH_LENGTH_BYTES * sizeof(hashes[0]),
                                    cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_nostream_gpu, start, stop));
        total_time_nostream_gpu += elapsed_time_nostream_gpu;
#endif
 
        // for (size_t i = 0; i < num_lines; i++)
        // {
        //     printf("String: %s\tHash: ", input_strings[i]);
        //     for (size_t j = 0; j < SHA256_HASH_LENGTH_BYTES; ++j) 
        //     {
        //         printf("%02x", hashes[(batch_iteration * BATCH_NUM_LINES + i) * SHA256_HASH_LENGTH_BYTES + j]);
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