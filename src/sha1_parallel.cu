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

static void cuda_streams_create(cudaStream_t *streams)
{
    for (size_t i = 0; i < NUM_STREAMS; i++) 
    {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }
}

static void cuda_streams_destroy(cudaStream_t *streams)
{
    /* cudaStreamDestroy will implicitly synchronize and wait for all work to finish before destruction */
    /* It is MANDATORY to destroy streams after executing (when using batch execution) otherwise the GPU will run in memory run out*/
    /* As an alternative we can use the permanent streams approach, creating the streams just once and destroying them after the execution is finished */
    for (size_t i = 0; i < NUM_STREAMS; i++) 
    {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
}

void parallel_sha1_batch_reading(uint32_t *hashes)
{
    char **input_strings = NULL;
    size_t *data_length = NULL;

    char *d_data;
    uint32_t *d_hashes;
    size_t *d_lengths;

    cudaEvent_t start, stop;
    static float elapsed_time_nostream_gpu = 0, total_time_nostream_gpu = 0;
    static float elapsed_time_stream_gpu = 0, total_time_stream_gpu = 0;
    static size_t elapsed_time_nostream = 0, total_time_nostream = 0;
    static size_t elapsed_time_stream = 0, total_time_stream = 0;
    
    size_t total_size = 0;
    size_t chunk_size = 0;
    size_t chunk_lines = 0;
    size_t active_chunks = 0;
    size_t num_lines = 0;

    size_t batch_iteration = 0;
    
#ifdef USE_STREAMS
    cudaStream_t streams[NUM_STREAMS] = {0};
#endif

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    batch_reader_t *reader = batch_reader_init(FILENAME_PATH);
    CHECK_NULL(reader);

    char *h_data = (char*)malloc(BATCH_SIZE);
    CHECK_NULL(h_data);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, BATCH_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hashes, BATCH_NUM_LINES * SHA1_HASH_LENGTH * sizeof(hashes[0])));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_lengths, BATCH_NUM_LINES * sizeof(size_t)));

    // print_device_info();
    
#ifdef USE_STREAMS
    cuda_streams_create(streams);
#endif

    do 
    {
        batch_read_strings_from_file(reader, &input_strings, &data_length);

        num_lines = reader->batch_read_items;

        total_size = num_lines * MAX_STRING_LENGTH;

        if (num_lines == 0)
        {
            break;
        }

        memset(h_data, 0, BATCH_SIZE);

        for (size_t i = 0; i < num_lines; i++) 
        {
            memcpy(h_data + i * MAX_STRING_LENGTH, input_strings[i], data_length[i]);
        }

#ifdef USE_STREAMS

        elapsed_time_stream = elapsed_time_usec(0);

        CHECK_CUDA_ERROR(cudaEventRecord(start));

        active_chunks = (num_lines < CHUNKS) ? num_lines : CHUNKS;

        // chunk_lines = num_lines / active_chunks;

        // chunk_size = total_size / active_chunks;

        size_t base_lines_per_chunk = num_lines / active_chunks;
        size_t extra_lines = num_lines % active_chunks;
    
        size_t offset_lengths = 0;
        size_t offset_data = 0;
        size_t offset_hash = 0;
    
        CHECK_CUDA_ERROR(cudaMemset(d_data, 0x0, BATCH_SIZE));
        CHECK_CUDA_ERROR(cudaMemset(d_lengths, 0x0, BATCH_NUM_LINES * sizeof(size_t)));
        CHECK_CUDA_ERROR(cudaMemset(d_hashes, 0x0, BATCH_NUM_LINES * SHA1_HASH_LENGTH * sizeof(hashes[0])));
    
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
        for (size_t i = 0; i < active_chunks; i++)
        {
            size_t current_lines = base_lines_per_chunk + (i < extra_lines ? 1 : 0);
            size_t current_data = current_lines * MAX_STRING_LENGTH;
    
            size_t current_stream = i % NUM_STREAMS;
    
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                d_data + offset_data,
                h_data + offset_data,
                current_data,
                cudaMemcpyHostToDevice,
                streams[current_stream]
            ));
    
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                d_lengths + offset_lengths,
                data_length + offset_lengths,
                current_lines * sizeof(size_t),
                cudaMemcpyHostToDevice,
                streams[current_stream]
            ));
    
            sha1_kernel<<<(current_lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, streams[current_stream]>>>(
                d_data + offset_data,
                d_hashes + offset_hash,
                d_lengths + offset_lengths,
                current_lines
            );
            CHECK_CUDA_ERROR(cudaGetLastError()); // this call is used to check if an error arisen during kernel launch
    
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                hashes + ((batch_iteration * BATCH_NUM_LINES + offset_lengths) * SHA1_HASH_LENGTH),
                d_hashes + offset_hash,
                current_lines * SHA1_HASH_LENGTH * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                streams[current_stream]
            ));
    
            // update offsets
            offset_lengths += current_lines;
            offset_data += current_data;
            offset_hash += current_lines * SHA1_HASH_LENGTH;
        }

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_stream_gpu, start, stop));

        total_time_stream_gpu += elapsed_time_stream_gpu;

        elapsed_time_stream = elapsed_time_usec(elapsed_time_stream);
        total_time_stream += elapsed_time_stream;
    
        for (size_t i = 0; i < num_lines; i++)
        {
            printf("String: %s\tHash: ", input_strings[i]);
            for (size_t j = 0; j < SHA1_HASH_LENGTH; ++j) 
            {
                printf("%08x", hashes[(batch_iteration * BATCH_NUM_LINES + i) * SHA1_HASH_LENGTH + j]);
            }
            printf("\n");
        }

#else

        elapsed_time_nostream = elapsed_time_usec(0);

        CHECK_CUDA_ERROR(cudaEventRecord(start));

        CHECK_CUDA_ERROR(cudaMemset(d_data, 0x0, BATCH_SIZE));
        CHECK_CUDA_ERROR(cudaMemset(d_lengths, 0x0, BATCH_NUM_LINES * sizeof(size_t)));
        CHECK_CUDA_ERROR(cudaMemset(d_hashes, 0x0, BATCH_NUM_LINES * SHA1_HASH_LENGTH * sizeof(hashes[0])));

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_lengths, data_length, num_lines * sizeof(size_t), cudaMemcpyHostToDevice));

        size_t blocks_per_grid = (num_lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        sha1_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(d_data, d_hashes, d_lengths, num_lines);
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaMemcpy(hashes + (batch_iteration * BATCH_NUM_LINES * SHA1_HASH_LENGTH), d_hashes, num_lines * SHA1_HASH_LENGTH * sizeof(hashes[0]), cudaMemcpyDeviceToHost));

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_nostream_gpu, start, stop));

        total_time_nostream_gpu += elapsed_time_nostream_gpu;

        elapsed_time_nostream = elapsed_time_usec(elapsed_time_nostream);
        total_time_nostream += elapsed_time_nostream;

        for (size_t i = 0; i < num_lines; i++)
        {
            printf("String: %s\tHash: ", input_strings[i]);
            for (size_t j = 0; j < SHA1_HASH_LENGTH; ++j) 
            {
                printf("%08x", hashes[(batch_iteration * BATCH_NUM_LINES + i) * SHA1_HASH_LENGTH + j]);
            }
            printf("\n");
        }

#endif

        batch_iteration++;

    } while (reader->batch_read_items == BATCH_NUM_LINES);

#ifdef USE_STREAMS
    cuda_streams_destroy(streams);
#endif

    batch_reader_close(reader);

#ifdef USE_STREAMS
    print_execution_time(total_time_stream_gpu, total_time_stream);
#else
    print_execution_time(total_time_nostream_gpu, total_time_nostream);
#endif
    // print_execution_time(total_time_nostream_gpu, total_time_stream_gpu, total_time_nostream, total_time_stream, NULL, NULL);

    free(h_data);
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_hashes));
    CHECK_CUDA_ERROR(cudaFree(d_lengths));

    h_data = NULL;
    d_data = NULL;
    d_hashes = NULL;
    d_lengths = NULL;
}

void parallel_sha1(const char **data, const size_t *lengths, size_t num_lines, uint32_t *hashes)
{
    char *d_data;
    uint32_t *d_hashes;
    size_t *d_lengths;

    cudaEvent_t start, stop;
    static float elapsed_time_nostream_gpu = 0, total_time_nostream_gpu = 0;
    static float elapsed_time_stream_gpu = 0, total_time_stream_gpu = 0;

    static size_t elapsed_time_nostream = 0, total_time_nostream = 0;
    static size_t elapsed_time_stream = 0, total_time_stream = 0;

    size_t total_size = 0;
    size_t chunk_size = 0;

#ifdef USE_STREAMS
    cudaStream_t streams[NUM_STREAMS];
#endif

    // size_t free_memory, total_available_memory;
    // get_gpu_memory_allocation_info(num_lines, sizeof(hashes[0]), sizeof(lengths[0]), &free_memory, &total_available_memory, MiB_MEMORY_VALUE);
    // get_gpu_memory_allocation_info(num_lines, SHA1_HASH_LENGTH * sizeof(hashes[0]), sizeof(lengths[0]), NULL, NULL, MiB_MEMORY_VALUE);

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    total_size = num_lines * MAX_STRING_LENGTH; // Each line can be up to MAX_LINE_LENGTH bytes

    /*
     * Memory allocation on the GPU
     *     d_data       : allocates memory to host all the strings read from the file
     *     d_hashes     : allocates memory to host the hash of each string read from the file
     *     d_lengths    : allocates memory to host the length of each string read from the file
     */
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, total_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hashes, num_lines * SHA1_HASH_LENGTH * sizeof(hashes[0])));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_lengths, num_lines * sizeof(size_t)));

    /*
     * Memory allocation to create a unidimensional array which hosts the content of the array of strings (data)
     */
    char *h_data = (char*)malloc(total_size);
    memset(h_data, 0, total_size);

    for (size_t i = 0; i < num_lines; i++) {
        memcpy(h_data + i * MAX_STRING_LENGTH, data[i], lengths[i]);
    }

#ifdef USE_STREAMS

    elapsed_time_stream = elapsed_time_usec(0);
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    cuda_streams_create(streams);

    size_t chunk_line_size = num_lines / CHUNKS;
    size_t remainder = num_lines % CHUNKS;

    size_t offset_lengths = 0;
    size_t offset_data = 0;

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    for (size_t i = 0; i < CHUNKS; i++) 
    {
        size_t lines_this_chunk = chunk_line_size + (i < remainder ? 1 : 0);
        size_t bytes_this_chunk = lines_this_chunk * MAX_STRING_LENGTH;

        int stream_idx = i % NUM_STREAMS;

        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data + offset_data, h_data + offset_data, bytes_this_chunk, cudaMemcpyHostToDevice, streams[stream_idx]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_lengths + offset_lengths, lengths + offset_lengths, lines_this_chunk * sizeof(size_t), cudaMemcpyHostToDevice, streams[stream_idx]));

        size_t blocks = (lines_this_chunk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sha1_kernel<<<blocks, THREADS_PER_BLOCK, 0, streams[stream_idx]>>>(
            d_data + offset_data,
            d_hashes + offset_lengths * SHA1_HASH_LENGTH,
            d_lengths + offset_lengths,
            lines_this_chunk
        );
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaMemcpyAsync(
            hashes + offset_lengths * SHA1_HASH_LENGTH,
            d_hashes + offset_lengths * SHA1_HASH_LENGTH,
            lines_this_chunk * SHA1_HASH_LENGTH * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            streams[stream_idx]
        ));

        offset_lengths += lines_this_chunk;
        offset_data += bytes_this_chunk;
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    cuda_streams_destroy(streams);

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_stream_gpu, start, stop));

    total_time_stream_gpu += elapsed_time_stream_gpu;

    elapsed_time_stream = elapsed_time_usec(elapsed_time_stream);
    total_time_stream += elapsed_time_stream;

    // print_execution_time(total_time_nostream_gpu, total_time_stream_gpu, total_time_nostream, total_time_stream, NULL, NULL);

#else

    elapsed_time_nostream = elapsed_time_usec(0);

    CHECK_CUDA_ERROR(cudaEventRecord(start));

    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_lengths, lengths, num_lines * sizeof(size_t), cudaMemcpyHostToDevice)); /* All the length array is copied to the GPU */

    size_t blocks_per_grid = (num_lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sha1_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(d_data, d_hashes, d_lengths, num_lines);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(hashes, d_hashes, num_lines * SHA1_HASH_LENGTH * sizeof(hashes[0]), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_nostream_gpu, start, stop));

    total_time_nostream_gpu += elapsed_time_nostream_gpu;

    elapsed_time_nostream = elapsed_time_usec(elapsed_time_nostream);
    total_time_nostream += elapsed_time_nostream;

#endif

    printf("//////////////////////////////////////////////////\n");
#ifdef USE_STREAMS
    print_execution_time(total_time_stream_gpu, total_time_stream);
#else
    print_execution_time(total_time_nostream_gpu, total_time_nostream);
#endif

    // Free memory
    free(h_data);
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_hashes));
    CHECK_CUDA_ERROR(cudaFree(d_lengths));
}
