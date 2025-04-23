#ifndef CONFIG_H
    #include "include/config.h"
#endif

#ifndef UTILITIES_H
    #include "include/utilities.h"
#endif

#ifndef CUDA_UTILITIES_CUH
    #include "include/cuda_utilities.cuh"
#endif

#ifndef SHA1_PARALLEL_CUH
    #include "include/sha1_parallel.cuh"
#endif

#ifndef SHA1_H
    #include "include/sha1.h"
#endif

#ifndef SHA256_H
    #include "include/sha256.h"
#endif

int main() {

    char **input_strings = NULL;
    size_t *data_length = NULL;

#ifdef PRINT_GPU_INFO
    size_t free_memory, total_available_memory;
    get_gpu_memory_info(&free_memory, &total_available_memory);

    print_device_info();
#endif

#ifdef BATCH_PROCESSING
    size_t num_lines = line_count(FILENAME_PATH);
#else
    size_t num_lines = read_strings_from_file(FILENAME_PATH, &input_strings, &data_length);
#endif

#ifdef USE_STREAMS
    uint32_t *sha1_hashes;
    allocate_cuda_host_memory((void**)&sha1_hashes, num_lines * SHA1_HASH_LENGTH * sizeof(sha1_hashes[0]));
#else
    uint32_t *sha1_hashes = (uint32_t*) malloc (num_lines * SHA1_HASH_LENGTH * sizeof(sha1_hashes[0]));
    CHECK_NULL(sha1_hashes);
#endif

#ifdef BATCH_PROCESSING
    batch_reader_t *reader = batch_reader_init(FILENAME_PATH);
    CHECK_NULL(reader);

    parallel_sha1_batch_reading(sha1_hashes); // handle in an another way the stop condition in the batch reading

#ifdef CPU_SHA1_RUN
    size_t cpu_elapsed_time = elapsed_time_usec(0);

    reader = batch_reader_init(FILENAME_PATH);
    CHECK_NULL(reader);

    do
    {
        batch_read_strings_from_file(reader, &input_strings, &data_length);

        uint8_t sha1_hash[20] = {0};

        for (size_t i = 0; i < reader->batch_read_items; i++)
        {
            // memset(sha1_hash, 0x0, SHA1_HASH_LENGTH * sizeof(sha1_hash[0]));

            sha1(input_strings[i], data_length[i], sha1_hash);
    
#ifdef DEBUG_PRINT_HASHES
            print_sha1_hashes_cpu(sha1_hash, &input_strings[i], 1);
#endif
        }
    } while (reader->batch_read_items == BATCH_NUM_LINES);

    batch_reader_close(reader);

    cpu_elapsed_time = elapsed_time_usec(cpu_elapsed_time);
    printf("Total execution of CPU SHA1: %.3fms\n", cpu_elapsed_time / (float)MSECPSEC);
#endif
#else

    parallel_sha1((const char **)input_strings, data_length, num_lines, sha1_hashes);

#ifdef CPU_SHA1_RUN
    size_t cpu_elapsed_time = elapsed_time_usec(0);

    uint8_t sha1_hash[20] = {0};

    for (size_t i = 0; i < num_lines; i++)
    {
        // memset(sha1_hash, 0x0, SHA1_HASH_LENGTH * sizeof(sha1_hash[0]));

        sha1(input_strings[i], data_length[i], (uint8_t *)sha1_hash);

#ifdef DEBUG_PRINT_HASHES
        print_sha1_hashes_cpu(sha1_hash, &input_strings[i], 1);
#endif
    }
    cpu_elapsed_time = elapsed_time_usec(cpu_elapsed_time);
    printf("Total execution of CPU SHA1: %.3fms\n", cpu_elapsed_time / (float)MSECPSEC);
#endif
#endif

#ifdef DEBUG_PRINT_HASHES
    print_sha1_hashes_gpu(sha1_hashes, input_strings, num_lines);
#endif

#ifdef USE_STREAMS
    free_cuda_host_memory((void*)sha1_hashes);
#else
    free(sha1_hashes);
#endif

    return 0;
}