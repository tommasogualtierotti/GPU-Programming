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

#ifndef SHA256_PARALLEL_CUH
    #include "include/sha256_parallel.cuh"
#endif

#ifndef SHA256_H
    #include "include/sha256.h"
#endif

#ifndef MD5_PARALLEL_CUH
    #include "include/md5_parallel.cuh"
#endif

#ifndef MD5_H
	#include "include/md5.h"
#endif

int main() {

    char **input_strings = NULL;
    size_t *data_length = NULL;

#if defined(PRINT_GPU_INFO)
    size_t free_memory, total_available_memory;
    get_gpu_memory_info(&free_memory, &total_available_memory);

    print_device_info();
#endif

#ifdef BATCH_PROCESSING
    size_t num_lines = line_count(FILENAME_PATH);
#else
    size_t num_lines = read_strings_from_file(FILENAME_PATH, &input_strings, &data_length);
#endif

#if defined(USE_STREAMS)
#if defined(SHA1_PARALLEL)
    uint32_t *sha1_hashes;
    allocate_cuda_host_memory((void**)&sha1_hashes, num_lines * SHA1_HASH_LENGTH * sizeof(sha1_hashes[0]));
#elif defined(MD5_PARALLEL)
    uint8_t *md5_hashes;
    allocate_cuda_host_memory((void **)&md5_hashes, num_lines * MD5_HASH_LENGTH_BYTES * sizeof(md5_hashes[0]));
#elif defined(SHA256_PARALLEL)
    uint8_t *sha256_hashes;
    allocate_cuda_host_memory((void **)&sha256_hashes, num_lines * SHA256_HASH_LENGTH_BYTES * sizeof(sha256_hashes[0]));
#endif
#else
#if defined(SHA1_PARALLEL)
    uint32_t *sha1_hashes = (uint32_t*) malloc (num_lines * SHA1_HASH_LENGTH * sizeof(sha1_hashes[0]));
    CHECK_NULL(sha1_hashes);
#elif defined(MD5_PARALLEL)
    uint8_t *md5_hashes = (uint8_t *) malloc (num_lines * MD5_HASH_LENGTH_BYTES * sizeof(md5_hashes[0]));
    CHECK_NULL(md5_hashes);
#elif defined(SHA256_PARALLEL)
    uint8_t *sha256_hashes = (uint8_t *) malloc (num_lines * SHA256_HASH_LENGTH_BYTES * sizeof(sha256_hashes[0]));
    CHECK_NULL(sha256_hashes);
#endif
#endif

#if defined(BATCH_PROCESSING)
    batch_reader_t *reader = batch_reader_init(FILENAME_PATH);
    CHECK_NULL(reader);
#if defined(SHA1_PARALLEL)
    parallel_sha1_batch_reading(sha1_hashes);
#elif defined(MD5_PARALLEL)
    parallel_md5_batch_reading(md5_hashes);
#elif defined(SHA256_PARALLEL)
    parallel_sha256_batch_reading(sha256_hashes);
#endif

#if defined(CPU_HASH_RUN)

#if defined(CPU_SHA1_RUN)
    size_t cpu_elapsed_time_SHA1 = elapsed_time_usec(0);
    uint8_t sha1_hash[20] = {0};
#elif defined(CPU_MD5_RUN)
    size_t cpu_elapsed_time_MD5 = elapsed_time_usec(0);
    uint8_t md5_hash[16] = {0};
    MD5_CTX_t ctx = {0};
#elif defined(CPU_SHA256_RUN)
    size_t cpu_elapsed_time_SHA256 = elapsed_time_usec(0);
    uint8_t sha256_hash[32] = {0};
#endif

    reader = batch_reader_init(FILENAME_PATH);
    CHECK_NULL(reader);

    do
    {
        batch_read_strings_from_file(reader, &input_strings, &data_length);

        for (size_t i = 0; i < reader->batch_read_items; i++)
        {

#if defined(CPU_SHA1_RUN)
            sha1(input_strings[i], data_length[i], (uint8_t *)sha1_hash);
#if defined(DEBUG_PRINT_HASHES)
            print_sha1_hashes_cpu(sha1_hash, &input_strings[i], 1);
#endif
#elif defined(CPU_MD5_RUN)
            md5(&ctx, (const uint8_t *)input_strings[i], data_length[i], (uint8_t *)md5_hash);
#if defined(DEBUG_PRINT_HASHES)
            print_md5_hashes_cpu(md5_hash, &input_strings[i], 1);
#endif
#elif defined(CPU_SHA256_RUN)
            sha256((const uint8_t *)input_strings[i], data_length[i], (uint8_t *)sha256_hash);
#if defined(DEBUG_PRINT_HASHES)
            print_sha256_hashes_cpu(sha256_hash, &input_strings[i], 1);
#endif
#endif
        }
    } while (reader->batch_read_items == BATCH_NUM_LINES);

#if defined(CPU_SHA1_RUN)
    cpu_elapsed_time_SHA1 = elapsed_time_usec(cpu_elapsed_time_SHA1);
    printf("Total execution of CPU SHA1: %.3fms\n", cpu_elapsed_time_SHA1 / (float)MSECPSEC);
#elif defined(CPU_MD5_RUN)
    cpu_elapsed_time_MD5 = elapsed_time_usec(cpu_elapsed_time_MD5);
    printf("Total execution of CPU MD5: %.3fms\n", cpu_elapsed_time_MD5 / (float)MSECPSEC);
#elif defined(CPU_SHA256_RUN)
    cpu_elapsed_time_SHA256 = elapsed_time_usec(cpu_elapsed_time_SHA256);
    printf("Total execution of CPU SHA256: %.3fms\n", cpu_elapsed_time_SHA256 / (float)MSECPSEC);
#endif

    batch_reader_close(reader);
#endif
#else

#if defined(SHA1_PARALLEL)
    parallel_sha1((const char **)input_strings, data_length, num_lines, sha1_hashes);
#elif defined(MD5_PARALLEL)
    parallel_md5((const char **)input_strings, data_length, num_lines, md5_hashes);
#elif defined(SHA256_PARALLEL)
    parallel_sha256((const char **)input_strings, data_length, num_lines, sha256_hashes);
#endif

#if defined(CPU_HASH_RUN)

#if defined(CPU_SHA1_RUN)
    size_t cpu_elapsed_time_SHA1 = elapsed_time_usec(0);
    uint8_t sha1_hash[20] = {0};
#elif defined(CPU_MD5_RUN)
    size_t cpu_elapsed_time_MD5 = elapsed_time_usec(0);
    uint8_t md5_hash[16] = {0};
    MD5_CTX_t ctx = {0};
#elif defined(CPU_SHA256_RUN)
    size_t cpu_elapsed_time_SHA256 = elapsed_time_usec(0);
    uint8_t sha256_hash[32] = {0};
#endif

    for (size_t i = 0; i < num_lines; i++)
    {
#if defined(CPU_SHA1_RUN)
        sha1(input_strings[i], data_length[i], (uint8_t *)sha1_hash);
#if defined(DEBUG_PRINT_HASHES)
        print_sha1_hashes_cpu(sha1_hash, &input_strings[i], 1);
#endif
#elif defined(CPU_MD5_RUN)
        md5(&ctx, (const uint8_t *)input_strings[i], data_length[i], (uint8_t *)md5_hash);
#if defined(DEBUG_PRINT_HASHES)
        print_md5_hashes_cpu(md5_hash, &input_strings[i], 1);
#endif
#elif defined(CPU_SHA256_RUN)
        sha256((const uint8_t *)input_strings[i], data_length[i], (uint8_t *)sha256_hash);
#ifdef DEBUG_PRINT_HASHES
        print_sha256_hashes_cpu(sha256_hash, &input_strings[i], 1);
#endif
#endif
    }
#if defined(CPU_SHA1_RUN)
    cpu_elapsed_time_SHA1 = elapsed_time_usec(cpu_elapsed_time_SHA1);
    printf("Total execution of CPU SHA1: %.3fms\n", cpu_elapsed_time_SHA1 / (float)MSECPSEC);
#elif defined(CPU_MD5_RUN)
    cpu_elapsed_time_MD5 = elapsed_time_usec(cpu_elapsed_time_MD5);
    printf("Total execution of CPU MD5: %.3fms\n", cpu_elapsed_time_MD5 / (float)MSECPSEC);
#elif defined(CPU_SHA256_RUN)
    cpu_elapsed_time_SHA256 = elapsed_time_usec(cpu_elapsed_time_SHA256);
    printf("Total execution of CPU SHA256: %.3fms\n", cpu_elapsed_time_SHA256 / (float)MSECPSEC);
#endif
#endif
#endif

#if defined(DEBUG_PRINT_HASHES) && !defined(BATCH_PROCESSING)
#if defined(SHA1_PARALLEL)
    print_sha1_hashes_gpu(sha1_hashes, input_strings, num_lines);
#elif defined(MD5_PARALLEL)
    print_md5_hashes_gpu(md5_hashes, input_strings, num_lines);
#elif defined(SHA256_PARALLEL)
    print_sha256_hashes_gpu(sha256_hashes, input_strings, num_lines);
#endif
#endif

#if defined(USE_STREAMS)
#if defined(SHA1_PARALLEL)
    free_cuda_host_memory((void*)sha1_hashes);
#elif defined(MD5_PARALLEL)
    free_cuda_host_memory((void*)md5_hashes);
#elif defined(SHA256_PARALLEL)
    free_cuda_host_memory((void*)sha256_hashes);
#endif
#else
#if defined(SHA1_PARALLEL)
    free(sha1_hashes);
#elif defined(MD5_PARALLEL)
    free(md5_hashes);
#elif defined(SHA256_PARALLEL)
    free(sha256_hashes);
#endif
#endif

    return 0;
}