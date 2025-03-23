#ifndef UTILITIES_H
    #include "utilities.h"
#endif

#ifndef CUDA_UTILITIES_H
    #define CUDA_UTILITIES_H
#endif

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void get_gpu_memory_allocation_info(size_t num_lines, size_t hash_size, size_t lengths_size, size_t *free_mem, size_t *total_avlbl_mem, size_t shift_modifier);

void get_gpu_memory_info(size_t *free_mem, size_t *total_avlbl_mem, size_t shift_modifier);
