#ifndef CUDA_UTILITIES_CUH
    #define CUDA_UTILITIES_CUH
#endif

#ifndef UTILITIES_H
    #include "utilities.h"
#endif

/**
 * @brief Left-rotate a 32-bit integer.
 *
 * Performs a circular left rotation of a 32-bit integer by a specified number of bits.
 *
 * @param a The 32-bit integer to rotate.
 * @param b The number of bits to rotate by (0–31).
 */
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))

/**
 * @brief Check and report CUDA API and kernel launch errors.
 *
 * Wraps a CUDA API call, checks its return value, and then checks for any
 * deferred kernel launch errors. On error, prints the failed call, file,
 * and line number, then exits.
 *
 * @param call The CUDA runtime API or kernel launch call to check.
 */
#define CHECK_CUDA_ERROR(call)                                                      \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error in call '%s' at %s:%d - %s\n",             \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err_));          \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
        cudaError_t launch_err_ = cudaGetLastError();                               \
        if (launch_err_ != cudaSuccess) {                                           \
            fprintf(stderr, "Kernel launch error after call '%s' at %s:%d - %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(launch_err_));   \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

/**
 * @brief Calculate and print GPU memory requirements for SHA-1 workload.
 *
 * Retrieves free and total GPU memory, then calculates and prints memory
 * needed for storing input lines, their lengths, and output hashes.
 *
 * @param num_lines Number of input strings to process.
 * @param hash_size Size in bytes of the hash storage per string.
 * @param lengths_size Size in bytes of the length array element.
 * @param free_mem Optional pointer to receive free device memory (bytes).
 * @param total_avlbl_mem Optional pointer to receive total device memory (bytes).
 */
void get_gpu_memory_allocation_info(size_t num_lines,
                                    size_t hash_size,
                                    size_t lengths_size,
                                    size_t *free_mem,
                                    size_t *total_avlbl_mem);

/**
 * @brief Print current free and total GPU memory.
 *
 * Queries the CUDA runtime for free and total device memory and prints the values.
 *
 * @param free_mem Optional pointer to receive free device memory (bytes).
 * @param total_avlbl_mem Optional pointer to receive total device memory (bytes).
 */
void get_gpu_memory_info(size_t *free_mem,
                         size_t *total_avlbl_mem);

/**
 * @brief Print measured GPU and host execution times.
 *
 * Depending on whether streams are used, prints GPU time and total host time.
 *
 * @param gpu_time GPU execution time measured by CUDA events (milliseconds).
 * @param host_time Host-side execution time measured in microseconds.
 */
void print_execution_time(float gpu_time, size_t host_time);

/**
 * @brief Print detailed properties of the current CUDA device.
 *
 * Retrieves and prints properties like compute capability, memory sizes,
 * multiprocessor count, clock rates, and more.
 */
void print_device_info();

/**
 * @brief Allocate pinned host memory for a single pointer.
 *
 * Uses cudaHostAlloc to allocate page-locked memory on the host.
 *
 * @param ptr Address of the host pointer to allocate.
 * @param alloc_size Number of bytes to allocate.
 */
void allocate_cuda_host_memory(void **ptr, size_t alloc_size);

/**
 * @brief Free pinned host memory allocated by cudaHostAlloc.
 *
 * Uses cudaFreeHost to release page-locked host memory.
 *
 * @param ptr Host pointer to free.
 */
void free_cuda_host_memory(void *ptr);

/**
 * @brief Allocate pinned host memory for an array of pointers.
 *
 * Allocates page-locked host memory sufficient to hold an array of pointers.
 *
 * @param ptr Address of the double pointer to allocate.
 * @param alloc_size Number of bytes to allocate (e.g., count * sizeof(void*)).
 */
void allocate_cuda_host_memory_double_ptr(void ***ptr, size_t alloc_size);

/**
 * @brief Free pinned host memory allocated for an array of pointers.
 *
 * Releases page-locked host memory previously allocated for a double pointer.
 *
 * @param ptr The double pointer whose memory should be freed.
 */
void free_cuda_host_memory_double_ptr(void **ptr);