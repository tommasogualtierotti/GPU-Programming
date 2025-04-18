#ifndef CUDA_UTILITIES_CUH
    #define CUDA_UTILITIES_CUH
#endif

#ifndef UTILITIES_H
    #include "utilities.h"
#endif

/**
 * @brief Left-rotate a 32-bit integer.
 *
 * This macro performs a circular left rotation of a 32-bit integer by a given number of bits.
 *
 * @param a The 32-bit integer to rotate.
 * @param b The number of bits to rotate.
 */
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))

/**
 * @brief Macro to check for CUDA errors.
 *
 * This macro wraps a CUDA API call and checks for errors. If an error occurs, it prints the
 * error message along with the file and line number, then exits the program.
 *
 * @param call The CUDA API function call to check.
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
 * @brief Retrieves GPU memory allocation information based on hashing workload.
 *
 * This function calculates the amount of free and total available GPU memory considering
 * the required allocations for a SHA-1 hashing workload.
 *
 * @param num_lines Number of input lines to process.
 * @param hash_size Size of the output hash storage.
 * @param lengths_size Size of the input lengths array.
 * @param free_mem Pointer to store the amount of free memory (in bytes).
 * @param total_avlbl_mem Pointer to store the total available GPU memory (in bytes).
 * @param shift_modifier Modifier to adjust measurement unit memory size displaying.
 */
void get_gpu_memory_allocation_info(size_t num_lines, size_t hash_size, size_t lengths_size, 
                                    size_t *free_mem, size_t *total_avlbl_mem, size_t shift_modifier);

/**
 * @brief Retrieves the current free and total available GPU memory.
 *
 * This function queries the GPU for its available and total memory, applying a shift modifier
 * if necessary.
 *
 * @param free_mem Pointer to store the amount of free memory (in bytes).
 * @param total_avlbl_mem Pointer to store the total available GPU memory (in bytes).
 * @param shift_modifier Modifier to adjust measurement unit memory size displaying.
 */
void get_gpu_memory_info(size_t *free_mem, size_t *total_avlbl_mem, size_t shift_modifier);

void print_execution_time(float gpu_time, size_t host_time);
// void print_execution_time(float nostream_gpu, float stream_gpu, size_t nostream_host, size_t stream_host, size_t divisor, size_t precision);

void print_device_info();

void allocate_cuda_host_memory(void **ptr, size_t alloc_size);

void free_cuda_host_memory(void *ptr);