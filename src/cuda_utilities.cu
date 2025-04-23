#ifndef CUDA_UTILITIES_CUH
    #include "../include/cuda_utilities.cuh"
#endif

/**
 * @brief Prints GPU memory allocation info for the SHA-1 batch reader.
 *
 * Queries free and total device memory, then prints:
 *   - Total GPU memory
 *   - Free GPU memory
 *   - Memory required for lines, lengths, and hashes
 *
 * @param num_lines Number of lines (strings) to process.
 * @param hash_size Size in bytes of the hash for each line.
 * @param lengths_size Size in bytes of the length array element.
 * @param free_mem Optional output for free device memory (bytes).
 * @param total_avlbl_mem Optional output for total device memory (bytes).
 */
void get_gpu_memory_allocation_info(size_t num_lines,
                                    size_t hash_size,
                                    size_t lengths_size,
                                    size_t *free_mem,
                                    size_t *total_avlbl_mem)
{
    size_t free_memory, total_available_memory, total_memory_allocated;

    cudaMemGetInfo(&free_memory, &total_available_memory);

    if (free_mem != NULL && total_avlbl_mem != NULL)
    {
        *free_mem = free_memory;
        *total_avlbl_mem = total_available_memory;
    }

    total_memory_allocated = num_lines * (MAX_STRING_LENGTH + hash_size + lengths_size);

    printf("Total memory GPU: %zu MB\nFree memory GPU: %zu MB\n",
           total_available_memory >> 20, free_memory >> 20);
    printf("Memory to be allocated on the GPU to store the lines: %zu MB\n",
           (num_lines * (size_t)MAX_STRING_LENGTH) >> 20);
    printf("Memory to be allocated on the GPU to store the lines' length: %zu MB\n",
           (num_lines * lengths_size) >> 20);
    printf("Memory to be allocated on the GPU to store the hashes: %zu MB\n",
           (num_lines * hash_size) >> 20);
    printf("Total memory to be allocated on the GPU: %zu MB\n",
           total_memory_allocated >> 20);
}

/**
 * @brief Prints basic GPU memory info.
 *
 * Queries and prints:
 *   - Total GPU memory
 *   - Free GPU memory
 *
 * @param free_mem Optional output for free device memory (bytes).
 * @param total_avlbl_mem Optional output for total device memory (bytes).
 */
void get_gpu_memory_info(size_t *free_mem, size_t *total_avlbl_mem)
{
    size_t free_memory, total_available_memory;

    cudaMemGetInfo(&free_memory, &total_available_memory);

    if (free_mem != NULL && total_avlbl_mem != NULL)
    {
        *free_mem = free_memory;
        *total_avlbl_mem = total_available_memory;
    }

    printf("Total memory GPU: %zu MB\nFree memory GPU: %zu MB\n",
           total_available_memory >> 20, free_memory >> 20);
}

/**
 * @brief Prints execution times for stream or non-stream GPU runs.
 *
 * Depending on whether USE_STREAMS is defined, prints:
 *   - GPU execution time
 *   - Total host-observed execution time
 *
 * @param gpu_time GPU time measured by CUDA events (ms).
 * @param host_time Host-side time measured by us (μs).
 */
void print_execution_time(float gpu_time, size_t host_time)
{
#ifdef USE_STREAMS
    printf("Time of GPU execution with streams execution: %.3f ms\n", gpu_time);
    printf("Total execution time seen from host with streams execution: %.3f ms\n",
           host_time / MSECPSEC_DIV);
#else
    printf("Time of GPU execution without streams execution: %.3f ms\n", gpu_time);
    printf("Total execution time seen from host without streams execution: %.3f ms\n",
           host_time / MSECPSEC_DIV);
#endif
}

/**
 * @brief Prints detailed device properties.
 *
 * Queries cudaGetDeviceProperties and prints information such as:
 *   - Device name and compute capability
 *   - Global, shared, and constant memory sizes
 *   - Warp size, max threads per block, grid/block dimensions
 *   - Multiprocessor count, clock rates, bus width, cache size, etc.
 */
void print_device_info() 
{
    cudaDeviceProp prop;
    
    cudaGetDeviceProperties(&prop, 0);

    printf("Device %d: %s\n", 0, prop.name);
    printf("\tCompute capability: %d.%d\n", prop.major, prop.minor);
    printf("\tTotal global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("\tTotal shared memory per block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("\tTotal constant memory: %.2f KB\n", prop.totalConstMem / 1024.0);
    printf("\tWarp size: %d\n", prop.warpSize);
    printf("\tMaximum threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("\tMaximum block dimensions: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("\tMaximum grid dimensions: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\tMaximum threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("\tNumber of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("\tCompute cores: %d\n", prop.multiProcessorCount * 64);
    printf("\tClock rate: %.2f GHz\n", prop.clockRate / 1.0e6);
    printf("\tMemory clock rate: %.2f GHz\n", prop.memoryClockRate / 1.0e6);
    printf("\tMemory bus width: %d bits\n", prop.memoryBusWidth);
    printf("\tL2 cache size: %.2f KB\n", prop.l2CacheSize / 1024.0);
    printf("\tTexture alignment: %zu bytes\n", prop.textureAlignment);
    printf("\tPCI bus ID: %d\n", prop.pciBusID);
    printf("\tPCI device ID: %d\n", prop.pciDeviceID);
    printf("\tPCI domain ID: %d\n", prop.pciDomainID);
    printf("\tUnified addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("\tCooperative groups: %s\n", prop.cooperativeLaunch ? "Supported" : "Not supported");
    printf("\tECC enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
}

/**
 * @brief Allocates pinned host memory for a single pointer.
 *
 * Wraps cudaHostAlloc, checks for errors, and verifies the pointer.
 *
 * @param ptr Pointer to the memory pointer to allocate.
 * @param alloc_size Number of bytes to allocate.
 */
void allocate_cuda_host_memory(void **ptr, size_t alloc_size)
{
    CHECK_CUDA_ERROR(cudaHostAlloc(ptr, alloc_size, cudaHostAllocDefault));
    CHECK_NULL(ptr);
}

/**
 * @brief Frees pinned host memory allocated with cudaHostAlloc.
 *
 * Wraps cudaFreeHost, checks for errors, and nullifies the pointer.
 *
 * @param ptr Pinned host memory pointer to free.
 */
void free_cuda_host_memory(void *ptr)
{
    if (ptr != NULL)
    {
        CHECK_CUDA_ERROR(cudaFreeHost(ptr));
        ptr = NULL;
    }
}

/**
 * @brief Allocates pinned host memory for a double pointer (e.g., char**).
 *
 * Allows allocation of an array of pointers in pinned memory.
 *
 * @param ptr Address of the double pointer to allocate.
 * @param alloc_size Number of bytes to allocate (typically count * sizeof(T*)).
 */
void allocate_cuda_host_memory_double_ptr(void ***ptr, size_t alloc_size)
{
    CHECK_NULL(ptr);
    CHECK_CUDA_ERROR(cudaHostAlloc((void**)ptr, alloc_size, cudaHostAllocDefault));
    CHECK_NULL(*ptr);
}

/**
 * @brief Frees pinned host memory allocated as a double pointer.
 *
 * Wraps cudaFreeHost and nullifies the allocated pointer.
 *
 * @param ptr The double pointer whose pointee should be freed.
 */
void free_cuda_host_memory_double_ptr(void **ptr)
{
    if (ptr != NULL && *ptr != NULL)
    {
        CHECK_CUDA_ERROR(cudaFreeHost(*ptr));
        *ptr = NULL;
    }
}
