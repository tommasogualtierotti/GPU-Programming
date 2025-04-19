#include "../include/cuda_utilities.cuh"

void get_gpu_memory_allocation_info(size_t num_lines, size_t hash_size, size_t lengths_size, size_t *free_mem, size_t *total_avlbl_mem, size_t shift_modifier)
{
    char unit[6] = {0};
    size_t free_memory, total_available_memory, total_memory_allocated;

    cudaMemGetInfo(&free_memory, &total_available_memory);

    if (free_mem != NULL && total_avlbl_mem != NULL)
    {
        *free_mem = free_memory;
        *total_avlbl_mem = total_available_memory;
    }

    total_memory_allocated = num_lines * (MAX_STRING_LENGTH + (hash_size) + lengths_size);

    switch (shift_modifier)
    {
        case 0:
            memcpy(unit, "Bytes", sizeof("Bytes"));
            break;
        case 10:
            memcpy(unit, "KiB", sizeof("KiB"));
            break;
        case 20:
            memcpy(unit, "MiB", sizeof("MiB"));
            break;
        case 30:
            memcpy(unit, "GiB", sizeof("GiB"));
            break;
        default:
            memcpy(unit, "Bytes", sizeof("Bytes"));
            break;
    }

    printf("Total memory GPU: %zu %s\nFree memory GPU: %zu %s\n", total_available_memory >> shift_modifier, unit, free_memory >> shift_modifier, unit);
    printf("Memory to be allocated on the GPU to store the lines: %zu %s\n", (num_lines * (size_t)MAX_STRING_LENGTH) >> shift_modifier, unit);
    printf("Memory to be allocated on the GPU to store the lines' length: %zu %s\n", (num_lines * lengths_size) >> shift_modifier, unit);
    printf("Memory to be allocated on the GPU to store the hashes: %zu %s\n", (num_lines * hash_size) >> shift_modifier, unit);
    printf("Total memory to be allocated on the GPU: %zu %s\n", total_memory_allocated >> shift_modifier, unit);
}

void get_gpu_memory_info(size_t *free_mem, size_t *total_avlbl_mem, size_t shift_modifier)
{
    char unit[6] = {0};
    size_t free_memory, total_available_memory;

    cudaMemGetInfo(&free_memory, &total_available_memory);

    if (free_mem != NULL && total_avlbl_mem != NULL)
    {
        *free_mem = free_memory;
        *total_avlbl_mem = total_available_memory;
    }

    switch (shift_modifier)
    {
        case 0:
            memcpy(unit, "Bytes", sizeof("Bytes"));
            break;
        case 10:
            memcpy(unit, "KiB", sizeof("KiB"));
            break;
        case 20:
            memcpy(unit, "MiB", sizeof("MiB"));
            break;
        case 30:
            memcpy(unit, "GiB", sizeof("GiB"));
            break;
        default:
            memcpy(unit, "Bytes", sizeof("Bytes"));
            break;
    }

    printf("Total memory GPU: %zu %s\nFree memory GPU: %zu %s\n", total_available_memory >> shift_modifier, unit, free_memory >> shift_modifier, unit);
}

// void print_execution_time(float nostream_gpu, float stream_gpu, size_t nostream_host, size_t stream_host, size_t divisor, size_t precision)
void print_execution_time(float gpu_time, size_t host_time)
{

    // add way to decide if we are using streams or not
    /* add enum with microseconds, milliseconds and seconds to decide */
    /* Add decision to print in milliseconds or seconds, or something else and maybe also the number of decimal digits */

    // printf("Time without streams: %.3f ms\n", nostream_gpu);
    // printf("Time with streams: %.3f ms\n", stream_gpu);
    // printf("Total execution time seen from host without streams: %.3fms\n", nostream_host / MSECPSEC_DIV);
    // printf("Total execution time seen from host with streams: %.3fms\n", stream_host / MSECPSEC_DIV);
    printf("Time of GPU: %.3f ms\n", gpu_time);
    printf("Total execution time seen from host: %.3fms\n", host_time / MSECPSEC_DIV);
}

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
    printf("\tMaximum block dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("\tMaximum grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
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

void allocate_cuda_host_memory(void **ptr, size_t alloc_size)
{
    CHECK_CUDA_ERROR(cudaHostAlloc((void**)ptr, alloc_size, cudaHostAllocDefault));
    CHECK_NULL(ptr);
}

void free_cuda_host_memory(void *ptr)
{
    CHECK_NULL(ptr);
    CHECK_CUDA_ERROR(cudaFreeHost(ptr));
    ptr = NULL;
}