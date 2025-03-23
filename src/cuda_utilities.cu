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