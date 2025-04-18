#ifndef CONFIG_H
    #define CONFIG_H
#endif

#define FILENAME_PATH "dataset/all_patterns.dict"

#define BATCH_NUM_LINES 1000000ULL

#define MAX_STRING_LENGTH 30ULL

#ifndef USE_STREAMS
// #define USE_STREAMS
#endif

#define NUM_STREAMS 8
#define CHUNKS 64

#define THREADS_PER_BLOCK 512ULL

// #define BATCH_PROCESSING

// #define CPU_SHA1_RUN
// #define CPU_SHA256_RUN
// #define CPU_SHAMD5_RUN

// #define DEBUG_PRINT_HASHES

// #define PRINT_GPU_MEMORY_INFO