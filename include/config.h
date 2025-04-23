#ifndef CONFIG_H
    #define CONFIG_H
#endif

/**
 * @brief The file path for the dataset.
 * 
 * This macro defines the path to the dictionary file that will be used in the application.
 */
#define FILENAME_PATH "dataset/italian.dict"

/**
 * @brief The number of lines to process in each batch.
 * 
 * This macro defines the batch size (in terms of the number of lines) to process at a time.
 */
#define BATCH_NUM_LINES 1000000ULL

/**
 * @brief The maximum length of a string.
 * 
 * This macro defines the maximum number of characters allowed in a string to be processed.
 */
#define MAX_STRING_LENGTH 30ULL

/**
 * @brief Enable CUDA streams for parallel execution.
 * 
 * If this macro is defined, CUDA streams will be used to perform asynchronous tasks and improve performance.
 */
#ifndef USE_STREAMS
    // #define USE_STREAMS
#endif

/**
 * @brief The number of CUDA streams to use.
 * 
 * This macro specifies the number of CUDA streams that will be utilized for parallel execution. 
 * It is typically optimized based on the target hardware.
 */
#define NUM_STREAMS 8

/**
 * @brief The number of chunks to process.
 * 
 * This macro defines the number of chunks to split the data into for parallel processing.
 */
#define CHUNKS 64

/**
 * @brief The number of threads per block in CUDA.
 * 
 * This macro defines the number of threads that will be assigned to each block in CUDA kernel execution.
 */
#define THREADS_PER_BLOCK 64ULL

/**
 * @brief Enable batch processing.
 * 
 * If this macro is defined, batch processing is enabled for the hashing operations.
 */
// #define BATCH_PROCESSING

/**
 * @brief Enable CPU-based SHA-1 hash computation.
 * 
 * If this macro is defined, the SHA-1 hashing operation will run on the CPU.
 */
#define CPU_SHA1_RUN

/**
 * @brief Enable CPU-based SHA-256 hash computation.
 * 
 * If this macro is defined, the SHA-256 hashing operation will run on the CPU.
 */
// #define CPU_SHA256_RUN

/**
 * @brief Enable CPU-based MD5 hash computation.
 * 
 * If this macro is defined, the MD5 hashing operation will run on the CPU.
 */
// #define CPU_SHAMD5_RUN

/**
 * @brief Enable debug printing of hashes.
 * 
 * If this macro is defined, debug information related to hash computations will be printed.
 */
// #define DEBUG_PRINT_HASHES

/**
 * @brief Enable printing of GPU information.
 * 
 * If this macro is defined, information about the GPU hardware and CUDA execution will be printed.
 */
// #define PRINT_GPU_INFO
