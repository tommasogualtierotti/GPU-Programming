#ifndef CONFIG_H
#define CONFIG_H

/**
 * @file config.h
 * @brief Configuration header for the CUDA-based cryptographic hashing application.
 *
 * This file contains macro definitions used to configure the behavior of
 * the parallel SHA-1, SHA-256, and MD5 hashing implementations.
 */

/** 
 * @def FILENAME_PATH
 * @brief The file path for the dataset.
 *
 * Defines the path to the dictionary file that will be used in the application.
 */
#define FILENAME_PATH "dataset/all_patterns.dict"

/** 
 * @def BATCH_NUM_LINES
 * @brief Number of lines to process in each batch.
 *
 * Defines the batch size (number of lines) to process at a time.
 */
#define BATCH_NUM_LINES 1000000ULL

/** 
 * @def MAX_STRING_LENGTH
 * @brief Maximum length of a string.
 *
 * Defines the maximum number of characters allowed in a string to be processed.
 */
#define MAX_STRING_LENGTH 30ULL

/**
 * @def USE_STREAMS
 * @brief Enable CUDA streams for parallel execution.
 *
 * If defined, CUDA streams will be used to perform asynchronous tasks and improve performance.
 */
#ifndef USE_STREAMS
// #define USE_STREAMS
#endif

/** 
 * @def NUM_STREAMS
 * @brief Number of CUDA streams to use.
 *
 * Specifies how many CUDA streams are utilized for parallel execution.
 */
#define NUM_STREAMS 8ULL

/** 
 * @def CHUNKS
 * @brief Number of chunks for data splitting.
 *
 * Defines into how many chunks the data should be split for parallel processing.
 */
#define CHUNKS 64ULL

/** 
 * @def THREADS_PER_BLOCK
 * @brief Number of threads per block in CUDA.
 *
 * Defines the number of threads assigned to each CUDA block during kernel execution.
 */
#define THREADS_PER_BLOCK 64ULL

/** 
 * @def BATCH_PROCESSING
 * @brief Enable batch processing.
 *
 * If defined, enables batch-based hashing operations.
 */
// #define BATCH_PROCESSING

/** 
 * @def CPU_SHA1_RUN
 * @brief Enable CPU-based SHA-1 hash computation.
 *
 * If defined, the SHA-1 hashing will be performed on the CPU.
 */
// #define CPU_SHA1_RUN

/** 
 * @def CPU_SHA256_RUN
 * @brief Enable CPU-based SHA-256 hash computation.
 *
 * If defined, the SHA-256 hashing will be performed on the CPU.
 */
// #define CPU_SHA256_RUN

/** 
 * @def CPU_MD5_RUN
 * @brief Enable CPU-based MD5 hash computation.
 *
 * If defined, the MD5 hashing will be performed on the CPU.
 */
// #define CPU_MD5_RUN

/** 
 * @def CPU_HASH_RUN
 * @brief Enable CPU-based hashing computation.
 *
 * If defined, the CPU versions of all hashing algorithms will run.
 */
// #define CPU_HASH_RUN

/** 
 * @def DEBUG_PRINT_HASHES
 * @brief Enable debug printing of hashes.
 *
 * If defined, debug information about hash computations will be printed.
 */
// #define DEBUG_PRINT_HASHES

/** 
 * @def PRINT_GPU_INFO
 * @brief Enable printing of GPU information.
 *
 * If defined, prints details about the GPU hardware and CUDA execution environment.
 */
// #define PRINT_GPU_INFO

/** 
 * @def SHA1_PARALLEL
 * @brief Enable parallel SHA-1 computation on GPU.
 */
#define SHA1_PARALLEL

/** 
 * @def MD5_PARALLEL
 * @brief Enable parallel MD5 computation on GPU.
 */
// #define MD5_PARALLEL

/** 
 * @def SHA256_PARALLEL
 * @brief Enable parallel SHA-256 computation on GPU.
 */
// #define SHA256_PARALLEL

#endif /* CONFIG_H */
