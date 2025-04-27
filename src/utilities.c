#ifndef UTILITIES_H
    #include "../include/utilities.h"
#endif

/**
 * @brief Count the number of lines in a file.
 *
 * Opens the specified file, reads through each line to count them,
 * then closes the file.
 *
 * @param filename Path to the input file.
 * @return Number of lines in the file.
 */
size_t line_count(const char *filename) 
{
    FILE *file = fopen(filename, "r");
    if (!file) 
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(-1);
    }

    size_t num_strings = 0;
    char buffer[MAX_STRING_LENGTH];
    while (fgets(buffer, MAX_STRING_LENGTH, file)) 
    {
        num_strings++;
    }
    fclose(file);
    return num_strings;
}

/**
 * @brief Reads all strings from a file into dynamically allocated arrays.
 *
 * Allocates arrays for the strings and their lengths, reads each line
 * into its own buffer, strips the newline, and records its length.
 *
 * @param filename Path to the input file.
 * @param strings Output pointer to array of string pointers.
 * @param lengths Output pointer to array of string lengths.
 * @return Number of strings read.
 */
size_t read_strings_from_file(const char *filename, char ***strings, size_t **lengths) 
{
    FILE *file = fopen(filename, "r");
    if (!file) 
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(-1);
    }

    size_t num_strings = 0;
    char buffer[MAX_STRING_LENGTH];
    while (fgets(buffer, MAX_STRING_LENGTH, file)) 
    {
        num_strings++;
    }

    *strings = (char **)malloc(num_strings * sizeof(char *));
    if (*strings == NULL) 
    {
        fprintf(stderr, "Failed to allocate memory for strings\n");
        fclose(file);
        exit(-1);
    }

    *lengths = (size_t *)malloc(num_strings * sizeof(size_t));
    if (*lengths == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for strings length\n");
        fclose(file);
        exit(-1);
    }

    rewind(file);
    for (size_t i = 0; i < num_strings; i++) 
    {
        (*strings)[i] = (char *)malloc(MAX_STRING_LENGTH * sizeof(char));
        if ((*strings)[i] == NULL) 
        {
            fprintf(stderr, "Failed to allocate memory for string %zu\n", i);
            fclose(file);
            exit(-1);
        }

        if (!fgets((*strings)[i], MAX_STRING_LENGTH, file)) 
        {
            fprintf(stderr, "Failed to read string %zu from file\n", i);
            fclose(file);
            exit(-1);
        }
        (*strings)[i][strcspn((*strings)[i], "\n")] = '\0';
        (*lengths)[i] = strlen((*strings)[i]);
    }

    fclose(file);
    return num_strings;
}

/**
 * @brief Initialize a batch reader for chunked file reading.
 *
 * Allocates and initializes a batch_reader_t, opens the file,
 * counts total lines, and prepares for batch reads.
 *
 * @param filename Path to the input file.
 * @return Pointer to the initialized batch_reader_t, or NULL on failure.
 */
batch_reader_t* batch_reader_init(const char *filename) 
{
    batch_reader_t* batch_reader = (batch_reader_t *)malloc(sizeof(batch_reader_t));
    if (batch_reader == NULL) 
    {
        perror("Memory allocation failed");
        return NULL;
    }

    batch_reader->file = fopen(filename, "r");
    if (batch_reader->file == NULL) 
    {
        perror("Error opening file");
        free(batch_reader);
        return NULL;
    }

    batch_reader->buffer_pos = 0;
    batch_reader->batch_read_items = 0;

    size_t num_strings = 0;
    char buffer[MAX_STRING_LENGTH];
    while (fgets(buffer, MAX_STRING_LENGTH, batch_reader->file)) 
    {
        num_strings++;
    }
    batch_reader->total_items = num_strings;
    rewind(batch_reader->file);

    return batch_reader;
}

/**
 * @brief Read a fixed-size batch of strings from a file.
 *
 * Reads up to BATCH_NUM_LINES lines into preallocated buffers,
 * zeroing or allocating as needed on each call.
 *
 * @param batch_reader Initialized batch_reader_t.
 * @param strings In/out pointer to array of string buffers.
 * @param lengths In/out pointer to array of string lengths.
 * @return Number of strings read in this batch.
 */
size_t batch_read_strings_from_file(batch_reader_t *batch_reader, char ***strings, size_t **lengths) 
{
    if (batch_reader == NULL || batch_reader->file == NULL)
    {
        perror("Batch reader not correctly initialized");
        exit(-1);
    }

    batch_reader->batch_read_items = 0;

    if (*strings != NULL)
    {
        memset(*strings, 0x0, BATCH_NUM_LINES * sizeof(char *));
    } 
    else
    {
        *strings = (char **)calloc(BATCH_NUM_LINES, sizeof(char *));
        if (*strings == NULL)
        {
            perror("Memory allocation failed");
            exit(-1);
        }
    }

    if (*lengths != NULL)
    {
        memset(*lengths, 0x0, BATCH_NUM_LINES * sizeof(size_t));
    } 
    else
    {
        *lengths = (size_t *)calloc(BATCH_NUM_LINES, sizeof(size_t));
        if (*lengths == NULL)
        {
            perror("Memory allocation failed");
            exit(-1);
        }
    }

    ssize_t lines_read = 0;
    while (lines_read < BATCH_NUM_LINES)
    {
        if ((*strings)[lines_read] != NULL)
        {
            memset((*strings)[lines_read], 0x0, MAX_STRING_LENGTH * sizeof(char));
        } 
        else
        {
            (*strings)[lines_read] = (char *)calloc(MAX_STRING_LENGTH, sizeof(char));
            if ((*strings)[lines_read] == NULL) 
            {
                fprintf(stderr, "Failed to allocate memory for string %zd\n", lines_read);
                batch_reader_close(batch_reader);
                exit(-1);
            }
        }

        if (!fgets((*strings)[lines_read], MAX_STRING_LENGTH, batch_reader->file)) 
        {
            batch_reader->batch_read_items = lines_read;
            return lines_read;
        }

        (*strings)[lines_read][strcspn((*strings)[lines_read], "\n")] = '\0';
        (*lengths)[lines_read] = strlen((*strings)[lines_read]);

        batch_reader->buffer_pos++;
        lines_read++;
    }

    batch_reader->batch_read_items = lines_read;
    batch_reader->iteration++;

    return lines_read;
}

/**
 * @brief Close and free resources for a batch reader.
 *
 * Closes the file and frees the batch_reader_t struct.
 *
 * @param batch_reader Pointer to the batch_reader_t to close.
 */
void batch_reader_close(batch_reader_t *batch_reader) 
{
    if (batch_reader != NULL) 
    {
        fclose(batch_reader->file);
        free(batch_reader);
        batch_reader = NULL;
    }
}

/**
 * @brief Rewind the batch reader's file to the beginning.
 *
 * @param batch_reader Pointer to the batch_reader_t to rewind.
 */
void batch_reader_rewind_file(batch_reader_t* batch_reader)
{
    CHECK_NULL(batch_reader);
    rewind(batch_reader->file);
}

/**
 * @brief Compute elapsed time in microseconds since start.
 *
 * Uses gettimeofday() to compute the delta from the given start time.
 *
 * @param start Start time in microseconds.
 * @return Elapsed time in microseconds.
 */
size_t elapsed_time_usec(size_t start)
{
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

/**
 * @brief Print SHA-1 hashes computed on the GPU.
 *
 * Iterates over each string and its corresponding hash,
 * printing the string and its 5-word (20-byte) hash in hex.
 *
 * @param hashes Array of uint32_t hashes (5 words per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_sha1_hashes_gpu(uint32_t *hashes, char **strings, size_t num_lines)
{
    CHECK_NULL(hashes);
    CHECK_NULL(strings);

    for (size_t i = 0; i < num_lines; i++)
    {
        printf("String: %s\tHash: ", strings[i]);
        for (size_t j = 0; j < SHA1_HASH_LENGTH; ++j) {
            printf("%08x", hashes[i * SHA1_HASH_LENGTH + j]);
        }
        printf("\n");
    }
}

/**
 * @brief Print SHA-1 hashes computed on the CPU.
 *
 * Iterates over each string and its corresponding 20-byte hash,
 * printing the string and hash in two-digit hex.
 *
 * @param hashes Array of uint8_t hashes (20 bytes per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_sha1_hashes_cpu(uint8_t *hashes, char **strings, size_t num_lines)
{
    CHECK_NULL(hashes);
    CHECK_NULL(strings);

    for (size_t i = 0; i < num_lines; i++)
    {
        printf("String: %s\tHash: ", strings[i]);
        for (size_t j = 0; j < SHA1_HASH_LENGTH_BYTES; ++j) {
            printf("%02x", hashes[i * SHA1_HASH_LENGTH_BYTES + j]);
        }
        printf("\n");
    }
}

/**
 * @brief Print MD5 hashes computed on the GPU.
 *
 * Iterates over each string and its corresponding hash,
 * printing the string and its 4-word (16-byte) hash in hex.
 *
 * @param hashes Array of uint32_t hashes (16 bytes per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_md5_hashes_gpu(uint8_t *hashes, char **strings, size_t num_lines)
{
    CHECK_NULL(hashes);
    CHECK_NULL(strings);

    for (size_t i = 0; i < num_lines; i++)
    {
        printf("String: %s\tHash: ", strings[i]);
        for (size_t j = 0; j < MD5_HASH_LENGTH_BYTES; ++j) {
            printf("%02x", hashes[i * MD5_HASH_LENGTH_BYTES + j]);
        }
        printf("\n");
    }
}

/**
 * @brief Print MD5 hashes computed on the CPU.
 *
 * For each string and its hash, prints the string followed by its
 * 16‑byte hash in two‑digit hexadecimal.
 *
 * @param hashes Array of uint8_t hashes (16 bytes per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_md5_hashes_cpu(uint8_t *hashes, char **strings, size_t num_lines)
{

    CHECK_NULL(hashes);
    CHECK_NULL(strings);

    for (size_t i = 0; i < num_lines; i++)
    {
        printf("String: %s\tHash: ", strings[i]);
        for (size_t j = 0; j < MD5_HASH_LENGTH_BYTES; ++j) {
            printf("%02x", hashes[i * MD5_HASH_LENGTH_BYTES + j]);
        }
        printf("\n");
    }
}

/**
 * @brief Print SHA-256 hashes computed on the GPU.
 *
 * For each string and its hash, prints the string followed by its
 * 8‑word (32‑byte) hash in hexadecimal.
 *
 * @param hashes Array of uint8_t hashes (32 bytes per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_sha256_hashes_gpu(uint8_t *hashes, char **strings, size_t num_lines)
{
    CHECK_NULL(hashes);
    CHECK_NULL(strings);

    for (size_t i = 0; i < num_lines; i++)
    {
        printf("String: %s\tHash: ", strings[i]);
        for (size_t j = 0; j < SHA256_HASH_LENGTH_BYTES; ++j) {
            printf("%02x", hashes[i * SHA256_HASH_LENGTH_BYTES + j]);
        }
        printf("\n");
    }
}

/**
 * @brief Print SHA‑256 hashes computed on the CPU.
 *
 * For each string and its hash, prints the string followed by its
 * 32‑byte hash in two‑digit hexadecimal.
 *
 * @param hashes Array of uint8_t hashes (32 bytes per string).
 * @param strings Array of input strings.
 * @param num_lines Number of strings (and hashes).
 */
void print_sha256_hashes_cpu(uint8_t *hashes, char **strings, size_t num_lines)
{
    CHECK_NULL(hashes);
    CHECK_NULL(strings);

    for (size_t i = 0; i < num_lines; i++)
    {
        printf("String: %s\tHash: ", strings[i]);
        for (size_t j = 0; j < SHA256_HASH_LENGTH_BYTES; ++j) {
            printf("%02x", hashes[i * SHA256_HASH_LENGTH_BYTES + j]);
        }
        printf("\n");
    }
}