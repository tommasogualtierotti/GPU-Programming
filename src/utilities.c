#include "../include/utilities.h"

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
 * @brief Reads all strings from a file into memory.
 *
 * This function reads all lines from the given file, storing them as strings
 * and their respective lengths in dynamically allocated arrays.
 *
 * @param filename The name of the file to read from.
 * @param strings Pointer to an array of strings (output).
 * @param lengths Pointer to an array of string lengths (output).
 * @return The number of strings read, or -1 on failure.
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
            fprintf(stderr, "Failed to allocate memory for string %ld\n", i);
            fclose(file);
            exit(-1);
        }

        if (!fgets((*strings)[i], MAX_STRING_LENGTH, file)) 
        {
            fprintf(stderr, "Failed to read string %ld from file\n", i);
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
 * @brief Initializes a batch reader for reading file contents in chunks.
 *
 * @param filename The name of the file to read.
 * @return A pointer to an initialized batch_reader_t structure, or NULL on failure.
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
 * @brief Reads a batch of strings from a file using a batch reader.
 *
 * @param batch_reader Pointer to an initialized batch_reader_t.
 * @param strings Pointer to an array of strings (output).
 * @param lengths Pointer to an array of string lengths (output).
 * @return The number of strings read in the batch, or -1 on failure.
 */
size_t batch_read_strings_from_file(batch_reader_t *batch_reader, char ***strings, size_t **lengths) 
{

    if (batch_reader == NULL || batch_reader -> file == NULL)
    {
        perror("Batch reader not correctly initialized");
        exit(-1);
    }

    batch_reader->batch_read_items = 0;

    /* If the pointer is not NULL it means it was already allocated in the previous iteration,
     * therefore the previous content is zeroed out and new content is written.
     */
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

    /* If the pointer is not NULL it means it was already allocated in the previous iteration,
     * therefore the previous content is zeroed out and new content is written.
     */
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
        /* If the pointer is not NULL it means it was already allocated in the previous iteration,
         * therefore the previous content is zeroed out and new content is written.
         */
        if ((*strings)[lines_read] != NULL)
        {
            memset(*strings[lines_read], 0x0, MAX_STRING_LENGTH * sizeof(char));
        } 
        else
        {
            (*strings)[lines_read] = (char *)calloc(MAX_STRING_LENGTH, sizeof(char));

            if ((*strings)[lines_read] == NULL) 
            {
                fprintf(stderr, "Failed to allocate memory for string %ld\n", lines_read);
                batch_reader_close(batch_reader);
                exit(-1);
            }
        }

        if (!fgets((*strings)[lines_read], MAX_STRING_LENGTH, batch_reader->file)) 
        {
            /* In case we terminate earlier than BATCH_NUM_LINES reading lines the number of read lines is returned,
             * in this way the fact that the end of the file was reached can be signaled.
             */
            batch_reader->batch_read_items = lines_read;
            return lines_read;
            // fprintf(stderr, "Failed to read string %ld from file\n", lines_read);
            // batch_reader_close(batch_reader);
            // return EXIT_FAILURE;
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
 * @brief Closes a batch reader and releases allocated resources.
 *
 * @param batch_reader Pointer to the batch_reader_t to be closed.
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

void batch_reader_rewind_file(batch_reader_t* batch_reader)
{
    CHECK_NULL(batch_reader);
    rewind(batch_reader->file);
}

/**
 * @brief Calculates the elapsed time in microseconds from a given start time.
 *
 * @param start The starting timestamp in microseconds.
 * @return The elapsed time in microseconds.
 */
size_t elapsed_time_usec(size_t start)
{
    timeval tv;

    gettimeofday(&tv, 0);

    return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

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

void print_sha1_hashes_cpu(uint8_t *hashes, char **strings, size_t num_lines)
{
    CHECK_NULL(hashes);
    CHECK_NULL(strings);

    uint8_t hash_size_bytes = 20;

    for (size_t i = 0; i < num_lines; i++)
    {
        printf("String: %s\tHash: ", strings[i]);
        for (size_t j = 0; j < hash_size_bytes; ++j) {
            printf("%02x", hashes[i * hash_size_bytes + j]);
        }
        printf("\n");
    }
}