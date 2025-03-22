#include "../include/utilities.h"

ssize_t read_strings_from_file(const char *filename, char ***strings, size_t **lengths) {

    FILE *file = fopen(filename, "r");

    if (!file) 
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return -1;
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
        return -1;
    }

    *lengths = (size_t *)malloc(num_strings * sizeof(size_t));
    if (*lengths == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for strings length\n");
        fclose(file);
        return -1;
    }

    rewind(file);
    for (size_t i = 0; i < num_strings; i++) 
    {
        (*strings)[i] = (char *)malloc(MAX_STRING_LENGTH * sizeof(char));

        if ((*strings)[i] == NULL) 
        {
            fprintf(stderr, "Failed to allocate memory for string %ld\n", i);
            fclose(file);
            return -1;
        }

        if (!fgets((*strings)[i], MAX_STRING_LENGTH, file)) 
        {
            fprintf(stderr, "Failed to read string %ld from file\n", i);
            fclose(file);
            return -1;
        }

        (*strings)[i][strcspn((*strings)[i], "\n")] = '\0';
        
        (*lengths)[i] = strlen((*strings)[i]);
    }

    fclose(file);
    return num_strings;
}

batch_reader_t* batch_reader_init(const char *filename) {

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

    rewind(batch_reader -> file);

    return batch_reader;
}

ssize_t batch_read_strings_from_file(batch_reader_t *batch_reader, char ***strings, size_t **lengths) 
{
    batch_reader->batch_read_items = 0;

    /* TODO: MOVE THE INITIALIZATION ELSEWHERE and inser the reset of the buffer when reading of new batch */
    if (batch_reader == NULL || batch_reader -> file == NULL)
    {
        perror("Batch reader not correctly initialized");
        return EXIT_FAILURE;
    }

    *strings = (char **)malloc(BATCH_NUM_LINES * sizeof(char *));

    if (*strings == NULL)
    {
        perror("Memory allocation failed");
        return EXIT_FAILURE;
    }

    *lengths = (size_t *)malloc(BATCH_NUM_LINES * sizeof(size_t));

    if (*lengths == NULL)
    {
        perror("Memory allocation failed");
        return EXIT_FAILURE;
    }

    ssize_t lines_read = 0;

    while (lines_read < BATCH_NUM_LINES)
    {
        (*strings)[lines_read] = (char *)malloc(MAX_STRING_LENGTH * sizeof(char));

        if ((*strings)[lines_read] == NULL) 
        {
            fprintf(stderr, "Failed to allocate memory for string %ld\n", lines_read);
            batch_reader_close(batch_reader);
            return EXIT_FAILURE;
        }

        if (!fgets((*strings)[lines_read], MAX_STRING_LENGTH, batch_reader->file)) 
        {
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

    return lines_read;
}

void batch_reader_close(batch_reader_t *batch_reader) 
{
    if (batch_reader != NULL) 
    {
        fclose(batch_reader->file);
        free(batch_reader);
        batch_reader = NULL;
    }
}

size_t elapsed_time_usec(size_t start)
{
    timeval tv;

    gettimeofday(&tv, 0);

    return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}
