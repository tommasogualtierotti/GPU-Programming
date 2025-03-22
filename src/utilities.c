#include "../include/utilities.h"

size_t read_strings_from_file(const char *filename, char ***strings, size_t **lengths) {

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

size_t elapsed_time_usec(size_t start)
{
    timeval tv;

    gettimeofday(&tv, 0);

    return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

size_t elapsed_time_msec(size_t start)
{
    return elapsed_time_usec(start) / 1000U;
}
