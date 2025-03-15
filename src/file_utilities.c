#include "../include/file_utilities.h"

size_t read_strings_from_file(const char *filename, char ***strings) {

    FILE *file = fopen(filename, "r");

    if (!file) 
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return -1;
    }

    // Count the number of strings in the file
    size_t num_strings = 0;
    char buffer[MAX_STRING_LENGTH];
    while (fgets(buffer, MAX_STRING_LENGTH, file)) 
    {
        num_strings++;
    }

    // Allocate memory for the strings
    *strings = (char **)malloc(num_strings * sizeof(char *));
    if (!*strings) 
    {
        fprintf(stderr, "Failed to allocate memory for strings\n");
        fclose(file);
        return -1;
    }

    // Rewind the file and read the strings
    rewind(file);
    for (size_t i = 0; i < num_strings; i++) 
    {
        (*strings)[i] = (char *)malloc(MAX_STRING_LENGTH * sizeof(char));
        if (!(*strings)[i]) 
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
        // Remove newline character if present
        (*strings)[i][strcspn((*strings)[i], "\n")] = '\0';
    }

    fclose(file);
    return num_strings;
}