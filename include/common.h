#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef COMMON_H
    #define COMMON_H
#endif

#define LEFTROTATE(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits))))