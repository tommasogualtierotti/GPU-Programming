###########################################################

## USER SPECIFIC DIRECTORIES ##
DEBUG=1
# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-12.6

##########################################################
MKDIR=mkdir
## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS= #-fpermissive
CC_LIBS=

##########################################################

### GPU architecture
SM_ARCH=75

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-gencode arch=compute_${SM_ARCH},code=sm_${SM_ARCH}
NVCC_LIBS=

ifeq ($(DEBUG),1)
	CC_FLAGS+= -g
    NVCC_FLAGS+= -g -G
endif
# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = test

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/sha1_parallel.o $(OBJ_DIR)/utilities.o $(OBJ_DIR)/sha1.o $(OBJ_DIR)/sha256.o $(OBJ_DIR)/md5.o $(OBJ_DIR)/cuda_utilities.o

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .c file to object files:
$(OBJ_DIR)/%.o : %.c $(OBJ_DIR) 
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c include/%.h $(OBJ_DIR) 
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh $(OBJ_DIR) 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR):
	$(MKDIR) -p $@
# Clean objects in object directory.
clean:
	$(RM) -r bin/ *.txt
	$(RM) $(EXE)

run:
	./${EXE} > stdout.txt 2> stderr.txt
