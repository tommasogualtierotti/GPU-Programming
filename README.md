# HOWTO run the program
1. Extract a file from the dataset folder and put the name of the file in the `include/config.h` file under the `#define FILENAME_PATH` macro.
2. Edit the Makefile according to the features of the GPU used to run the code (more specifically edit the root directory of CUDA binaries and the SM_ARCH flag).
3. Edit the `include/config.h` file, and set the configuration to run the code, i.e. decided whether to use STREAM execution or not, whether to use the batch processing (if the dataset is too large to fit either in the CPU or the GPU ram memory) and so on.
4. Now type `make` in the terminal and then run the binary generated.