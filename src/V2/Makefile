######################################################################
# Choose your favorite C compiler
CC = gcc
# CUDA compiler
NVCC = /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin/nvcc
CUDA_PATH = /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda
######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
FLAG1 = -DNDEBUG
FLAG2= -pg
######################################################################
# -DKLT_USE_QSORT forces the code to use the standard qsort() 
# routine.  Otherwise it will use a quicksort routine that takes
# advantage of our specific data structure to greatly reduce the
# running time on some machines.  Uncomment this line if for some
# reason you are unhappy with the special routine.
# FLAG2 = -DKLT_USE_QSORT
######################################################################
# Add your favorite C flags here.
# Add CUDA include path for C files that include CUDA headers
CUDA_INCLUDE = -I$(CUDA_PATH)/include
CFLAGS = $(FLAG1) $(FLAG2) $(CUDA_INCLUDE) -DUSE_GPU=0
# CUDA compilation flags
# RTX 3080 has compute capability 8.6 (Ampere architecture)
NVCCFLAGS = -DNDEBUG -O3 -arch=sm_86 -DUSE_GPU=1
######################################################################
# There should be no need to modify anything below this line (but
# feel free to if you want).
EXAMPLES = example1.c example2.c example3.c example4.c example5.c

# Split ARCH into C and CUDA files
ARCH_C = error.c pnmio.c pyramid.c selectGoodFeatures.c \
         storeFeatures.c klt.c klt_util.c writeFeatures.c
ARCH_CU = convolve.cu trackFeatures.cu

# CPU-only version uses C implementation of convolve and trackFeatures
ARCH_C_CPU = $(ARCH_C) convolve.c trackFeatures.c
ARCH_OBJS_CPU = $(ARCH_C_CPU:.c=.o)

# GPU version uses CUDA implementation of convolve and trackFeatures
ARCH_OBJS_GPU = $(ARCH_C:.c=.o) $(ARCH_CU:.cu=.o)

LIB_GPU = -L/usr/local/lib -L/usr/lib -L$(CUDA_PATH)/lib64 -lcudart
LIB_CPU = -L/usr/local/lib -L/usr/lib

.SUFFIXES:  .c .o .cu

# Default target shows help
help:
	@echo "Available targets:"
	@echo "  make cpu        - Build CPU version (no CUDA required)"
	@echo "  make gpu        - Build GPU version (requires CUDA)"
	@echo "  make all        - Build GPU version (default)"
	@echo "  make clean      - Remove all build artifacts"
	@echo "  make depend     - Generate dependencies"

# GPU version (default)
all: gpu

gpu: 
	@echo "Starting GPU build at $(date '+%H:%M:%S')"
	@start_time=$(date +%s); \
	$(MAKE) lib-gpu $(EXAMPLES:.c=-gpu); \
	end_time=$(date +%s); \
	elapsed=$((end_time - start_time)); \
	echo "GPU build completed in $elapsed seconds"

# CPU version
cpu:
	@echo "Starting CPU build at $(date '+%H:%M:%S')"
	@start_time=$(date +%s); \
	$(MAKE) lib-cpu $(EXAMPLES:.c=-cpu); \
	end_time=$(date +%s); \
	elapsed=$((end_time - start_time)); \
	echo "CPU build completed in $elapsed seconds"

.c.o:
	$(CC) -c $(CFLAGS) $<

.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $<

# GPU library
lib-gpu: libklt-gpu.a

libklt-gpu.a: $(ARCH_OBJS_GPU)
	rm -f libklt-gpu.a
	ar ruv libklt-gpu.a $(ARCH_OBJS_GPU)

# CPU library
lib-cpu: libklt-cpu.a

libklt-cpu.a: $(ARCH_OBJS_CPU)
	rm -f libklt-cpu.a
	ar ruv libklt-cpu.a $(ARCH_OBJS_CPU)

# GPU examples
example1-gpu: libklt-gpu.a
	$(NVCC) -O3 $(NVCCFLAGS) -o example1 example1.c -L. -lklt-gpu $(LIB_GPU) -lm

example2-gpu: libklt-gpu.a
	$(NVCC) -O3 $(NVCCFLAGS) -o example2 example2.c -L. -lklt-gpu $(LIB_GPU) -lm
	
example3-gpu: libklt-gpu.a
	@echo "Starting GPU build for example3 at $(shell date '+%H:%M:%S.%3N')"
	@start_time=$$(date +%s.%3N); \
	$(NVCC) -O3 $(NVCCFLAGS) -o example3 example3.c -L. -lklt-gpu $(LIB_GPU) -lm; \
	end_time=$$(date +%s.%3N); \
	elapsed=$$(echo "$$end_time - $$start_time" | bc); \
	echo "GPU build for example3 completed in $$elapsed seconds"

example4-gpu: libklt-gpu.a
	$(NVCC) -O3 $(NVCCFLAGS) -o example4 example4.c -L. -lklt-gpu $(LIB_GPU) -lm

example5-gpu: libklt-gpu.a
	$(NVCC) -O3 $(NVCCFLAGS) -o example5 example5.c -L. -lklt-gpu $(LIB_GPU) -lm

# CPU examples
example1-cpu: libklt-cpu.a
	$(CC) -O3 $(CFLAGS) -o example1 example1.c -L. -lklt-cpu $(LIB_CPU) -lm

example2-cpu: libklt-cpu.a
	$(CC) -O3 $(CFLAGS) -o example2 example2.c -L. -lklt-cpu $(LIB_CPU) -lm

example3-cpu: libklt-cpu.a
	@echo "Starting CPU build for example3 at $(shell date '+%H:%M:%S.%3N')"
	@start_time=$$(date +%s.%3N); \
	$(CC) -O3 $(CFLAGS) -o example3 example3.c -L. -lklt-cpu $(LIB_CPU) -lm; \
	end_time=$$(date +%s.%3N); \
	elapsed=$$(echo "$$end_time - $$start_time" | bc); \
	echo "CPU build for example3 completed in $$elapsed seconds"

example4-cpu: libklt-cpu.a
	$(CC) -O3 $(CFLAGS) -o example4 example4.c -L. -lklt-cpu $(LIB_CPU) -lm

example5-cpu: libklt-cpu.a
	$(CC) -O3 $(CFLAGS) -o example5 example5.c -L. -lklt-cpu $(LIB_CPU) -lm

depend:
	makedepend $(ARCH_C) $(ARCH_CU) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt-gpu.a libklt-cpu.a \
	      feat*.ppm features.ft features.txt

.PHONY: all cpu gpu help lib-cpu lib-gpu clean depend
