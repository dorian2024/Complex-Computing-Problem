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
CFLAGS = $(FLAG1) $(FLAG2) $(CUDA_INCLUDE)
# CUDA compilation flags
# RTX 3080 has compute capability 8.6 (Ampere architecture)
NVCCFLAGS = -DNDEBUG -O3 -arch=sm_86
######################################################################
# There should be no need to modify anything below this line (but
# feel free to if you want).
EXAMPLES = example1.c example2.c example3.c example4.c example5.c
# Split ARCH into C and CUDA files
ARCH_C = error.c pnmio.c pyramid.c selectGoodFeatures.c \
         storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
ARCH_CU = convolve.cu
ARCH_OBJS = $(ARCH_C:.c=.o) $(ARCH_CU:.cu=.o)
LIB = -L/usr/local/lib -L/usr/lib -L$(CUDA_PATH)/lib64 -lcudart
.SUFFIXES:  .c .o .cu
all:  lib $(EXAMPLES:.c=)
.c.o:
	$(CC) -c $(CFLAGS) $<
.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $<
lib: libklt.a

libklt.a: $(ARCH_OBJS)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH_OBJS)
example1: libklt.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm
example2: libklt.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm
example3: libklt.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm
example4: libklt.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm
example5: libklt.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm
depend:
	makedepend $(ARCH_C) $(ARCH_CU) $(EXAMPLES)
clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt
