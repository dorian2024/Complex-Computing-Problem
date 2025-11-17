# NVIDIA HPC SDK (recommended)
nvc -acc -Minfo=accel -ta=tesla:cc70 -O3 *.c -o klt

# GCC
gcc -fopenacc -O3 *.c -o klt
