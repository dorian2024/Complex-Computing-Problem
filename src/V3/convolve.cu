/*********************************************************************
 * convolve.cu
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>   /* malloc(), realloc() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"   /* printing */
#include "klt.h"

// Pool for convolution operations
typedef struct {
    float* d_imgin;
    float* d_imgout;
    float* d_tmpimg;
    float* d_kernel;

    // Batch processing buffers
    float** d_batch_inputs;
    float** d_batch_temps;
    float** d_batch_outputs;
    float** h_batch_inputs;
    float** h_batch_temps;
    float** h_batch_outputs;

    // Prefetch buffers
    float* d_prefetch_buffer[2];
    int prefetch_buffer_size;
    int current_prefetch;

    size_t img_capacity;
    size_t batch_capacity;
    int max_batch_size;
    bool initialized;
    bool batch_initialized;
} ConvolutionPool;

static ConvolutionPool g_conv_pool = {
    NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL,
    {NULL, NULL}, 0, 0,
    0, 0, 0, false, false
};

// Pool for kernel data (persistent across calls)
static float* d_gauss_kernel_data = NULL;
static float* d_gaussderiv_kernel_data = NULL;
static bool kernels_on_device = false;

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CONVOLUTION_BLOCK_SIZE 32
#define GRADIENT_BLOCK_SIZE 16
#define PREFETCH_ENABLED 1
#define USE_PINNED_MEMORY 1
#define MAX_CONCURRENT_STREAMS 4

// Dynamic shared memory calculator
inline size_t calculateSharedMemory(int blockDim, int kernelWidth) {
    return (blockDim * blockDim + kernelWidth) * sizeof(float);
}

void _KLTInitBatchConvolution(int max_batch_size) {
    if (g_conv_pool.batch_initialized) return;

    g_conv_pool.max_batch_size = max_batch_size;

    // Allocate host arrays for batch pointers
    g_conv_pool.h_batch_inputs = (float**)malloc(max_batch_size * sizeof(float*));
    g_conv_pool.h_batch_temps = (float**)malloc(max_batch_size * sizeof(float*));
    g_conv_pool.h_batch_outputs = (float**)malloc(max_batch_size * sizeof(float*));

    // Allocate device arrays for batch pointers
    cudaCheck(cudaMalloc(&g_conv_pool.d_batch_inputs,
        max_batch_size * sizeof(float*)));
    cudaCheck(cudaMalloc(&g_conv_pool.d_batch_temps,
        max_batch_size * sizeof(float*)));
    cudaCheck(cudaMalloc(&g_conv_pool.d_batch_outputs,
        max_batch_size * sizeof(float*)));

    g_conv_pool.batch_initialized = true;
}

void _KLTCleanupBatchConvolution(void) {
    if (g_conv_pool.batch_initialized) {
        cudaFree(g_conv_pool.d_batch_inputs);
        cudaFree(g_conv_pool.d_batch_temps);
        cudaFree(g_conv_pool.d_batch_outputs);

        free(g_conv_pool.h_batch_inputs);
        free(g_conv_pool.h_batch_temps);
        free(g_conv_pool.h_batch_outputs);

        g_conv_pool.batch_initialized = false;
    }

    // Cleanup prefetch buffers
    for (int i = 0; i < 2; i++) {
        if (g_conv_pool.d_prefetch_buffer[i]) {
            cudaFree(g_conv_pool.d_prefetch_buffer[i]);
            g_conv_pool.d_prefetch_buffer[i] = NULL;
        }
    }
}

void _KLTCleanupConvolveCUDA(void) {
    // Free convolution pool
    if (g_conv_pool.initialized) {
        cudaFree(g_conv_pool.d_imgin);
        cudaFree(g_conv_pool.d_imgout);
        cudaFree(g_conv_pool.d_tmpimg);
        cudaFree(g_conv_pool.d_kernel);
        g_conv_pool.initialized = false;
        g_conv_pool.img_capacity = 0;
    }

    // Free persistent kernel data
    if (kernels_on_device) {
        cudaFree(d_gauss_kernel_data);
        cudaFree(d_gaussderiv_kernel_data);
        kernels_on_device = false;
    }

    // Free batch resources
    _KLTCleanupBatchConvolution();
}

// Add this function to convolve.cu
static cudaStream_t _getStream(int index) {
    static cudaStream_t streams[4] = { 0 };
    static bool initialized = false;

    if (!initialized) {
        for (int i = 0; i < 4; i++) {
            cudaCheck(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        }
        initialized = true;
    }

    return streams[index % 4];
}

#define MAX_KERNEL_WIDTH 71

typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;


/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */

 /*********************************************************************
  * _KLTToFloatImage - OPTIMIZED with parallel CPU conversion
  */
void _KLTToFloatImage(
    KLT_PixelType* img,
    int ncols, int nrows,
    _KLT_FloatImage floatimg)
{
    assert(floatimg->ncols >= ncols);
    assert(floatimg->nrows >= nrows);

    floatimg->ncols = ncols;
    floatimg->nrows = nrows;

    int total = ncols * nrows;

    for (int i = 0; i < total; i++) {
        floatimg->data[i] = (float)img[i];
    }
}


/*********************************************************************
 * _computeKernels
 */

static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    /* Compute gauss and deriv */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}
	

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

/*********************************************************************
 * _convolveImageHoriz with Shared Memory - MATCHES NAIVE BEHAVIOR
 */
__global__ void convolveImageHorizKernel_SM(
    const float* imgin,
    const float* kernel_data,
    int kernel_width,
    int ncols,
    int nrows,
    float* imgout)
{
    extern __shared__ float shared_data[];

    // Shared memory layout: [tile_data][kernel_data]
    float* sdata = shared_data;
    float* skernel = shared_data + blockDim.x * blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int radius = kernel_width / 2;

    // Cooperative kernel loading (once per block)
    if (ty == 0 && tx < kernel_width) {
        skernel[tx] = kernel_data[tx];
    }

    // Global position accounting for radius offset
    int col = blockIdx.x * (blockDim.x - 2 * radius) + tx - radius;
    int row = blockIdx.y * (blockDim.y - 2 * radius) + ty - radius;

    // Coalesced load into shared memory
    if (row >= 0 && row < nrows && col >= 0 && col < ncols) {
        sdata[ty * blockDim.x + tx] = imgin[row * ncols + col];
    }
    else {
        sdata[ty * blockDim.x + tx] = 0.0f;
    }

    __syncthreads();

    // Tile coordinates
    int tileCol = tx - radius;
    int tileRow = ty - radius;

    // Compute output for valid threads
    if (tileCol >= 0 && tileCol < blockDim.x - 2 * radius &&
        tileRow >= 0 && tileRow < blockDim.y - 2 * radius) {

        if (col >= 0 && col < ncols && row >= 0 && row < nrows) {

            // Zero out leftmost and rightmost columns
            if (col < radius || col >= ncols - radius) {
                imgout[row * ncols + col] = 0.0f;
                return;
            }

            // Optimized convolution with unrolling hint
            float sum = 0.0f;
#pragma unroll 8
            for (int k = 0; k < kernel_width; k++) {
                int sharedCol = tileCol + k;
                sum += sdata[ty * blockDim.x + sharedCol] * skernel[kernel_width - 1 - k];
            }

            imgout[row * ncols + col] = sum;
        }
    }
}

/*********************************************************************
 * _convolveImageVert with Shared Memory - MATCHES NAIVE BEHAVIOR
 */
__global__ void convolveImageVertKernel_SM(
    const float* imgin,
    const float* kernel_data,
    int kernel_width,
    int ncols,
    int nrows,
    float* imgout)
{
    extern __shared__ float shared_data[];

    // Shared memory layout: [tile_data][kernel_data]
    float* sdata = shared_data;
    float* skernel = shared_data + blockDim.x * blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int radius = kernel_width / 2;

    // Cooperative kernel loading
    if (ty == 0 && tx < kernel_width) {
        skernel[tx] = kernel_data[tx];
    }

    // Global position accounting for radius offset
    int col = blockIdx.x * (blockDim.x - 2 * radius) + tx - radius;
    int row = blockIdx.y * (blockDim.y - 2 * radius) + ty - radius;

    // Load tile into shared memory
    if (row >= 0 && row < nrows && col >= 0 && col < ncols) {
        sdata[ty * blockDim.x + tx] = imgin[row * ncols + col];
    }
    else {
        sdata[ty * blockDim.x + tx] = 0.0f;
    }

    __syncthreads();

    // Tile coordinates
    int tileCol = tx - radius;
    int tileRow = ty - radius;

    // Compute output for valid threads
    if (tileCol >= 0 && tileCol < blockDim.x - 2 * radius &&
        tileRow >= 0 && tileRow < blockDim.y - 2 * radius) {

        if (col >= 0 && col < ncols && row >= 0 && row < nrows) {

            // Zero out topmost and bottommost rows
            if (row < radius || row >= nrows - radius) {
                imgout[row * ncols + col] = 0.0f;
                return;
            }

            // Optimized convolution
            float sum = 0.0f;
#pragma unroll 8
            for (int k = 0; k < kernel_width; k++) {
                int sharedRow = tileRow + k;
                sum += sdata[sharedRow * blockDim.x + tx] * skernel[kernel_width - 1 - k];
            }

            imgout[row * ncols + col] = sum;
        }
    }
}

/*********************************************************************
 * _convolveSeparate_cuda - OPTIMIZED with memory pool
 */
static void _convolveSeparate_cuda(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout,
    cudaStream_t stream)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    size_t img_size = ncols * nrows * sizeof(float);

    // Initialize or resize pool if needed
    if (!g_conv_pool.initialized || g_conv_pool.img_capacity < img_size) {
        if (g_conv_pool.initialized) {
            cudaFree(g_conv_pool.d_imgin);
            cudaFree(g_conv_pool.d_imgout);
            cudaFree(g_conv_pool.d_tmpimg);
            cudaFree(g_conv_pool.d_kernel);
        }
        cudaCheck(cudaMalloc(&g_conv_pool.d_imgin, img_size));
        cudaCheck(cudaMalloc(&g_conv_pool.d_imgout, img_size));
        cudaCheck(cudaMalloc(&g_conv_pool.d_tmpimg, img_size));
        cudaCheck(cudaMalloc(&g_conv_pool.d_kernel, MAX_KERNEL_WIDTH * sizeof(float)));
        g_conv_pool.img_capacity = img_size;
        g_conv_pool.initialized = true;
    }

    // Initialize persistent kernel storage if needed
    if (!kernels_on_device) {
        cudaCheck(cudaMalloc(&d_gauss_kernel_data, MAX_KERNEL_WIDTH * sizeof(float)));
        cudaCheck(cudaMalloc(&d_gaussderiv_kernel_data, MAX_KERNEL_WIDTH * sizeof(float)));
        kernels_on_device = true;
    }

    // Use persistent buffers from pool
    float* d_imgin = g_conv_pool.d_imgin;
    float* d_tmpimg = g_conv_pool.d_tmpimg;
    float* d_imgout = g_conv_pool.d_imgout;
    float* d_kernel = g_conv_pool.d_kernel;

    // Async copy input image
    cudaCheck(cudaMemcpyAsync(d_imgin, imgin->data, img_size,
        cudaMemcpyHostToDevice, stream));

    // Configure kernel dimensions
    int radius = horiz_kernel.width / 2;
    dim3 blockDim(32, 32);
    int out_tile_dim = blockDim.x - 2 * radius;
    dim3 gridDim((ncols + out_tile_dim - 1) / out_tile_dim,
        (nrows + out_tile_dim - 1) / out_tile_dim);

    // OPTIMIZED: kernel data included in shared memory size
    size_t shared_mem = blockDim.x * blockDim.y * sizeof(float) +
        horiz_kernel.width * sizeof(float);

    // === HORIZONTAL CONVOLUTION ===
    cudaCheck(cudaMemcpyAsync(d_kernel, horiz_kernel.data,
        horiz_kernel.width * sizeof(float),
        cudaMemcpyHostToDevice, stream));

    convolveImageHorizKernel_SM << <gridDim, blockDim, shared_mem, stream >> > (
        d_imgin, d_kernel, horiz_kernel.width, ncols, nrows, d_tmpimg);
    cudaCheck(cudaGetLastError());

    // === VERTICAL CONVOLUTION ===
    cudaCheck(cudaMemcpyAsync(d_kernel, vert_kernel.data,
        vert_kernel.width * sizeof(float),
        cudaMemcpyHostToDevice, stream));

    convolveImageVertKernel_SM << <gridDim, blockDim, shared_mem, stream >> > (
        d_tmpimg, d_kernel, vert_kernel.width, ncols, nrows, d_imgout);
    cudaCheck(cudaGetLastError());

    // === ASYNC COPY RESULT BACK ===
    cudaCheck(cudaMemcpyAsync(imgout->data, d_imgout, img_size,
        cudaMemcpyDeviceToHost, stream));
}

// Performance optimization: Batch process multiple convolutions
void _convolveSeparate_batch(
    _KLT_FloatImage* imgs_in,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage* imgs_out,
    int batch_size)
{
    if (batch_size == 0) return;
    if (batch_size == 1) {
        cudaStream_t stream = _getStream(0);
        _convolveSeparate_cuda(imgs_in[0], horiz_kernel, vert_kernel, imgs_out[0], stream);
        cudaStreamSynchronize(stream);
        return;
    }

    // For multiple images, use parallel streams
    cudaStream_t* streams = (cudaStream_t*)malloc(batch_size * sizeof(cudaStream_t));
    for (int i = 0; i < batch_size; i++) {
        streams[i] = _getStream(i % 4);
    }

    // Launch all convolutions in parallel
    for (int i = 0; i < batch_size; i++) {
        _convolveSeparate_cuda(imgs_in[i], horiz_kernel, vert_kernel,
            imgs_out[i], streams[i]);
    }

    // Synchronize all
    for (int i = 0; i < batch_size; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    free(streams);
}

// Prefetch next image while processing current
void _prefetchImage(float* d_buffer, float* h_data, size_t size, cudaStream_t stream) {
    cudaMemcpyAsync(d_buffer, h_data, size, cudaMemcpyHostToDevice, stream);
}
	
void _KLTComputeGradients(
    _KLT_FloatImage img, float sigma,
    _KLT_FloatImage gradx, _KLT_FloatImage grady)
{
    assert(gradx->ncols >= img->ncols);
    assert(gradx->nrows >= img->nrows);
    assert(grady->ncols >= img->ncols);
    assert(grady->nrows >= img->nrows);

    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    // Use stream pool for parallel execution
    cudaStream_t stream1 = _getStream(0);
    cudaStream_t stream2 = _getStream(1);

    // Compute gradx and grady in parallel streams
    _convolveSeparate_cuda(img, gaussderiv_kernel, gauss_kernel, gradx, stream1);
    _convolveSeparate_cuda(img, gauss_kernel, gaussderiv_kernel, grady, stream2);

    // Synchronize both streams
    cudaCheck(cudaStreamSynchronize(stream1));
    cudaCheck(cudaStreamSynchronize(stream2));
}

void _KLTComputeSmoothedImage(
    _KLT_FloatImage img, float sigma, _KLT_FloatImage smooth)
{
    assert(smooth->ncols >= img->ncols);
    assert(smooth->nrows >= img->nrows);

    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    // Use dedicated stream for smoothing
    cudaStream_t stream = _getStream(2);
    _convolveSeparate_cuda(img, gauss_kernel, gauss_kernel, smooth, stream);
    cudaCheck(cudaStreamSynchronize(stream));
}


