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

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


#define MAX_KERNEL_WIDTH 	71

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

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols*nrows;
  float *ptrout = floatimg->data;

  /* Output image must be large enough to hold result */
  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)  *ptrout++ = (float) *img++;
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
 * _convolveImageHoriz
 */

//this function it blurs or filters the image along the x-direction (left to right) using a 1D kernel.
//static void _convolveImageHoriz(
//  _KLT_FloatImage imgin, 
//  ConvolutionKernel kernel,
//  _KLT_FloatImage imgout)
//{
//  float *ptrrow = imgin->data;           /* Points to row's first pixel */
//  register float *ptrout = imgout->data, /* Points to next output pixel */
//    *ppp;
//  register float sum;
//  register int radius = kernel.width / 2;
//  register int ncols = imgin->ncols, nrows = imgin->nrows;
//  register int i, j, k;
//
//  /* Kernel width must be odd */
//  assert(kernel.width % 2 == 1);
//
//  /* Must read from and write to different images */
//  assert(imgin != imgout);
//
//  /* Output image must be large enough to hold result */
//  assert(imgout->ncols >= imgin->ncols);
//  assert(imgout->nrows >= imgin->nrows);
//
//  /* For each row, do ... */
//  for (j = 0 ; j < nrows ; j++)  {
//
//    /* Zero leftmost columns */
//    for (i = 0 ; i < radius ; i++)
//      *ptrout++ = 0.0;
//
//    /* Convolve middle columns with kernel */
//    for ( ; i < ncols - radius ; i++)  {
//      ppp = ptrrow + i - radius;
//      sum = 0.0;
//      for (k = kernel.width-1 ; k >= 0 ; k--)
//        sum += *ppp++ * kernel.data[k];
//      *ptrout++ = sum;
//    }
//
//    /* Zero rightmost columns */
//    for ( ; i < ncols ; i++)
//      *ptrout++ = 0.0;
//
//    ptrrow += ncols;
//  }
//}
//using shared memory
__global__ void convolveImageHorizKernel_SM(
    const float* imgin,
    const float* kernel_data,
    int kernel_width,
    int ncols,
    int nrows,
    float* imgout)
{
    extern __shared__ float sdata[]; // shared memory for image row segment

    int tx = threadIdx.x;             // thread column within block
    int ty = threadIdx.y;             // thread row within block
    int i  = blockIdx.x * blockDim.x + tx; // global column
    int j  = blockIdx.y * blockDim.y + ty; // global row

    if (j >= nrows) return;  // outside image vertically

    int radius = kernel_width / 2;
    int row_start = j * ncols;

    // global index for this thread’s pixel
    int global_idx = row_start + i;

    // Shared memory tile width = blockDim.x + 2*radius
    int tile_width = blockDim.x + 2 * radius;
    int shared_idx = tx + radius; // position in shared memory

    // 1. Load main tile (each thread loads one pixel)
    if (i < ncols)
        sdata[shared_idx] = imgin[global_idx];
    else
        sdata[shared_idx] = 0.0f;

    // 2. Load left halo
    if (tx < radius) {
        int left_idx = i - radius;
        sdata[tx] = (left_idx >= 0) ? imgin[row_start + left_idx] : 0.0f;
    }

    // 3. Load right halo
    if (tx >= blockDim.x - radius) {
        int right_idx = i + radius;
        int s_right = shared_idx + radius;
        if (s_right < tile_width)
            sdata[s_right] = (right_idx < ncols) ? imgin[row_start + right_idx] : 0.0f;
    }

    __syncthreads();  // wait for all threads to fill the tile

    // 4. Now perform convolution using shared memory
    if (i < radius || i >= ncols - radius) {
        if (i < ncols)
            imgout[global_idx] = 0.0f; // boundary handling
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < kernel_width; k++) {
        sum += sdata[shared_idx - radius + k] * kernel_data[k];
    }

    imgout[global_idx] = sum;
}


// version of kernel not using shared mem
__global__ void convolveImageHorizKernel(
    const float* imgin,
    const float* kernel_data,
    int kernel_width,
    int ncols,
    int nrows,
    float* imgout)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // column index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index

    if (i >= ncols || j >= nrows)
        return;

    int radius = kernel_width / 2;
    int out_idx = j * ncols + i;

    // Zero leftmost columns
    if (i < radius) {
        imgout[out_idx] = 0.0f;
        return;
    }

    // Zero rightmost columns
    if (i >= ncols - radius) {
        imgout[out_idx] = 0.0f;
        return;
    }

    // Convolve middle columns with kernel
    float sum = 0.0f;
    int row_start = j * ncols;

    for (int k = kernel_width - 1; k >= 0; k--) {
        int ppp_idx = row_start + i - radius + (kernel_width - 1 - k);
        sum += imgin[ppp_idx] * kernel_data[k];
    }

    imgout[out_idx] = sum;
}




// CUDA version of _convolveImageHoriz
static void _convolveImageHoriz_cuda(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout, 
    cudaStream_t stream)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    size_t img_size = ncols * nrows * sizeof(float);
    size_t kernel_size = kernel.width * sizeof(float);

    // Allocate device memory
    float* d_imgin, * d_imgout, * d_kernel;
    cudaMalloc(&d_imgin, img_size);
    cudaMalloc(&d_imgout, img_size);
    cudaMalloc(&d_kernel, kernel_size);

    // Copy data to device
    cudaMemcpy(d_imgin, imgin->data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data, kernel_size, cudaMemcpyHostToDevice);

    // Configure and launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ncols + blockDim.x - 1) / blockDim.x,
        (nrows + blockDim.y - 1) / blockDim.y);

    convolveImageHorizKernel_SM << <gridDim, blockDim, 0, stream >> > (
        d_imgin, d_kernel, kernel.width, ncols, nrows, d_imgout);

    // Copy result back to host
    cudaMemcpy(imgout->data, d_imgout, img_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_imgin);
    cudaFree(d_imgout);
    cudaFree(d_kernel);
}

/*********************************************************************
 * _convolveImageVert
 */

//static void _convolveImageVert(
//  _KLT_FloatImage imgin,
//  ConvolutionKernel kernel,
//  _KLT_FloatImage imgout)
//{
//  float *ptrcol = imgin->data;            /* Points to row's first pixel */
//  register float *ptrout = imgout->data,  /* Points to next output pixel */
//    *ppp;
//  register float sum;
//  register int radius = kernel.width / 2;
//  register int ncols = imgin->ncols, nrows = imgin->nrows;
//  register int i, j, k;
//
//  /* Kernel width must be odd */
//  assert(kernel.width % 2 == 1);
//
//  /* Must read from and write to different images */
//  assert(imgin != imgout);
//
//  /* Output image must be large enough to hold result */
//  assert(imgout->ncols >= imgin->ncols);
//  assert(imgout->nrows >= imgin->nrows);
//
//  /* For each column, do ... */
//  for (i = 0 ; i < ncols ; i++)  {
//
//    /* Zero topmost rows */
//    for (j = 0 ; j < radius ; j++)  {
//      *ptrout = 0.0;
//      ptrout += ncols;
//    }
//
//    /* Convolve middle rows with kernel */
//    for ( ; j < nrows - radius ; j++)  {
//      ppp = ptrcol + ncols * (j - radius);
//      sum = 0.0;
//      for (k = kernel.width-1 ; k >= 0 ; k--)  {
//        sum += *ppp * kernel.data[k];
//        ppp += ncols;
//      }
//      *ptrout = sum;
//      ptrout += ncols;
//    }
//
//    /* Zero bottommost rows */
//    for ( ; j < nrows ; j++)  {
//      *ptrout = 0.0;
//      ptrout += ncols;
//    }
//
//    ptrcol++;
//    ptrout -= nrows * ncols - 1;
//  }
//}

//cuda kernel with SM 
__global__ void convolveImageVertKernel_SM(
    const float* imgin,
    const float* kernel_data,
    int kernel_width,
    int ncols,
    int nrows,
    float* imgout)
{
    // Shared memory tile for a column segment
    extern __shared__ float sdata[];

    int tx = threadIdx.x;  // column index within block
    int ty = threadIdx.y;  // row index within block

    int i = blockIdx.x * blockDim.x + tx; // global column
    int j = blockIdx.y * blockDim.y + ty; // global row

    if (i >= ncols) return; // out of horizontal bounds

    int radius = kernel_width / 2;
    int tile_height = blockDim.y + 2 * radius;

    // Each thread copies one pixel from global memory to shared memory
    // Compute global and shared indices
    int global_idx = j * ncols + i;
    int shared_idx = ty + radius; // vertical offset in shared memory

    // 1️⃣ Load main tile pixels
    if (j < nrows)
        sdata[shared_idx * blockDim.x + tx] = imgin[global_idx];
    else
        sdata[shared_idx * blockDim.x + tx] = 0.0f;

    // 2️⃣ Load top halo
    if (ty < radius) {
        int top_j = j - radius;
        int top_shared = ty;
        sdata[top_shared * blockDim.x + tx] = (top_j >= 0)
            ? imgin[top_j * ncols + i]
            : 0.0f;
    }

    // 3️⃣ Load bottom halo
    if (ty >= blockDim.y - radius) {
        int bottom_j = j + radius;
        int bottom_shared = shared_idx + radius;
        if (bottom_shared < tile_height)
            sdata[bottom_shared * blockDim.x + tx] = (bottom_j < nrows)
                ? imgin[bottom_j * ncols + i]
                : 0.0f;
    }

    __syncthreads();

    // 4️⃣ Handle image borders
    if (j < radius || j >= nrows - radius) {
        if (i < ncols)
            imgout[global_idx] = 0.0f;
        return;
    }

    // 5️⃣ Convolve vertically using shared memory
    float sum = 0.0f;
    for (int k = 0; k < kernel_width; k++) {
        int s_row = shared_idx - radius + k;
        sum += sdata[s_row * blockDim.x + tx] * kernel_data[k];
    }

    imgout[global_idx] = sum;
}

//cuda kernel without SM 
// CUDA kernel for vertical convolution
__global__ void convolveImageVertKernel(
    const float* imgin,
    const float* kernel_data,
    int kernel_width,
    int ncols,
    int nrows,
    float* imgout)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // column index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    
    if (i >= ncols || j >= nrows)
        return;
    
    int radius = kernel_width / 2;
    int out_idx = j * ncols + i;
    
    // Zero topmost rows
    if (j < radius) {
        imgout[out_idx] = 0.0f;
        return;
    }
    
    // Zero bottommost rows
    if (j >= nrows - radius) {
        imgout[out_idx] = 0.0f;
        return;
    }
    
    // Convolve middle rows with kernel
    float sum = 0.0f;
    
    for (int k = kernel_width - 1; k >= 0; k--) {
        int ppp_idx = (j - radius + (kernel_width - 1 - k)) * ncols + i;
        sum += imgin[ppp_idx] * kernel_data[k];
    }
    
    imgout[out_idx] = sum;
}

// CUDA version of _convolveImageVert
static void _convolveImageVert_cuda(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout, 
    cudaStream_t stream)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    size_t img_size = ncols * nrows * sizeof(float);
    size_t kernel_size = kernel.width * sizeof(float);

    // Allocate device memory
    float* d_imgin, * d_imgout, * d_kernel;
    cudaMalloc(&d_imgin, img_size);
    cudaMalloc(&d_imgout, img_size);
    cudaMalloc(&d_kernel, kernel_size);

    // Copy data to device
    cudaMemcpy(d_imgin, imgin->data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data, kernel_size, cudaMemcpyHostToDevice);

    // Configure and launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((ncols + blockDim.x - 1) / blockDim.x,
        (nrows + blockDim.y - 1) / blockDim.y);

    convolveImageVertKernel_SM<< <gridDim, blockDim, 0, stream>> > (
        d_imgin, d_kernel, kernel.width, ncols, nrows, d_imgout);

    // Copy result back to host
    cudaMemcpy(imgout->data, d_imgout, img_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_imgin);
    cudaFree(d_imgout);
    cudaFree(d_kernel);
}


/*********************************************************************
 * _convolveSeparate
 */

static void _convolveSeparate(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout)
{
    /* Create temporary image */
    _KLT_FloatImage tmpimg;
    tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
    
    /* Do convolution */
    _convolveImageHoriz_cuda(imgin, horiz_kernel, tmpimg);
    _convolveImageVert_cuda(tmpimg, vert_kernel, imgout);
    
    /* Free memory */
    _KLTFreeFloatImage(tmpimg);
}

//use a stream 
static void _convolveSeparate_cuda(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout,
    cudaStream_t stream)
{
    _KLT_FloatImage tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

    _convolveImageHoriz_cuda(imgin, horiz_kernel, tmpimg, stream);
    _convolveImageVert_cuda(tmpimg, vert_kernel, imgout, stream);

    _KLTFreeFloatImage(tmpimg);
}


/*********************************************************************
 * _KLTComputeGradients
 */

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
				
  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
	
  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);

}
	
//use streaming 
void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  // Create two CUDA streams
  cudaStream_t streamX, streamY;
  cudaStreamCreate(&streamX);
  cudaStreamCreate(&streamY);

  // Launch both convolutions in parallel streams
  _convolveSeparate_cuda(img, gaussderiv_kernel, gauss_kernel, gradx, streamX);
  _convolveSeparate_cuda(img, gauss_kernel, gaussderiv_kernel, grady, streamY);

  // Wait for both to finish
  cudaStreamSynchronize(streamX);
  cudaStreamSynchronize(streamY);

  // Cleanup
  cudaStreamDestroy(streamX);
  cudaStreamDestroy(streamY);
}



/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}


