/*********************************************************************
 * convolve.c
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
    _KLT_FloatImage imgout)
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

    convolveImageHorizKernel << <gridDim, blockDim >> > (
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
    _KLT_FloatImage imgout)
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

    convolveImageVertKernel << <gridDim, blockDim >> > (
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


