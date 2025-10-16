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
 //runs on the kernel 
__global__ void convolveHorizontalKernel ( const float *d_imgin, //input image
int ncols, int nrows, //height and width 
const float *d_kernel, //device convolution array 
int kwidth, //kernel width  
float *d_imgout //output image 
) 
{ 
//compute indices
  int col = blockIdx.x * blockDim.x + threadIdx.x;   // x => column
  int row = blockIdx.y * blockDim.y + threadIdx.y;   // y => row
  
//boundary checking 
  if (col >= ncols || row >= nrows) return; 
  
//radius = half of kernel width and out index is our picture index in this thread for the output image  
  const int radius = kwidth / 2; 
  int out_idx = row * ncols + col;
  
//handle left and right pixels of the image
  if (col < radius || col > = (ncols - radius)) {
    d_imgout[out_idx] = 0.0f;  //make left and right columns 0. same as c code 
    return;
  }


  float sum = 0.0f; //init with zero 
  const int start = col - radius;   // leftmost pixel covered by kernel on this row
  int in_base = row * ncols + start; //
  
  // multiply input pixels by kernel weights 
  for (int k = 0; k < kwidth; ++k) {
    sum += d_imgin[in_base + k] * d_kernel[k];
  }
  
  d_imgout[out_idx] = sum;
}

//host function 
//copies data to device
//launches kernel
//copies data back 

void _convolveImageHoriz_cuda(
    const _KLT_FloatImage *imgin,    /* host input image */
    const ConvolutionKernel kernel,  /* host kernel */
    _KLT_FloatImage *imgout)         /* host output image */
{
  assert(kernel.width % 2 == 1); //check that kernel width is odd
  assert(imgin != imgout); //both input and output images should be distinct 
  assert(imgout->ncols >= imgin->ncols); //output image mustnt be smaller
  assert(imgout->nrows >= imgin->nrows); // ^^ 
  
//initialise const sizes and boundaries 
  const int ncols = imgin->ncols; 
  const int nrows = imgin->nrows;
  const size_t npixels = (size_t)ncols * (size_t)nrows;
  const size_t nbytes = npixels * sizeof(float);
  const int kw = kernel.width;
  const size_t kernel_bytes = kw * sizeof(float);
  
   // Allocate device buffers for input output image and convol array 
  float *d_imgin = NULL;
  float *d_imgout = NULL;
  float *d_kernel = NULL;
  
  //memory alloc with error check 
  cudaCheck(cudaMalloc((void**)&d_imgin, nbytes));
  cudaCheck(cudaMalloc((void**)&d_imgout, nbytes));
  cudaCheck(cudaMalloc((void**)&d_kernel, kernel_bytes));
  
  //copy data 
  //host to device 
  cudaCheck(cudaMemcpy(d_imgin, imgin->data, nbytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_kernel, kernel.data, kernel_bytes, cudaMemcpyHostToDevice));

//set launch configuration
//test config
  dim3 block(16, 16);
  dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);

//launch kernel
  convolveHorizKernel<<<gridDim, blockDim>>>(d_imgin, ncols, nrows, d_kernel, kw, d_imgout);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
  
  //copy result back to host 
  cudaCheck(cudaMemcpy(imgout->data, d_imgout, nbytes, cudaMemcpyDeviceToHost));

  //free device memory 
  cudaCheck(cudaFree(d_imgin));
  cudaCheck(cudaFree(d_imgout));
  cudaCheck(cudaFree(d_kernel));

}




//it blurs or filters the image along the x-direction (left to right) using a 1D kernel
//to parallelise use one thread for one pixel

/*
static void _convolveImageHoriz(
  _KLT_FloatImage imgin, //input image 
  ConvolutionKernel kernel, //the 1D convolution array of weights
  _KLT_FloatImage imgout) //output array 
{
  float *ptrrow = imgin->data;           // Points to row's first pixel 
  register float *ptrout = imgout->data, // Points to next output pixel 
    *ppp; //temporary pointer for use during convultion
  register float sum; //holds the convolution result for one pixel 
  register int radius = kernel.width / 2; //half of kernel width 
  register int ncols = imgin->ncols, nrows = imgin->nrows; //image width and height  
  register int i, j, k; //loop variables 

  // Kernel width must be odd 
  assert(kernel.width % 2 == 1); 

  // Must read from and write to different images //
  assert(imgin != imgout);

  // Output image must be large enough to hold result //
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  // For each row, do ... 
  for (j = 0 ; j < nrows ; j++)  { 

    // Zero leftmost columns 
    for (i = 0 ; i < radius ; i++)
      *ptrout++ = 0.0; //The kernel cannot fully overlap the image near the left edge (not enough pixels to the left), so these pixels are set to 0.

    // Convolve middle columns with kernel 
    for ( ; i < ncols - radius ; i++)  {
      ppp = ptrrow + i - radius;
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)
        sum += *ppp++ * kernel.data[k];
      *ptrout++ = sum;
    }

    // Zero rightmost columns 
    for ( ; i < ncols ; i++)
      *ptrout++ = 0.0;

    ptrrow += ncols;
  }
}
*/


/*********************************************************************
 * _convolveImageVert
 */

static void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *ptrcol = imgin->data;            /* Points to row's first pixel */
  register float *ptrout = imgout->data,  /* Points to next output pixel */
    *ppp;
  register float sum;
  register int radius = kernel.width / 2;
  register int ncols = imgin->ncols, nrows = imgin->nrows;
  register int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each column, do ... */
  for (i = 0 ; i < ncols ; i++)  {

    /* Zero topmost rows */
    for (j = 0 ; j < radius ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    /* Convolve middle rows with kernel */
    for ( ; j < nrows - radius ; j++)  {
      ppp = ptrcol + ncols * (j - radius);
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)  {
        sum += *ppp * kernel.data[k];
        ppp += ncols;
      }
      *ptrout = sum;
      ptrout += ncols;
    }

    /* Zero bottommost rows */
    for ( ; j < nrows ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    ptrcol++;
    ptrout -= nrows * ncols - 1;
  }
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
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);

  _convolveImageVert(tmpimg, vert_kernel, imgout);

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



