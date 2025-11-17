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

//this function it blurs or filters the image along the x-direction (left to right) using a 1D kernel.
//
static void _convolveImageHoriz(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
    float* ptrrow = imgin->data;
    float* ptrout = imgout->data;
    int radius = kernel.width / 2;
    int ncols = imgin->ncols, nrows = imgin->nrows;
    int i, j, k;

    assert(kernel.width % 2 == 1);
    assert(imgin != imgout);
    assert(imgout->ncols >= imgin->ncols);
    assert(imgout->nrows >= imgin->nrows);

    int total_size = ncols * nrows;
    int kernel_size = kernel.width;

    /* Optimized: Use gang and vector parallelism with better collapse */
#pragma acc parallel loop gang vector(128) \
          copyin(ptrrow[0:total_size], kernel.data[0:kernel_size]) \
          copyout(ptrout[0:total_size])
    for (j = 0; j < nrows; j++) {
        int row_offset = j * ncols;

#pragma acc loop vector
        for (i = 0; i < ncols; i++) {
            int out_idx = row_offset + i;

            if (i < radius || i >= ncols - radius) {
                ptrout[out_idx] = 0.0;
            }
            else {
                float sum = 0.0;
                int start_idx = row_offset + i - radius;

#pragma acc loop seq
                for (k = kernel_size - 1; k >= 0; k--) {
                    sum += ptrrow[start_idx + (kernel_size - 1 - k)] * kernel.data[k];
                }

                ptrout[out_idx] = sum;
            }
        }
    }
}


/*********************************************************************
 * _convolveImageVert
 */

static void _convolveImageVert(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
    float* ptrcol = imgin->data;
    float* ptrout = imgout->data;
    int radius = kernel.width / 2;
    int ncols = imgin->ncols, nrows = imgin->nrows;
    int i, j, k;

    assert(kernel.width % 2 == 1);
    assert(imgin != imgout);
    assert(imgout->ncols >= imgin->ncols);
    assert(imgout->nrows >= imgin->nrows);

    int total_size = ncols * nrows;
    int kernel_size = kernel.width;

    /* Optimized: Better memory coalescing by processing rows in parallel */
#pragma acc parallel loop gang vector(128) \
          copyin(ptrcol[0:total_size], kernel.data[0:kernel_size]) \
          copyout(ptrout[0:total_size])
    for (j = 0; j < nrows; j++) {

#pragma acc loop vector
        for (i = 0; i < ncols; i++) {
            int out_idx = j * ncols + i;

            if (j < radius || j >= nrows - radius) {
                ptrout[out_idx] = 0.0;
            }
            else {
                float sum = 0.0;
                int start_row = j - radius;

#pragma acc loop seq
                for (k = kernel_size - 1; k >= 0; k--) {
                    int in_idx = (start_row + (kernel_size - 1 - k)) * ncols + i;
                    sum += ptrcol[in_idx] * kernel.data[k];
                }

                ptrout[out_idx] = sum;
            }
        }
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

    int img_size = imgin->ncols * imgin->nrows;

    /* Keep data on GPU between horizontal and vertical convolutions */
#pragma acc data copyin(imgin->data[0:img_size]) \
                   create(tmpimg->data[0:img_size]) \
                   copyout(imgout->data[0:img_size])
    {
        /* Horizontal convolution - data stays on GPU */
        {
            float *in_ptr  = imgin->data;
            float *tmp_ptr = tmpimg->data;
            int radius_h   = horiz_kernel.width / 2;
            int ncols      = imgin->ncols;
            int nrows      = imgin->nrows;

            /* Parallel region on data already present */
#pragma acc parallel present(in_ptr[0:img_size], tmp_ptr[0:img_size])
            {
#pragma acc loop gang vector
                for (int j = 0; j < nrows; j++) {
                    int row_offset = j * ncols;

//#pragma acc loop vector
                    for (int i = 0; i < ncols; i++) {
                        int out_idx = row_offset + i;
                        if (i < radius_h || i >= ncols - radius_h) {
                            tmp_ptr[out_idx] = 0.0f;
                        } else {
                            float sum = 0.0f;
                            int start_idx = row_offset + i - radius_h;

//#pragma acc loop seq
                            for (int k = horiz_kernel.width - 1; k >= 0; k--) {
                                sum += in_ptr[start_idx + (horiz_kernel.width - 1 - k)] *
                                       horiz_kernel.data[k];
                            }
                            tmp_ptr[out_idx] = sum;
                        }
                    }
                }
            } /* end parallel */
        }

        /* Vertical convolution - data already on GPU */
        {
            float *tmp_ptr2 = tmpimg->data;
            float *out_ptr  = imgout->data;
            int radius_v    = vert_kernel.width / 2;
            int ncols       = tmpimg->ncols;
            int nrows       = tmpimg->nrows;

#pragma acc parallel present(tmp_ptr2[0:img_size], out_ptr[0:img_size])
            {
#pragma acc loop gang vector
                for (int j = 0; j < nrows; j++) {

#pragma acc loop vector
                    for (int i = 0; i < ncols; i++) {
                        int out_idx = j * ncols + i;
                        if (j < radius_v || j >= nrows - radius_v) {
                            out_ptr[out_idx] = 0.0f;
                        } else {
                            float sum      = 0.0f;
                            int start_row  = j - radius_v;

#pragma acc loop seq
                            for (int k = vert_kernel.width - 1; k >= 0; k--) {
                                int in_idx = (start_row + (vert_kernel.width - 1 - k)) * ncols + i;
                                sum += tmp_ptr2[in_idx] * vert_kernel.data[k];
                            }
                            out_ptr[out_idx] = sum;
                        }
                    }
                }
            } /* end parallel */
        }
    } /* end data */

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



