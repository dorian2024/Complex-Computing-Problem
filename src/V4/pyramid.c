/*********************************************************************
 * pyramid.c
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>		/* malloc() ? */
#include <string.h>		/* memset() ? */
#include <math.h>		/* */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"	/* for computing pyramid */
#include "pyramid.h"


/*********************************************************************
 *
 */

_KLT_Pyramid _KLTCreatePyramid(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels)
{
  _KLT_Pyramid pyramid;
  int nbytes = sizeof(_KLT_PyramidRec) +	
    nlevels * sizeof(_KLT_FloatImage *) +
    nlevels * sizeof(int) +
    nlevels * sizeof(int);
  int i;

  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTCreatePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

     
  /* Allocate memory for structure and set parameters */
  pyramid = (_KLT_Pyramid)  malloc(nbytes);
  if (pyramid == NULL)
    KLTError("(_KLTCreatePyramid)  Out of memory");
     
  /* Set parameters */
  pyramid->subsampling = subsampling;
  pyramid->nLevels = nlevels;
  pyramid->img = (_KLT_FloatImage *) (pyramid + 1);
  pyramid->ncols = (int *) (pyramid->img + nlevels);
  pyramid->nrows = (int *) (pyramid->ncols + nlevels);

  /* Allocate memory for each level of pyramid and assign pointers */
  for (i = 0 ; i < nlevels ; i++)  {
    pyramid->img[i] =  _KLTCreateFloatImage(ncols, nrows);
    pyramid->ncols[i] = ncols;  pyramid->nrows[i] = nrows;
    ncols /= subsampling;  nrows /= subsampling;
  }

  return pyramid;
}


/*********************************************************************
 *
 */

void _KLTFreePyramid(
  _KLT_Pyramid pyramid)
{
  int i;

  /* Free images */
  for (i = 0 ; i < pyramid->nLevels ; i++)
    _KLTFreeFloatImage(pyramid->img[i]);

  /* Free structure */
  free(pyramid);
}


/*********************************************************************
 *
 */

void _KLTComputePyramid(
    _KLT_FloatImage img,
    _KLT_Pyramid pyramid,
    float sigma_fact)
{
    _KLT_FloatImage currimg, tmpimg;
    int ncols = img->ncols, nrows = img->nrows;
    int subsampling = pyramid->subsampling;
    int subhalf = subsampling / 2;
    float sigma = subsampling * sigma_fact;
    int oldncols;
    int i, x, y;

    if (subsampling != 2 && subsampling != 4 &&
        subsampling != 8 && subsampling != 16 && subsampling != 32)
        KLTError("(_KLTComputePyramid)  Pyramid's subsampling must "
            "be either 2, 4, 8, 16, or 32");

    assert(pyramid->ncols[0] == img->ncols);
    assert(pyramid->nrows[0] == img->nrows);

    /* Copy original image to level 0 of pyramid */
    int base_size = ncols * nrows;
    memcpy(pyramid->img[0]->data, img->data, base_size * sizeof(float));

    /* Keep data on GPU across all pyramid levels */
#pragma acc data copyin(img->data[0:base_size])
    {
        currimg = img;
        for (i = 1; i < pyramid->nLevels; i++) {
            int curr_size = ncols * nrows;
            tmpimg = _KLTCreateFloatImage(ncols, nrows);

            /* Smooth image - this will use GPU */
            _KLTComputeSmoothedImage(currimg, sigma, tmpimg);

            /* Subsample on GPU */
            oldncols = ncols;
            ncols /= subsampling;
            nrows /= subsampling;
            int new_size = ncols * nrows;

            float* tmpimg_data = tmpimg->data;
            float* pyramid_data = pyramid->img[i]->data;

#pragma acc parallel loop gang vector collapse(2) \
              copyin(tmpimg_data[0:curr_size]) \
              copyout(pyramid_data[0:new_size])
            for (y = 0; y < nrows; y++) {
                for (x = 0; x < ncols; x++) {
                    pyramid_data[y * ncols + x] =
                        tmpimg_data[(subsampling * y + subhalf) * oldncols + (subsampling * x + subhalf)];
                }
            }

            /* Reassign current image */
            currimg = pyramid->img[i];
            _KLTFreeFloatImage(tmpimg);
        }
    }
}
 











