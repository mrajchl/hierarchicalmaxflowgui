/***************************************************************************/
/* Name:      cmf3DCut_std                              

   Authors:
   Martin Rajchl   mrajchl@imaging.robarts.ca
   Jing Yuan       cn.yuanjing@googlemail.com

   Description:
   Parallelized Continuous MaxFlow GraphCuts using CUDA

   For more details, see the report:
   Jing Yuan et al.
   "A Study on Continuous Max-Flow and Min-Cut Approaches"
   CVPR, 2010

   Jing Yuan, et al.
   "A Study on Continuous Max-Flow and Min-Cut Approaches: Part I: Binary Labeling"
   UCLA CAM, 2010

   Date: 2011/09/29

*/
/***************************************************************************/

/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda.h"

#include "cmf3DCut_std_kernels.cu"
#include "cmf3DCut_std.h"

/* Defines */
#define YES 0
#define NO 1
#define BLOCK_SIZE 512

#define PI 3.1415926

#define MAX(a,b) ( a > b ? a : b )
#define MIN(a,b) ( a <= b ? a : b )
#define SIGN(x) ( x >= 0.0 ? 1.0 : -1.0 )
#define ABS(x) ( (x) > 0.0 ? x : -(x) )
#define SQR(x) (x)*(x)

#ifndef HAVE_RINT 
#define rint(A) floor((A)+(((A) < 0)? -0.5 : 0.5)) 
#endif



float SQRT(float number) {
  long i;
  float x, y;
  const float f = 1.5F;

  x = number * 0.5F;
  y  = number;
  i  = * ( long * ) &y;
  i  = 0x5f3759df - ( i >> 1 );
  y  = * ( float * ) &i;
  y  = y * ( f - ( x * y * y ) );
  y  = y * ( f - ( x * y * y ) );
  return number * y;
}


void cudasafe( cudaError_t error, char* message)
{
  if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %i\n",message,error);return; }
}

void cmf3DCut_std(float *h_lambda,
		    float *h_fCs,
		    float *h_fCt,
		    float *pars){


  /* Declaration of host variables */
  float fps;

  /* Timing */
  cudaEvent_t start, stop; 
  float time;
  /* Size */
  int iNx = (int) pars[0];
  int iNy = (int) pars[1];
  int iNz = (int) pars[2];

  /*  Declaration of host variables */
  float *h_gk, *h_cvg, *h_FPS;

  float *h_bx, *h_by, *h_bz, *h_ps, *h_pt;

  /* Declaration of device variables */
  float   *d_lambda, *d_fCs, *d_fCt, *d_bx, *d_by, *d_bz, *d_ps, *d_pt, *d_gk, *d_FPS;

  /* Choice of region segmentation model */
  float alpha = (float) pars[3]; /* penalty parameter alpha */
  int iNbIters = (int) pars[4]; /* total number of iterations */
  float fError = (float) pars[5]; /* error criterion */
  float cc = (float) pars[6]; /* cc for ALM */
  float steps = (float) pars[7]; /* steps for each iteration */
	
  /* CUDA event-based timer start */
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );  


  /* 
   *Parameter Setting
   * [0] : number of columns 
   * [1] : number of rows
   * [2] : number of slices
   * [3] : penalty parameter alpha
   * [4] : total iteration number
   * [5] : error criterion
   * [6] : cc for ALM
   * [7] : steps for each iteration
   */

  /* Host memory allocation */
  h_bx =   (float *) calloc( (unsigned)(iNy*iNx*iNz), sizeof(float) );
  h_by =   (float *) calloc( (unsigned)(iNy*iNx*iNz), sizeof(float) );
  h_bz =   (float *) calloc( (unsigned)(iNy*iNx*iNz), sizeof(float) );
  h_ps =   (float *) calloc( (unsigned)(iNy*iNx*iNz), sizeof(float) );
  h_pt =   (float *) calloc( (unsigned)(iNy*iNx*iNz), sizeof(float) );

  h_gk =   (float *) calloc( (unsigned)(iNx*iNy*iNz), sizeof(float) );
  h_cvg =  (float *) calloc( (unsigned)iNbIters+1, sizeof(double) );
  h_FPS =  (float *) calloc( (unsigned)(iNy*iNx*iNz), sizeof(float) );

  if (!h_bx   ||
      !h_by   ||
      !h_bz   ||
      !h_ps   ||
      !h_pt   ||

      !h_FPS  ||
      !h_gk   ||
      !h_cvg)
    printf("Host memory allocation failure\n");
 


  /* Device memory allocation */
  cudasafe( cudaMalloc( (void**) &d_lambda, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_fCs, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_fCt, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_bx, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc"); 
  cudasafe( cudaMalloc( (void**) &d_by, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_bz, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_ps, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_pt, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");

  cudasafe( cudaMalloc( (void**) &d_gk, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_FPS, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");

  // cudaMalloc( (void**) &d_lambda, sizeof(float)*(unsigned)(iNx*iNy*iNz));
  // cudaMalloc( (void**) &d_fCs, sizeof(float)*(unsigned)(iNx*iNy*iNz));
  // cudaMalloc( (void**) &d_fCt, sizeof(float)*(unsigned)(iNx*iNy*iNz));
  // cudaMalloc( (void**) &d_bx, sizeof(float)*(unsigned)(iNx*iNy*iNz)); 
  // cudaMalloc( (void**) &d_by, sizeof(float)*(unsigned)(iNx*iNy*iNz));
  // cudaMalloc( (void**) &d_bz, sizeof(float)*(unsigned)(iNx*iNy*iNz));
  // cudaMalloc( (void**) &d_ps, sizeof(float)*(unsigned)(iNx*iNy*iNz));
  // cudaMalloc( (void**) &d_pt, sizeof(float)*(unsigned)(iNx*iNy*iNz));
  
  // cudaMalloc( (void**) &d_gk, sizeof(float)*(unsigned)(iNx*iNy*iNz));
  // cudaMalloc( (void**) &d_FPS, sizeof(float)*(unsigned)(iNx*iNy*iNz));

  /* Memcpy image buffers from Host to Device */
  cudasafe( cudaMemcpy( d_lambda, h_lambda, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice), "cudaMemcpy lambda");
  cudasafe( cudaMemcpy( d_fCs, h_fCs, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice), "cudaMemcpy fCs");
  cudasafe( cudaMemcpy( d_fCt, h_fCt, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice), "cudaMemcpy fCt");
  cudasafe( cudaMemcpy( d_bx, h_bx, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice), "cudaMemcpy bx");
  cudasafe( cudaMemcpy( d_by, h_by, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice), "cudaMemcpy by");
  cudasafe( cudaMemcpy( d_bz, h_bz, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice), "cudaMemcpy bz");
  cudasafe( cudaMemcpy( d_ps, h_ps, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice), "cudaMemcpy ps");
  cudasafe( cudaMemcpy( d_pt, h_pt, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice), "cudaMemcpy pt");

  cudasafe( cudaMemcpy( d_gk, h_gk, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice), "cudaMemcpy gk");
  cudasafe( cudaMemcpy( d_FPS, h_FPS,   sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice), "cudaMemcpy FPS");

  // cudaMemcpy( d_lambda, h_lambda, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice);
  // cudaMemcpy( d_fCs, h_fCs, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice);
  // cudaMemcpy( d_fCt, h_fCt, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice);
  // cudaMemcpy( d_bx, h_bx, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice);
  // cudaMemcpy( d_by, h_by, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice);
  // cudaMemcpy( d_bz, h_bz, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice);
  // cudaMemcpy( d_ps, h_ps, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice);
  // cudaMemcpy( d_pt, h_pt, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice);
  
  // cudaMemcpy( d_gk, h_gk, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice);
  // cudaMemcpy( d_FPS, h_FPS,   sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice);



  /* Main iterations */

  /* Setup GPU execution parameters */
  dim3 threads(BLOCK_SIZE);
  dim3 grid( ((iNx*iNy*iNz)/threads.x +( (!(iNx*iNy*iNz)%threads.x) ? 0:1)) );

  printf (" Grid: %d, Threads: %d \n", grid.x, threads.x);


	
  int iNI;
  for( iNI = 0; iNI<iNbIters; iNI++ ) 
    {	

      /* Solver kernels */
      updateP1<<<grid, threads>>>(d_gk,
      				  d_bx, d_by, d_bz,
      				  d_ps, d_pt,
      				  d_lambda,
      				  cc,
      				  iNx, iNy, iNz);

      updateP<<<grid, threads>>>(d_bx, d_by, d_bz,
      				 steps,
      				 d_gk,
      				 iNx, iNy, iNz);

      // projStep1omega<<<grid, threads>>>(d_bx, d_by, d_bz,
      // d_omega,
      // 	d_gk,
      // 	iNx, iNy);

      projStep1alpha<<<grid, threads>>>(d_bx, d_by, d_bz,
      					alpha,
      					d_gk,
      					iNx, iNy, iNz);

      projStep2Total<<<grid, threads>>>(d_bx, d_by, d_bz,
      					d_gk,
      					iNx, iNy, iNz);

      updatePstMult<<<grid, threads>>>(d_gk,
      				       d_bx, d_by, d_bz,
      				       d_ps, d_pt,
      				       d_fCs, d_fCt,
      				       d_lambda,
      				       d_FPS,
      				       cc,
      				       iNx, iNy, iNz);


      /* Pixelwise error accumulation for convergence rate calculation */
      for(unsigned int errorAccumBlockSize = 1; errorAccumBlockSize < iNx*iNy*iNz; errorAccumBlockSize += errorAccumBlockSize){
      	dim3 errorAccumThreads(16);
      	dim3 errorAccumGrid( (iNx*iNy*iNz) / (2*errorAccumBlockSize*errorAccumThreads.x) );
      	if( (iNx*iNy*iNz) % (2*errorAccumBlockSize*errorAccumThreads.x) != 0) errorAccumGrid.x += 1;
      	errorAccumulation<<<errorAccumGrid, errorAccumThreads>>>(d_FPS, errorAccumBlockSize, iNx*iNy*iNz);
      }
   
      cudaMemcpy( &fps, d_FPS, sizeof(float), cudaMemcpyDeviceToHost);
		
      if ( fps <= (iNx*iNy*iNz) * fError && iNI > 10)
      	break;
     
      printf("cvg: %f ", fps / (iNx*iNy*iNz));
      printf("fps: %f \n", fps);

     }

  /* Memcpy relevant image data buffers back to host for next frame */
  cudasafe(  cudaMemcpy(h_lambda, d_lambda, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyDeviceToHost), "cudaMemcpy lambda d2h");
  //cudaMemcpy(h_lambda, d_lambda, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyDeviceToHost); 
   

  printf ("fps = %f\n",fps / (iNx*iNy*iNz));
  printf("Iterations: %d\n",iNI);

  /* Free memory */
  free( (float *) h_bx );
  free( (float *) h_by );
  free( (float *) h_bz );
  free( (float *) h_ps );
  free( (float *) h_pt );

  free( (float *) h_gk );
  free( (float *) h_FPS );
  free( (float *) h_cvg );

  /* Free GPU Memory */
  cudaFree(d_lambda);
  cudaFree(d_fCs);
  cudaFree(d_fCt);
  cudaFree(d_bx);
  cudaFree(d_by);
  cudaFree(d_bz);
  cudaFree(d_ps);
  cudaFree(d_pt);

  cudaFree(d_gk);
  cudaFree(d_FPS);


  /* CUDA event-based timer stop */
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  printf("CUDA event time = %.5f msec \n",time);


}
