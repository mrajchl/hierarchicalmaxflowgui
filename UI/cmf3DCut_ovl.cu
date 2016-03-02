/***************************************************************************/
/* Name:       cmf3DCut_ovl.cu                               
Date:          Nov 07, 2011                                                 

Authors:       Jing Yuan (cn.yuanjing@googlemail.com) 
Martin Rajchl (mrajchl@imaging.robarts.ca)  

Description:   Fast Max-Flow Implementation for multilayered min-cut
Inputs 
(C_t, para, alpha): C_t - the capacities of n flows
para 0,1 - rows, cols
para 2 - n: the label number
para 3 - the total number of iterations
para 4 - the error criterion
para 5 - cc for ALM
para 6 - steps for each iteration
alpha - the vector of penalty parameters with n-1 elements

Outputs
(u, erriter, num): u - the computed labeling result 
erriter - the error at each iteration
num - the iteration on stop

For more details, see the report:
Egil Bae, Jing Yuan, Xue-Cheng Tai and Yuri Boykov
"A FAST CONTINUOUS MAX-FLOW APPROACH TO NON-CONVEX MULTILABELING PROBLEMS" 
Submitted to Math. Comp. (AMS Journals)

*/
/***************************************************************************/

/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda.h"

#include "cmf3DCut_ovl_kernels.cu"
#include "cmf3DCut_ovl.h"

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

void cmf3DCut_ovl(float *h_Ct,
				  float *pars,
				  float *h_lambda,
				  float *h_pt,
				  float *h_u
				  ){



					  /*  Declaration of host variables */
					  float *h_gk, *h_cvg, *h_FPS, *h_div;
					  float *h_bx, *h_by, *h_bz;

					  float fps;

					  /* Declaration of device variables */
					  float *d_lambda, *d_Ct, *d_pt; 

					  float *d_u, *d_gk, *d_div;
					  float *d_bx, *d_by, *d_bz;

					  float *d_cvg, *d_FPS;

					  /* Timing */
					  cudaEvent_t start, stop; 
					  float time;

					  /* Size */
					  int iNx = (int) pars[0];
					  int iNy = (int) pars[1];
					  int iNz = (int) pars[2];
					  int iLab = (int)pars[3];

					  printf("x: %d y:%d z:%d iLab:%d",iNx, iNy, iNz, iLab);

					  /* Choice of region segmentation model */
					  int iNbIters = (int) pars[4]; /* total number of iterations */
					  float fError = (float) pars[5]; /* error criterion */
					  float cc = (float) pars[6]; /* cc for ALM */
					  float steps = (float) pars[7]; /* steps for each iteration */



					cudaSetDevice(3);



					  /* CUDA event-based timer start */
					  cudaEventCreate(&start);
					  cudaEventCreate(&stop);
					  cudaEventRecord( start, 0 );  


					  /* 
					  *Parameter Setting
					  * [0] : number of columns 
					  * [1] : number of rows
					  * [2] : number of slices
					  * [3] : array of alphas
					  * [4] : total iteration number
					  * [5] : error criterion
					  * [6] : cc for ALM
					  * [7] : steps for each iteration
					  */

					  /* Host memory allocation */
					  //h_tempu =   (float *) calloc( (unsigned)(iNy*iNx*iNz*(iLab-1)), sizeof(float) );
					  h_bx =   (float *) calloc( (unsigned)(iNy*iNx*iNz*(iLab-1)), sizeof(float) );
					  h_by =   (float *) calloc( (unsigned)(iNy*iNx*iNz*(iLab-1)), sizeof(float) );
					  h_bz =   (float *) calloc( (unsigned)(iNy*iNx*iNz*(iLab-1)), sizeof(float) );

					  h_gk =   (float *) calloc( (unsigned)(iNx*iNy*iNz*(iLab-1)), sizeof(float) );
					  h_div =   (float *) calloc( (unsigned)(iNx*iNy*iNz*(iLab-1)), sizeof(float) );
					  h_cvg =  (float *) calloc( (unsigned)iNbIters+1, sizeof(float) );
					  h_FPS =  (float *) calloc( (unsigned)(iNy*iNx*iNz*(iLab-1)), sizeof(float) );

					  if (!h_bx   ||
						  !h_by   ||
						  !h_bz   ||

						  !h_FPS  ||
						  !h_gk   ||
						  !h_div  ||
						  !h_cvg)
						  printf("Host memory allocation failure\n");



					  /* Device memory allocation */
					  cudasafe( cudaMalloc( (void**) &d_u, sizeof(float)*(unsigned)(iNx*iNy*iNz)*(iLab-1)), "cudaMalloc"); 
					  cudasafe( cudaMalloc( (void**) &d_lambda, sizeof(float)*(unsigned)(iLab-1)), "cudaMalloc");
					  cudasafe( cudaMalloc( (void**) &d_Ct, sizeof(float)*(unsigned)(iNx*iNy*iNz*iLab)), "cudaMalloc");
					  cudasafe( cudaMalloc( (void**) &d_pt, sizeof(float)*(unsigned)(iNx*iNy*iNz)*iLab), "cudaMalloc");

					  cudasafe( cudaMalloc( (void**) &d_bx, sizeof(float)*(unsigned)(iNx*iNy*iNz)*(iLab-1)), "cudaMalloc"); 
					  cudasafe( cudaMalloc( (void**) &d_by, sizeof(float)*(unsigned)(iNx*iNy*iNz)*(iLab-1)), "cudaMalloc");
					  cudasafe( cudaMalloc( (void**) &d_bz, sizeof(float)*(unsigned)(iNx*iNy*iNz)*(iLab-1)), "cudaMalloc");

					  cudasafe( cudaMalloc( (void**) &d_gk, sizeof(float)*(unsigned)(iNx*iNy*iNz)*(iLab-1)), "cudaMalloc");
					  cudasafe( cudaMalloc( (void**) &d_div, sizeof(float)*(unsigned)(iNx*iNy*iNz)*(iLab-1)), "cudaMalloc");
					  cudasafe( cudaMalloc( (void**) &d_FPS, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");


					  /* Memcpy image buffers from Host to Device */
					  cudasafe(cudaMemcpy(d_u, h_u, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab-1)), cudaMemcpyHostToDevice),"cudaMemcpy u");
					  cudasafe(cudaMemcpy(d_lambda, h_lambda, sizeof(float)*(unsigned)(iLab-1), cudaMemcpyHostToDevice),"cudaMemcpy lambda");
					  cudasafe(cudaMemcpy(d_Ct, h_Ct, sizeof(float)*(unsigned)(iNx*iNy*iNz*iLab), cudaMemcpyHostToDevice),"cudaMemcpy Ct");
					  cudasafe(cudaMemcpy(d_pt, h_pt, sizeof(float)*(unsigned)(iNx*iNy*iNz*iLab), cudaMemcpyHostToDevice),"cudaMemcpy pt");

					  cudasafe(cudaMemcpy(d_bx, h_bx, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab-1)), cudaMemcpyHostToDevice),"cudaMemcpy bx");
					  cudasafe(cudaMemcpy(d_by, h_by, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab-1)), cudaMemcpyHostToDevice),"cudaMemcpy by");
					  cudasafe(cudaMemcpy(d_bz, h_bz, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab-1)), cudaMemcpyHostToDevice),"cudaMemcpy bz");

					  cudasafe(cudaMemcpy(d_gk, h_gk, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab-1)), cudaMemcpyHostToDevice),"cudaMemcpy gk");
					  cudasafe(cudaMemcpy(d_div, h_div, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab-1)), cudaMemcpyHostToDevice),"cudaMemcpy div");
					  cudasafe(cudaMemcpy(d_FPS, h_FPS,   sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice),"cudaMemcpy FPS");



					  /* Main iterations */

					  /* Setup GPU execution parameters */
					  dim3 threads(BLOCK_SIZE);
					  dim3 grid( ((iNx*iNy*iNz)/threads.x +( (!(iNx*iNy*iNz)%threads.x) ? 0:1)) );

					  printf (" Grid: %d, Threads: %d \n", grid.x, threads.x);

					
					  int iNI;
					  for( iNI = 0; iNI<iNbIters; iNI++ ) 
					  {	
						//  printf("Iteration %d\n", iNI);


						  /* Solver kernels */
						  updateP1<<<grid, threads>>>(d_gk, d_u, d_pt, d_div,
							  d_bx, d_by, d_bz,
							  cc,
							  iNx, iNy, iNz, iLab);

						  updateP<<<grid, threads>>>(d_bx, d_by, d_bz,
							  steps,
							  d_gk,
							  iNx, iNy, iNz, iLab);

						  projStep1alpha<<<grid, threads>>>(d_bx, d_by, d_bz,
							  d_lambda,
							  d_gk,
							  iNx, iNy, iNz, iLab);

						  projStep2Total<<<grid, threads>>>(d_bx, d_by, d_bz,
							  d_gk,
							  iNx, iNy, iNz, iLab);

						  updatePstMult<<<grid, threads>>>(d_gk, d_u,
							  d_bx, d_by, d_bz,
							  d_div, d_pt,
							  d_Ct,
							  d_FPS,
							  cc,
							  iNx, iNy, iNz, iLab);


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
					
					  /* resolveBoundaryCondtions<<<grid, threads>>>(d_u, 
						   iNx, iNy, iNz, iLab);*/
	
					  /* Memcpy relevant image data buffers back to host for next frame */
					  cudasafe(cudaMemcpy(h_u, d_u, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab-1)), cudaMemcpyDeviceToHost), "cudaMemcpy lambda d2h");


					  //  printf ("fps = %f\n",fps / (iNx*iNy*iNz));
					  printf("Iterations: %d\n",iNI);

					  /* Free memory */
					//  free( (float *) h_tempu );
					  free( (float *) h_bx );
					  free( (float *) h_by );
					  free( (float *) h_bz );
					  free( (float *) h_div );
					  
					  free( (float *) h_gk );
					  free( (float *) h_FPS );
					  free( (float *) h_cvg );

					  /* Free GPU Memory */
					  cudaFree(d_lambda);
					  cudaFree(d_Ct);
					  cudaFree(d_u);
					  cudaFree(d_bx);
					  cudaFree(d_by);
					  cudaFree(d_bz);
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
