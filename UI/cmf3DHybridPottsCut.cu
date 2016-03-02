/***************************************************************************/
/*      Name:       cmf3DHybridPottsCut.c                               
                                                
        GPU program to perform the continuous max-flow algorithm to solve the
        3D continuous-cut problem with multiple partially-ordered labels (Potts Model)
 
        Usage: [u, erriter, i, timet] = CMF3D_HP02_GPU(penalty, C_t, para);
 
        Inputs (penalty1, penalty2, C_t, para): 
 
	- penalty1(x), penalty2(x): point to the edge-weight penalty to
	the total-variation function.
 
	For the case without incorporating image-edge weights, 
	penalty is given by the constant everywhere. For the case 
	with image-edge weights, penalty is given by the pixelwise 
	weight function:
 
	for example, penalty(x) = b/(1 + a*| grad f(x)|) where b,a > 0.
   
	- C_t(x,i=1...nlab): point to the capacities of 
	sink flows pt(x,i=1...nlab);
 
	- para: a sequence of parameters for the algorithm
	para[0,1,2]: rows, cols, heights of the given 3D data
	para[3]: the number of labels or segmentations
	para[4]: the maximum iteration number
	para[5]: the error bound for convergence
	para[6]: cc for the step-size of augmented Lagrangian method
	para[7]: the step-size for the graident-projection step to the
	total-variation function. Its optimal range is [0.06, 0.12].
 
        Outputs (u, erriter, i, timet):
 
	- u: the computed continuous labeling function u(x,i=1...nlab) in [0,1]. 
	As the following paper [2], the final label function is 
	given by the maximum of u(x,i=1...nlab) at each x.
 
	- erriter: it returns the error evaluation of each iteration,
	i.e. it shows the convergence rate. One can check the algorithm
	performance.

	- i: gives the total number of iterations, when the algorithm converges.

	- timet: gives the total computation time.

	This software can be used only for research purposes, you should cite ALL of
	the aforementioned papers in any resulting publication.

	Please email Jing Yuan (cn.yuanjing@gmail.com) for any questions, suggestions and bug reports

	The Software is provided "as is", without warranty of any kind.


	Version 1.0
	https://sites.google.com/site/wwwjingyuan/       

	Copyright 2012 Jing Yuan (cn.yuanjing@gmail.com)      

*/
/***************************************************************************/

/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda.h"

#include "cmf3DHybridPottsCut_kernels.cu"
#include "cmf3DHybridPottsCut.h"

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

void cmf3DHybridPottsCut(float *h_penalty1,
			 float *h_penalty2,
			 float *h_Ct,
			 float *pars,
			 float *h_u
			 ){



  /*  Declaration of host variables */
  float *h_ps, *h_pt, *h_div, *h_gk;
  float *h_bx, *h_by, *h_bz;

  float *h_cvg, *h_FPS;

  float fps;

  /* Declaration of device variables */
  float *d_penalty1, *d_penalty2, *d_Ct, *d_u;

  float *d_ps, *d_pt, *d_div, *d_gk;
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

  h_bx =   (float *) calloc( (unsigned)(iNy*iNx*iNz*(iLab)), sizeof(float) );
  h_by =   (float *) calloc( (unsigned)(iNy*iNx*iNz*(iLab)), sizeof(float) );
  h_bz =   (float *) calloc( (unsigned)(iNy*iNx*iNz*(iLab)), sizeof(float) );

  h_ps =   (float *) calloc( (unsigned)(iNy*iNx*iNz), sizeof(float) );
  h_pt =   (float *) calloc( (unsigned)(iNy*iNx*iNz*(iLab)), sizeof(float) );

  h_gk =  (float *) calloc( (unsigned)(iNx*iNy*iNz*(iLab)), sizeof(float) );
  h_div =  (float *) calloc( (unsigned)(iNx*iNy*iNz*(iLab)), sizeof(float) );
  h_cvg =  (float *) calloc( (unsigned)iNbIters+1, sizeof(float) );
  h_FPS =  (float *) calloc( (unsigned)(iNy*iNx*iNz), sizeof(float) );

  if (!h_bx   ||
      !h_by   ||
      !h_bz   ||

      !h_ps   ||
      !h_pt   ||

      !h_FPS  ||
      !h_gk   ||
      !h_div  ||
      !h_cvg)
    printf("Host memory allocation failure\n");


  // /* Preprocessing initial values */

  // int ix, iy, iz, S2D(iNx*iNy), fpt, S3D(iNx*iNy*iNz),id;

  //   for (iz=0; iz< iNz; iz++){
  //       int indz = iz*S2D;
  //       for (ix=0; ix < iNx; ix++){
  //           int indy = ix*iNy;
  //           for (iy=0; iy < iNy; iy++){
  //               int index = indz + indy + iy;

  //               fpt = h_Ct[index+S3D];
  //               int ik = 1;
                
  //               for (id = 2; id < iLab; id++){
  //                   int index1 = index + id*S3D;
  //                   if (fpt >= h_Ct[index1]){
  //                       fpt = h_Ct[index1];
  //                       ik = id;
  //                   }
  //               }
                    
  //               h_ps[index] = fpt;
  //               h_u[index+ik*S3D] = 1/cc;
                
  //               for (id = 0; id < iLab; id++){        
  //                   h_pt[index+id*S3D] = fpt;
  //               }

  //               /* initialize u1 = u3 + u4 + u5 */
  //               for (id = 2; id < iLab; id++){
  //                   h_u[index] += h_u[index+id*S3D];
  //               }
  //           }
  //       }          
  //   }




  /* Device memory allocation */
  cudasafe( cudaMalloc( (void**) &d_penalty1, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc"); 
  cudasafe( cudaMalloc( (void**) &d_penalty2, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc"); 
  cudasafe( cudaMalloc( (void**) &d_Ct, sizeof(float)*(unsigned)(iNx*iNy*iNz*iLab)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_u, sizeof(float)*(unsigned)(iNx*iNy*iNz*iLab)), "cudaMalloc");

  cudasafe( cudaMalloc( (void**) &d_ps, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_pt, sizeof(float)*(unsigned)(iNx*iNy*iNz)*iLab), "cudaMalloc");

  cudasafe( cudaMalloc( (void**) &d_bx, sizeof(float)*(unsigned)(iNx*iNy*iNz)*(iLab)), "cudaMalloc"); 
  cudasafe( cudaMalloc( (void**) &d_by, sizeof(float)*(unsigned)(iNx*iNy*iNz)*(iLab)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_bz, sizeof(float)*(unsigned)(iNx*iNy*iNz)*(iLab)), "cudaMalloc");

  cudasafe( cudaMalloc( (void**) &d_gk, sizeof(float)*(unsigned)(iNx*iNy*iNz)*(iLab)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_div, sizeof(float)*(unsigned)(iNx*iNy*iNz)*(iLab)), "cudaMalloc");
  cudasafe( cudaMalloc( (void**) &d_FPS, sizeof(float)*(unsigned)(iNx*iNy*iNz)), "cudaMalloc");


  /* Memcpy image buffers from Host to Device */

  cudasafe(cudaMemcpy(d_penalty1, h_penalty1, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice),"cudaMemcpy pen1");
  cudasafe(cudaMemcpy(d_penalty2, h_penalty2, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice),"cudaMemcpy pen2");
  cudasafe(cudaMemcpy(d_Ct, h_Ct, sizeof(float)*(unsigned)(iNx*iNy*iNz*iLab), cudaMemcpyHostToDevice),"cudaMemcpy Ct");
  cudasafe(cudaMemcpy(d_u, h_u, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab)), cudaMemcpyHostToDevice),"cudaMemcpy u");

  cudasafe(cudaMemcpy(d_ps, h_ps, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice),"cudaMemcpy ps");
  cudasafe(cudaMemcpy(d_pt, h_pt, sizeof(float)*(unsigned)(iNx*iNy*iNz*iLab), cudaMemcpyHostToDevice),"cudaMemcpy pt");

  cudasafe(cudaMemcpy(d_bx, h_bx, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab)), cudaMemcpyHostToDevice),"cudaMemcpy bx");
  cudasafe(cudaMemcpy(d_by, h_by, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab)), cudaMemcpyHostToDevice),"cudaMemcpy by");
  cudasafe(cudaMemcpy(d_bz, h_bz, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab)), cudaMemcpyHostToDevice),"cudaMemcpy bz");

  cudasafe(cudaMemcpy(d_gk, h_gk, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab)), cudaMemcpyHostToDevice),"cudaMemcpy gk");
  cudasafe(cudaMemcpy(d_div, h_div, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab)), cudaMemcpyHostToDevice),"cudaMemcpy div");
  cudasafe(cudaMemcpy(d_FPS, h_FPS,   sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyHostToDevice),"cudaMemcpy FPS");



  /* Main iterations */

  /* Setup GPU execution parameters */
  dim3 threads(BLOCK_SIZE);
  dim3 grid( ((iNx*iNy*iNz)/threads.x +( (!(iNx*iNy*iNz)%threads.x) ? 0:1)) );

  printf (" Grid: %d, Threads: %d \n", grid.x, threads.x);

		
  //  iNbIters = 2;	
  int iNI;
  for( iNI = 0; iNI<iNbIters; iNI++ ) 
    {	


      /* Solver kernels */
      updateP1<<<grid, threads>>>(d_ps, d_pt, d_div, d_gk,
				  d_u,
				  cc,
				  iNx, iNy, iNz, iLab);

      

      updateP<<<grid, threads>>>(d_bx, d_by, d_bz,
				 steps,
				 d_gk,
				 iNx, iNy, iNz, iLab);

      projStep1alpha<<<grid, threads>>>(d_bx, d_by, d_bz,
					d_gk,
					d_penalty1, d_penalty2,
					iNx, iNy, iNz, iLab);

      projStep2Total<<<grid, threads>>>(d_bx, d_by, d_bz,
					d_gk,
					iNx, iNy, iNz, iLab);

      // calcDivergence<<<grid, threads>>>(d_bx, d_by, d_bz,
      // 					d_div,
      // 				       iNx, iNy, iNz, iLab);

      updatePstMult<<<grid, threads>>>(d_gk, d_u,
				       d_bx, d_by, d_bz,
				       d_div, d_ps, d_pt,
				       d_Ct,
				       d_FPS,
				       cc,
				       iNx, iNy, iNz, iLab);

      // fps =0;
      /* Pixelwise error accumulation for convergence rate calculation */
      // for(unsigned int errorAccumBlockSize = 1; errorAccumBlockSize < iNx*iNy*iNz; errorAccumBlockSize += errorAccumBlockSize){
      // 	dim3 errorAccumThreads(16);
      // 	dim3 errorAccumGrid( (iNx*iNy*iNz) / (2*errorAccumBlockSize*errorAccumThreads.x) );
      // 	if( (iNx*iNy*iNz) % (2*errorAccumBlockSize*errorAccumThreads.x) != 0) errorAccumGrid.x += 1;
      // 	errorAccumulation<<<errorAccumGrid, errorAccumThreads>>>(d_FPS, errorAccumBlockSize, iNx*iNy*iNz);
      // }

      // cudaMemcpy( &fps, d_FPS, sizeof(float), cudaMemcpyDeviceToHost);

      
      cudasafe( cudaMemcpy( h_FPS, d_FPS, sizeof(float)*(unsigned)(iNx*iNy*iNz), cudaMemcpyDeviceToHost), "cudaMemcpy FPS d2H");
      

      fps = 0;
      for (int ii=0; ii< (iNx*iNy*iNz); ii++){
      	fps += h_FPS[ii];

      }
      
      h_cvg[iNI] = cc* fps / (iNx*iNy*iNz*iLab);

      printf("cvg: %f ", h_cvg[iNI]);
      printf("fps: %f \n", fps);

       

      if (h_cvg[iNI] <= fError)
      	break;


    }
					
  /* resolveBoundaryCondtions<<<grid, threads>>>(d_u, 
     iNx, iNy, iNz, iLab);*/
	
  /* Memcpy relevant image data buffers back to host for next frame */
  cudasafe(cudaMemcpy(h_u, d_u, sizeof(float)*(unsigned)(iNx*iNy*iNz*(iLab)), cudaMemcpyDeviceToHost), "cudaMemcpy lambda d2h");


  //  printf ("fps = %f\n",fps / (iNx*iNy*iNz));
  printf("Iterations: %d\n",iNI);

  /* Free memory */
  //  free( (float *) h_tempu );
  free( (float *) h_bx );
  free( (float *) h_by );
  free( (float *) h_bz );

  free( (float *) h_ps );
  free( (float *) h_pt );

  free( (float *) h_div );
  free( (float *) h_gk );

  free( (float *) h_FPS );
  free( (float *) h_cvg );

  /* Free GPU Memory */
  cudaFree(d_penalty1);
  cudaFree(d_penalty2);
  cudaFree(d_Ct);
  cudaFree(d_u);
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
