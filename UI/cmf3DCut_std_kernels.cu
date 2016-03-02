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


#include <stdio.h>

#define BLOCK_SIZE 256
#define SQR(x) (x)*(x)
#define MAX(a,b) ( a > b ? a : b )
#define MIN(a,b) ( a <= b ? a : b )
#define SIGN(x) ( x >= 0.0 ? 1.0 : -1.0 )
#define ABS(x) ( (x) > 0.0 ? x : -(x) )
#define X(iy,ix) (ix)*iNy + iy
#define Xe(iy,ix) (ix)*iNye+ (iy)

#define SQRTgpu sqrt


__global__ void updateP1(float *gk,
			 float *bx, float *by, float *bz,
			 float *ps, float *pt,
			 float *lambda,
			 float cc,
			 int iNx, int iNy, int iNz){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;


   if( ( (idx%iNx) != (iNx-1) ) &&
       ( (idx/(iNx*iNy)) < (iNz-1) ) &&
       ( ((idx/iNx)%iNy) != (iNy-1))
       ){
  
     gk[idx] =  ( bx[idx+1] - bx[idx] +
		  by[idx+iNx] - by[idx] +
		  bz[idx+(iNx*iNy)] - bz[idx] ) -
       ( ps[idx] - pt[idx] + 
	 lambda[idx]/cc );
   }
  
}


__global__ void updateP(float *bx, float *by, float *bz,
			float steps,
			float *gk,
			int iNx, int iNy, int iNz){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float currVal = gk[idx];


  if( ( (idx%iNx) != (iNx-1) ) &&
      ( (idx/(iNx*iNy)) < (iNz-1) ) &&
      ( ((idx/iNx)%iNy) != (iNy-1))
      ){
    bx[idx+1] = steps * ( gk[idx+1] - currVal ) + bx[idx+1];

    by[idx+iNx] = steps * ( gk[idx+iNx] - currVal ) + by[idx+iNx];

    bz[idx+(iNx*iNy)] = steps * (gk[idx+(iNx*iNy)] - currVal) + bz[idx+(iNx*iNy)];
  }

}


__global__ void projStep1alpha(float *bx, float *by, float *bz,
			       float alpha,
			       float *gk,
			       int iNx, int iNy, int iNz){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;


  if( ( (idx%iNx) != (iNx-1) ) &&
      ( (idx/(iNx*iNy)) < (iNz-1) ) &&
      ( ((idx/iNx)%iNy) != (iNy-1))
      ){
    float fpt = SQRTgpu((SQR(bx[idx]) + SQR(bx[idx+1]) +
			 SQR(by[idx]) + SQR(by[idx+iNx]) + 
			 SQR(bz[idx]) + SQR(bz[idx+(iNx*iNy)]) ) * 0.5f );
    
    gk[idx] = (fpt > alpha) ? alpha / fpt : 1.0f;
  }
}


__global__ void projStep1omega(float *bx, float *by, float *bz,
			       float *omega,
			       float *gk,
			       int iNx, int iNy, int iNz){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  float fpt = SQRTgpu( (SQR(bx[idx]) + SQR(bx[idx+1]) +
			SQR(by[idx]) + SQR(by[idx+iNx]) + 
			SQR(bz[idx]) + SQR(bz[idx+(iNx*iNy)]) ) * 0.5f );
  
  gk[idx] = (fpt > omega[idx]) ? omega[idx] / fpt : 1.0f;
}



__global__ void projStep2Total(float *bx, float *by, float *bz,
			       float *gk,
			       int iNx, int iNy, int iNz){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float gkVal = gk[idx];
  if( ( (idx%iNx) != (iNx-1) ) &&
      ( (idx/(iNx*iNy)) < (iNz-1) ) &&
      ( ((idx/iNx)%iNy) != (iNy-1))
      ){
    bx[idx+1] = ( gk[idx+1] + gkVal ) * 0.5f * bx[idx+1];
    by[idx+iNx] = ( gk[idx+iNx] + gkVal ) * 0.5f * by[idx+iNx]; 
    bz[idx+(iNx*iNy)] = ( gk[idx+(iNx*iNy)] + gkVal ) * 0.5f * bz[idx+(iNx*iNy)]; 
  }
}



__global__ void updatePstMult(float *gk,
			      float *bx, float *by, float *bz,
			      float *ps, float *pt,
			      float *Cs, float *Ct,
			      float *lambda,
			      float *FPS,
			      float cc,
			      int iNx, int iNy, int iNz){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float gkVal = 0.0f;

  if( ( (idx%iNx) != (iNx-1) ) &&
      ( (idx/(iNx*iNy)) < (iNz-1) ) &&
      ( ((idx/iNx)%iNy) != (iNy-1))
      ){
    gkVal = ( bx[idx+1] - bx[idx] +
	      by[idx+iNx] - by[idx] +
	      bz[idx+(iNx*iNy)] - bz[idx]
	      );
    
    gk[idx] = gkVal;
  }
  else{
    gk[idx] = 0.0f;
  }
  
  ps[idx] = MIN( ( gkVal + pt[idx] - lambda[idx]/cc + 1.0f/cc ), Cs[idx]);
  pt[idx] = MIN( (- gkVal + ps[idx] + lambda[idx]/cc) , Ct[idx]);
  
  
  float fpsVal = cc*( gkVal + pt[idx] - ps[idx]);
  
  lambda[idx] -= fpsVal;
  FPS[idx] = ABS(fpsVal);

}


__global__ void errorAccumulation(float* errorBuffer, unsigned int blockSize, unsigned int arraySize){

  int idx = (blockSize + blockSize) * (blockIdx.x * blockDim.x + threadIdx.x);
  int idxUp = idx + blockSize;

  float error1 = (idx < arraySize) ? errorBuffer[idx] : 0.0f;
  float error2 = (idxUp < arraySize) ? errorBuffer[idxUp] : 0.0f;

  __syncthreads();
  if(idx < arraySize) errorBuffer[idx] = error1 + error2;

}
