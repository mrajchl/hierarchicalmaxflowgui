/***************************************************************************/
/* Name:      cmf3DCut_ovl_kernels.cu                           

Authors:
Martin Rajchl   mrajchl@imaging.robarts.ca
Jing Yuan       cn.yuanjing@googlemail.com

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
"A FAST CONTINUOUS MAX-FLOW APPROACH TO NON-CONVEX
MULTILABELING PROBLEMS" 
Submitted to Math. Comp. (AMS Journals)



*/
/***************************************************************************/


#include <stdio.h>

#define SQR(x) (x)*(x)
#define MAX(a,b) ( a > b ? a : b )
#define MIN(a,b) ( a <= b ? a : b )
#define SIGN(x) ( x >= 0.0 ? 1.0 : -1.0 )
#define ABS(x) ( (x) > 0.0 ? x : -(x) )
#define X(iy,ix) (ix)*iNy + iy
#define Xe(iy,ix) (ix)*iNye+ (iy)

#define SQRTgpu sqrt


__global__ void updateP1(float *gk, float *u, float *pt, float *div,
						 float *bx, float *by, float *bz,
						 float cc,
						 int iNx, int iNy, int iNz, int iLab){

							 int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;


							 if( ( (idxVolume%iNx) != (iNx-1) ) &&
								 ( (idxVolume/(iNx*iNy)) < (iNz-1) ) &&
								 ( ((idxVolume/iNx)%iNy) != (iNy-1))
								 ){

									 for(int id = 0; id < iLab-1; id++){

										 int idx = idxVolume + id*(iNx*iNy*iNz);

										 gk[idx] = div[idx] - 
											 ( pt[idx] - pt[idx+(iNx*iNy*iNz)] + u[idx]/cc );

									 }
							 }  
}


__global__ void updateP(float *bx, float *by, float *bz,
						float steps,
						float *gk,
						int iNx, int iNy, int iNz, int iLab){

							int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;


							if( ( (idxVolume%iNx) != (iNx-1) ) &&
								( (idxVolume/(iNx*iNy)) < (iNz-1) ) &&
								( ((idxVolume/iNx)%iNy) != (iNy-1))
								){

									for(int id = 0; id < iLab-1; id++){

										int idx = idxVolume + id*(iNx*iNy*iNz);
										float currVal = gk[idx];



										bx[idx+1] = steps * ( gk[idx+1] - currVal ) + bx[idx+1];

										by[idx+iNx] = steps * ( gk[idx+iNx] - currVal ) + by[idx+iNx];

										bz[idx+(iNx*iNy)] = steps * (gk[idx+(iNx*iNy)] - currVal) + bz[idx+(iNx*iNy)];
									}
							}

}


__global__ void projStep1alpha(float *bx, float *by, float *bz,
							   float *lambda,
							   float *gk,
							   int iNx, int iNy, int iNz, int iLab){

								   int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;



								   if( ( (idxVolume%iNx) != (iNx-1) ) &&
									   ( (idxVolume/(iNx*iNy)) < (iNz-1) ) &&
									   ( ((idxVolume/iNx)%iNy) != (iNy-1))
									   ){

										   for(int id = 0; id < iLab-1; id++){

											   int idx = idxVolume + id*(iNx*iNy*iNz);


											   float fpt = SQRTgpu((SQR(bx[idx]) + SQR(bx[idx+1]) +
												   SQR(by[idx]) + SQR(by[idx+iNx]) + 
												   SQR(bz[idx]) + SQR(bz[idx+(iNx*iNy)]) ) * 0.5f );

											   gk[idx] = (fpt > lambda[id]) ? lambda[id] / fpt : 1.0f;

										   }
								   }
}



__global__ void projStep2Total(float *bx, float *by, float *bz,
							   float *gk,
							   int iNx, int iNy, int iNz, int iLab){

								   int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;



								   if( ( (idxVolume%iNx) != (iNx-1) ) &&
									   ( (idxVolume/(iNx*iNy)) < (iNz-1) ) &&
									   ( ((idxVolume/iNx)%iNy) != (iNy-1))
									   ){

										   for(int id = 0; id < iLab-1; id++){

											   int idx = idxVolume + id*(iNx*iNy*iNz);

											   float gkVal = gk[idx];


											   bx[idx+1] = ( gk[idx+1] + gkVal ) * 0.5f * bx[idx+1];
											   by[idx+iNx] = ( gk[idx+iNx] + gkVal ) * 0.5f * by[idx+iNx]; 
											   bz[idx+(iNx*iNy)] = ( gk[idx+(iNx*iNy)] + gkVal ) * 0.5f * bz[idx+(iNx*iNy)]; 

										   }
								   }
}



__global__ void updatePstMult(float *gk, float *u,
							  float *bx, float *by, float *bz,
							  float *div, float *pt,
							  float *Ct,
							  float *FPS,
							  float cc,
							  int iNx, int iNy, int iNz, int iLab){

								  int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;


								  float fpt = 0.0f;
								  FPS[idxVolume] = 0.0f;

								  if( ( (idxVolume%iNx) != (iNx-1) ) &&
									  ( (idxVolume/(iNx*iNy)) < (iNz-1) ) &&
									  ( ((idxVolume/iNx)%iNy) != (iNy-1))
									  ){

										  for(int id = 0; id < iLab-1; id++){

											  int idx = idxVolume + id*(iNx*iNy*iNz);




											  div[idx] = bx[idx+1] - bx[idx] + by[idx+iNx] - by[idx] 
											  + bz[idx+(iNx*iNy)] - bz[idx];


										  }


										  for(int id = 0; id < iLab; id++){

											  int idx = idxVolume + id*(iNx*iNy*iNz);
											  int idx1 = idx - (iNx*iNy*iNz);



											  if(id == 0){

												  fpt = div[idx] + pt[idx+(iNx*iNy*iNz)] - u[idx]/cc + 1.0f/cc;
												  pt[idx] = MIN(fpt, Ct[idx]);

											  }
											  else if (id == iLab-1){

												  fpt = - div[idx1] + pt[idx1] + u[idx1]/cc;
												  pt[idx] = MIN(fpt, Ct[idx]);

											  }
											  else{

												  fpt = - div[idx1] + pt[idx1] + u[idx1]/cc;
												  fpt = fpt + div[idx] + pt[idx+(iNx*iNy*iNz)] - u[idx]/cc;
												  fpt = fpt/2.0f;
												  pt[idx] = MIN(fpt, Ct[idx]);

											  }
										  }






										  for(int id = 0; id < iLab-1; id++){
											  int idx = idxVolume + id*(iNx*iNy*iNz);

											  float fpsVal = cc*( div[idx] + pt[idx+(iNx*iNy*iNz)] - pt[idx]);

											  u[idx] -= fpsVal;
											  FPS[idxVolume] += ABS(fpsVal);
										  }
								  }

}




__global__ void errorAccumulation(float* errorBuffer, unsigned int blockSize, unsigned int arraySize){

	int idx = (blockSize + blockSize) * (blockIdx.x * blockDim.x + threadIdx.x);
	int idxUp = idx + blockSize;

	float error1 = (idx < arraySize) ? errorBuffer[idx] : 0.0f;
	float error2 = (idxUp < arraySize) ? errorBuffer[idxUp] : 0.0f;

	__syncthreads();
	if(idx < arraySize) errorBuffer[idx] = error1 + error2;

}



// NOT FUNCTIONAL YET! 
__global__ void resolveBoundaryCondtions(float *u, int iNx, int iNy, int iNz, int iLab){

	int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;

	if(!(  (idxVolume%iNx) != (iNx-1) )){
		for(int id = 0; id < iLab; id++){

			int idx = idxVolume + id*(iNx*iNy*iNz);
			u[idx] = u[idx-1];

		}
	}


	if (! ((idxVolume/iNx)%iNy) != (iNy-1)){

		for(int id = 0; id < iLab; id++){

			int idx = idxVolume + id*(iNx*iNy*iNz);
			u[idx] = u[idx-iNx];

		}
	}

	if(! (idxVolume/(iNx*iNy)) < (iNz-1) ){
		for(int id = 0; id < iLab; id++){

			int idx = idxVolume + id*(iNx*iNy*iNz);
			u[idx] = u[idx-(iNx*iNy)];

		}

	}
}
