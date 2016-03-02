/***************************************************************************/
/* Name:      cmf3DHybridPottsCut_kernels.cu                           

   Authors:
   Martin Rajchl   mrajchl@imaging.robarts.ca
   Jing Yuan       cn.yuanjing@googlemail.com

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


__global__ void updateP1(float *ps, float *pt, float *div, float *gk, 
			 float *u,
			 float cc,
			 int iNx, int iNy, int iNz, int iLab){
  
  int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;

  float fpt = 0.0f;

  //  if( idxVolume <= iNx*iNy*iNz){
    //    return;


    if( ( (idxVolume%iNx) != (iNx-1) ) &&
	( (idxVolume/(iNx*iNy)) < (iNz-1) ) &&
	( ((idxVolume/iNx)%iNy) != (iNy-1))
	){

      for(int id = 0; id < iLab; id++){

	int idx = idxVolume + id*(iNx*iNy*iNz);

	if (id < 2){
	  fpt = ps[idxVolume];
	}
	else{
	  fpt = pt[idxVolume];
	}

	gk[idx] = div[idx] - (fpt - pt[idx] + u[idx]/cc) ;

	//gk[idx] = (((div[idx] - fpt) + pt[idx]) - u[idx]);

      }
    }

    __syncthreads();

  
  }
//}


__global__ void updateP(float *bx, float *by, float *bz,
			float steps,
			float *gk,
			int iNx, int iNy, int iNz, int iLab){

  int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;

  // if( idxVolume <= iNx*iNy*iNz){
    

    if( ( (idxVolume%iNx) != (iNx-1) ) &&
	( (idxVolume/(iNx*iNy)) < (iNz-1) ) &&
	( ((idxVolume/iNx)%iNy) != (iNy-1))
	){

      for(int id = 0; id < iLab; id++){

	int idx = idxVolume + id*(iNx*iNy*iNz);
	float currVal = gk[idx];

	bx[idx+1] = steps * ( gk[idx+1] - currVal ) + bx[idx+1];

	by[idx+iNx] = steps * ( gk[idx+iNx] - currVal ) + by[idx+iNx];

	bz[idx+(iNx*iNy)] = steps * (gk[idx+(iNx*iNy)] - currVal) + bz[idx+(iNx*iNy)];
      }
    }
    __syncthreads();
  }
//}


__global__ void projStep1alpha(float *bx, float *by, float *bz,
			       float *gk,
			       float *penalty1, float *penalty2,
			       int iNx, int iNy, int iNz, int iLab){

  int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;

  // if( idxVolume <= iNx*iNy*iNz){
   

    if( ( (idxVolume%iNx) != (iNx-1) ) &&
	( (idxVolume/(iNx*iNy)) < (iNz-1) ) &&
	( ((idxVolume/iNx)%iNy) != (iNy-1))
	){

      for(int id = 0; id < iLab; id++){

	int idx = idxVolume + id*(iNx*iNy*iNz);

	float fps = 0.0f;
	float fpt = 0.0f;

	fpt = SQRTgpu((SQR(bx[idx]) + SQR(bx[idx+1]) +
		       SQR(by[idx]) + SQR(by[idx+iNx]) + 
		       SQR(bz[idx]) + SQR(bz[idx+(iNx*iNy)]) ) * 0.5f );
      
	if(id < 2){
	  fps = penalty1[idxVolume];
	}
	else{
	  fps = penalty2[idxVolume];
	}

	if(fpt > fps)
	  gk[idx] =  fps/fpt;
	else
	  gk[idx] = 1.0f;

      }
    }
    __syncthreads();
  }
//}


__global__ void projStep2Total(float *bx, float *by, float *bz,
			       float *gk,
			       int iNx, int iNy, int iNz, int iLab){

  int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;


  // if( idxVolume <= iNx*iNy*iNz){

    if( ( (idxVolume%iNx) != (iNx-1) ) &&
	( (idxVolume/(iNx*iNy)) < (iNz-1) ) &&
	( ((idxVolume/iNx)%iNy) != (iNy-1))
	){

      for(int id = 0; id < iLab; id++){

	int idx = idxVolume + id*(iNx*iNy*iNz);

	float gkVal = gk[idx];


	bx[idx+1] = ( gk[idx+1] + gkVal ) * 0.5f * bx[idx+1];
	by[idx+iNx] = ( gk[idx+iNx] + gkVal ) * 0.5f * by[idx+iNx]; 
	bz[idx+(iNx*iNy)] = ( gk[idx+(iNx*iNy)] + gkVal ) * 0.5f * bz[idx+(iNx*iNy)]; 

      }
    }
    __syncthreads();
  }

//}

__global__ void calcDivergence(float *bx, float *by, float *bz,
			       float *div, 
			       int iNx, int iNy, int iNz, int iLab){

 int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;

 if( ( (idxVolume%iNx) != (iNx-1) ) &&
	( (idxVolume/(iNx*iNy)) < (iNz-1) ) &&
	( ((idxVolume/iNx)%iNy) != (iNy-1))
	){


   // calculate divergence
      for(int id = 0; id < iLab; id++){
      
	int idx = idxVolume + id*(iNx*iNy*iNz);

	div[idx] = bx[idx+1] - bx[idx] + by[idx+iNx] - by[idx] 
	  + bz[idx+(iNx*iNy)] - bz[idx];
      }
      //

 }
    __syncthreads();
}



__global__ void updatePstMult(float *gk, float *u,
			      float *bx, float *by, float *bz,
			      float *div, float *ps, float *pt,
			      float *Ct,
			      float *FPS,
			      float cc,
			      int iNx, int iNy, int iNz, int iLab){

  int idxVolume = blockIdx.x * blockDim.x + threadIdx.x;


 
  float fps = 0.0f;
  float fpt = 0.0f;
  
  FPS[idxVolume] = 0.0f;
  // if( idxVolume <= iNx*iNy*iNz){

    if( ( (idxVolume%iNx) != (iNx-1) ) &&
  	( (idxVolume/(iNx*iNy)) < (iNz-1) ) &&
  	( ((idxVolume/iNx)%iNy) != (iNy-1))
  	){



      // calculate divergence
      for(int id = 0; id < iLab; id++){
      
  	int idx = idxVolume + id*(iNx*iNy*iNz);

  	div[idx] = bx[idx+1] - bx[idx] + by[idx+iNx] - by[idx] 
  	  + bz[idx+(iNx*iNy)] - bz[idx];
      }
      //


      // update the sink flow field pt (x,1)
      fpt = ps[idxVolume] - div[idxVolume] + u[idxVolume]/cc;
      //fpt = ps[idxVolume] - div[idxVolume] + u[idxVolume];

      for(int id = 2; id < iLab; id++){

	int idx = idxVolume + id*(iNx*iNy*iNz);

	fpt += div[idx] + pt[idx] - u[idx]/cc;
	//fpt += (div[idx] + pt[idx] - u[idx]);
      
      }
          
      pt[idxVolume] = fpt / 4.0f;


      // update the source flow ps
      fpt = 0.0f;

      for (int id = 0; id < 2;  id++){

	int idx = idxVolume + id*(iNx*iNy*iNz);
      
	fpt += (div[idx] + pt[idx] - u[idx]/cc);  
	//fpt += (div[idx] + pt[idx] - u[idx]);  
      }

      ps[idxVolume] = (fpt/2.0f) + (1.0f/(cc*2.0f)); // FCP
      // float FCP = (1.0f/(cc*2.0f)); // FCP
      // ps[idxVolume] = __fadd_rz(__fdiv_rz(fpt,2), FCP);
      //


      // update the sink flow field pt(x,i)
      for (int id = 1; id < iLab; id++){

	int idx = idxVolume + id*(iNx*iNy*iNz);    

	if (id == 1){
	  fps = ps[idxVolume] + u[idx]/cc - div[idx];
	  //fps = ps[idxVolume] + u[idx] - div[idx];
	}
	else{
	  fps = pt[idxVolume] + u[idx]/cc - div[idx];
	  //fps = pt[idxVolume] + u[idx] - div[idx];
	} 

	pt[idx] = MIN(fps , Ct[idx]);
 
      }


     
      //
     
      /* update the multipliers */

      for (int id = 0; id < iLab; id++){

	int idx = idxVolume + id*(iNx*iNy*iNz);    
	fpt = 0.0f;

	if(id < 2){
	  fpt = cc*(div[idx] + pt[idx] - ps[idxVolume]);
	  //fpt = div[idx] + pt[idx] - ps[idxVolume];
	}
	else{
	  fpt = cc*(div[idx] + pt[idx] - pt[idxVolume]);
	  //fpt = div[idx] + pt[idx] - pt[idxVolume];
	}

	u[idx] -= fpt;

	FPS[idxVolume] += ABS(fpt);
      }


    }
    __syncthreads();
  }
//}





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

