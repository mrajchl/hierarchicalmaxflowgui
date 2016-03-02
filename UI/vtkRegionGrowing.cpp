#include "vtkRegionGrowing.h"
#define MARKED_VOXEL 1
static  int nbr4map[4][2] = {-1,0, 1,0,  0,-1,  0,1};
static  int neigh4map[6][3] = {-1,0,0, 1,0,0, 0,-1,0, 0,1,0, 0,0,-1, 0,0,1};
//////////////////////////////////////////////////////////////////////////
MVoxel::MVoxel(int _x, int _y, int _z)
:x(_x), y(_y), z(_z)
{
}
//////////////////////////////////////////////////////////////////////////
MVoxel::MVoxel(const MVoxel& v)
    : x(v.x), y(v.y), z(v.z)
{
}
//////////////////////////////////////////////////////////////////////////
MVoxel::~MVoxel()
{

}
//////////////////////////////////////////////////////////////////////////
MVoxel &MVoxel::operator=(const MVoxel& v)
{
    if (this != &v)
    {
        x=v.x;
		y=v.y;
		z=v.z;
        
    }
    return *this;
}
//////////////////////////////////////////////////////////////////////////
vtkRegionGrowing::vtkRegionGrowing()
{
mVoxelMark=1;
mGradStrength=0.5;

}
//////////////////////////////////////////////////////////////////////////
void vtkRegionGrowing::copySlice()
{
 int ext[6];
 mSlice->GetExtent(ext);
int ix,iy,iz;
unsigned char* inVal;
unsigned char* outVal;

if(mOrientation==2)
{

	iz=mSliceNumber;
	for(iy=ext[2]; iy<=ext[3]; iy++)
	for(ix=ext[0]; ix<=ext[1]; ix++)
	{
		inVal=(unsigned char*)mSlice->GetScalarPointer(ix,iy,iz);
		outVal=(unsigned char*)mMask->GetScalarPointer(ix,iy,iz);
		if(*inVal==mVoxelMark)*outVal=*inVal;
		
	}
}


if(mOrientation==1)
{

	iy=mSliceNumber;
	for(iz=ext[4]; iz<=ext[5]; iz++)
	for(ix=ext[0]; ix<=ext[1]; ix++)
	{
		inVal=(unsigned char*)mSlice->GetScalarPointer(ix,iy,iz);
		outVal=(unsigned char*)mMask->GetScalarPointer(ix,iy,iz);
		if(*inVal==mVoxelMark)*outVal=*inVal;
	}



}


if(mOrientation==0)
{
	ix=mSliceNumber;
	for(iz=ext[4]; iz<=ext[5]; iz++)
	for(iy=ext[2]; iy<=ext[3]; iy++)
	{
		inVal=(unsigned char*)mSlice->GetScalarPointer(ix,iy,iz);
		outVal=(unsigned char*)mMask->GetScalarPointer(ix,iy,iz);
		if(*inVal==mVoxelMark)*outVal=*inVal;
	}
}

}
//////////////////////////////////////////////////////////////////////////
void vtkRegionGrowing::createSlice()
{
double spacing[3];
 int ext[6], nExt[6];
 int dim[3], nDim[3];
 double orig[3];
mVolume->GetExtent(ext);
mVolume->GetSpacing(spacing);
mVolume->GetDimensions(dim);
mVolume->GetOrigin(orig);
/*
if(mOrientation==2)
{
nExt[0]=ext[0];
nExt[1]=ext[1];
nExt[2]=ext[2];
nExt[3]=ext[3];
nExt[4]=mSliceNumber;
nExt[5]=mSliceNumber;

nDim[0]=dim[0];
nDim[1]=dim[1];
nDim[2]=1;
}


if(mOrientation==1)
{
nExt[0]=ext[0];
nExt[1]=ext[1];
nExt[2]=mSliceNumber;
nExt[3]=mSliceNumber;
nExt[4]=ext[4];
nExt[5]=ext[5];

nDim[0]=dim[0];
nDim[1]=1;
nDim[2]=dim[2];
}

if(mOrientation==0)
{
nExt[0]=mSliceNumber;
nExt[1]=mSliceNumber;
nExt[2]=ext[2];
nExt[3]=ext[3];
nExt[4]=ext[4];
nExt[5]=ext[5];

nDim[0]=1;
nDim[1]=dim[1];
nDim[2]=dim[2];
}
*/
mSlice=vtkImageData::New();
mSlice->SetSpacing(spacing);
mSlice->SetExtent(ext);
mSlice->SetDimensions(dim);
mSlice->SetOrigin(orig);
mSlice->SetScalarTypeToUnsignedChar();
mSlice->SetNumberOfScalarComponents(1);
mSlice->AllocateScalars();
mSlice->Update();
//mSlice->DebugOn();

}
//////////////////////////////////////////////////////////////////////////
void vtkRegionGrowing::Start()
{
	
mVolume->GetExtent(mExt);
createSlice();


int n=mSeeds.size();
for(int i=0; i<n; i++)mQueue.push(mSeeds[i]);

if(mOrientation==0)processX();
if(mOrientation==1)processY();
if(mOrientation==2)processZ();

copySlice();
mSlice->Delete();	
}
//////////////////////////////////////////////////////////////////////////
void vtkRegionGrowing::processZ()
{

int i,ii,x,y,z;
z=mSliceNumber;

unsigned char* mv;

MVoxel voxel;



while(!mQueue.empty())
{


voxel=mQueue.front();

     for (ii=0; ii<4; ii++)
      {

		x= voxel.x+nbr4map[ii][0];
		y= voxel.y+nbr4map[ii][1];
		
			if((x>=mExt[0]) && (y>=mExt[2]) && (x<=mExt[1]) && (y<=mExt[3]) )
			{
				if(this->checkVoxel(x,y,z) )  
				{
				mv=(unsigned char*)mSlice->GetScalarPointer(x,y,z);
				*mv=mVoxelMark;
				mQueue.push(MVoxel(x,y,z));
				}

			}
	 }
mQueue.pop();
}



}
//////////////////////////////////////////////////////////////////////////
void vtkRegionGrowing::processY()
{

int i,ii,x,y,z;
y=mSliceNumber;

unsigned char* mv;

MVoxel voxel;


while(!mQueue.empty())
{


voxel=mQueue.front();

     for (ii=0; ii<4; ii++)
      {

		x= voxel.x+nbr4map[ii][0];
		z= voxel.z+nbr4map[ii][1];
		
			if((x>=mExt[0]) && (z>=mExt[4]) && (x<=mExt[1]) && (z<=mExt[5]) )
			{
				if(this->checkVoxel(x,y,z) )  
				{
				mv=(unsigned char*)mSlice->GetScalarPointer(x,y,z);
				*mv=mVoxelMark;
				mQueue.push(MVoxel(x,y,z));
				}

			}
	 }
mQueue.pop();
}




}
//////////////////////////////////////////////////////////////////////////
void vtkRegionGrowing::processX()
{


int i,ii,x,y,z;
x=mSliceNumber;

unsigned char* mv;

MVoxel voxel;


while(!mQueue.empty())
{


voxel=mQueue.front();

     for (ii=0; ii<4; ii++)
      {

		z= voxel.z+nbr4map[ii][0];
		y= voxel.y+nbr4map[ii][1];
		
			if((z>=mExt[4]) && (y>=mExt[2]) && (z<=mExt[5]) && (y<=mExt[3]) )
			{
				if(this->checkVoxel(x,y,z) )  
				{
				mv=(unsigned char*)mSlice->GetScalarPointer(x,y,z);
				*mv=mVoxelMark;
				mQueue.push(MVoxel(x,y,z));
				}

			}
	 }
mQueue.pop();
}




}
//////////////////////////////////////////////////////////////////////////
bool vtkRegionGrowing::checkVoxel(int& x, int& y, int& z)
{
double val;

unsigned char* mv=(unsigned char*)mSlice->GetScalarPointer(x,y,z);
val=mVolume->GetScalarComponentAsDouble(x,y,z,0);
if( (mLT<=val) && (val<=mUT) && (*mv)!=mVoxelMark)return true;
else return false; 

	
}

//////////////////////////////////////////////////////////////////////////
bool vtkRegionGrowing::checkGradient(int& x, int& y, int& z)
{
double val;
double nval;
val=mVolume->GetScalarComponentAsDouble(x,y,z,0);
 int i, x1,y1,z1;
double grad = 0;
double gradMax = 0;
   int count = 0;
   
 for(i=0; i<6; ++i)        
 {  
	 x1=x+neigh4map[i][0];
     y1=y+neigh4map[i][1];
     z1=z+neigh4map[i][2];

 if((x1>=mExt[0]) && (y1>=mExt[2]) && (x1<=mExt[1]) && (y1<=mExt[3]) && (z1>=mExt[4]) && (z1<=mExt[5])) 
   { 
	 nval=mVolume->GetScalarComponentAsDouble(x1,y1,z1,0);
	   grad=abs(nval - val); 
       count++;
       if(grad>gradMax)gradMax = grad ;
   }
  }
          
          if(gradMax<(this->mGradStrength*val*count))return true;
          else return false;           

 }