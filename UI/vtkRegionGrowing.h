#ifndef __VTK_REGION_GROWING_H__
#define __VTK_REGION_GROWING_H__


#include "vtkImageData.h"
#include <vector>
#include <queue>
//////////////////////////////////////////////////////////////////////////
class MVoxel
{
public:
	MVoxel(int _x=0, int _y=0, int _z=0);
	MVoxel(const MVoxel& v);
	virtual ~MVoxel();
     MVoxel &operator=(const MVoxel& v);
int x,y,z;

};
class vtkExtractVOI;
//////////////////////////////////////////////////////////////////////////
class vtkRegionGrowing
{
public:
	vtkRegionGrowing();
	
	void SetVolume(vtkImageData* v){this->mVolume=v;};
	void SetMask(vtkImageData* v){this->mMask=v;};
	void SetSliceNumber(int n){this->mSliceNumber=n;};
	void SetOrientation(int n){this->mOrientation=n;};
	void SetLower(double a){this->mLT=a;};
	void SetUpper(double a){this->mUT=a;};
	void SetSeeds(std::vector<MVoxel> v){this->mSeeds=v;}
	void SetVoxelMark(unsigned char n){this->mVoxelMark=n;};
	void Start();

protected:
	bool checkVoxel(int& x, int& y, int& z);
	bool checkGradient(int& x, int& y, int& z);
	void createSlice();
	void copySlice();
	void processZ();
	void processY();
	void processX();

protected:
	vtkImageData* mVolume;
	vtkImageData* mMask;
	int mSliceNumber;
	int mOrientation;
	double mLT;
	double mUT;
	std::vector<MVoxel> mSeeds;
	std::queue<MVoxel > mQueue;
	int mExt[6];
	unsigned char mVoxelMark;
	vtkImageData* mSlice;
	double mGradStrength;
	
};
#endif