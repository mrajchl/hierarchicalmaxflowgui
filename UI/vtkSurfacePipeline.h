#ifndef __VTK_SURFACE_PIPELINE_H__
#define __VTK_SURFACE_PIPELINE_H__


#include "vtkImageData.h"
//#include "qcutil.h"
class vtkContourFilter;
class  vtkPolyDataNormals;
class vtkPolyData;
class vtkDecimatePro;
class vtkSmoothPolyDataFilter;
class vtkStripper; 
class vtkCleanPolyData;
class vtkImageThreshold;
class vtkMarchingCubes;
class vtkImageMarchingCubes;
class vtkContourGrid;
class vtkPolyDataConnectivityFilter;


//////////////////////////////////////////////////////////////////////////
#define HTKSetMacro(name,type) \
    virtual void Set##name (type _arg) \
{ \
    this->m##name = _arg; \
} 
//////////////////////////////////////////////////////////////////////////
#define HTKGetMacro(name,type) \
    virtual type Get##name () const { \
    return this->m##name; \
} 
/*!
  *
 *\author Usaf E. Aladl (2007)
 *
 */
//////////////////////////////////////////////////////////////////////////
class vtkSurfacePipeline
{
public:
	vtkSurfacePipeline();
	~vtkSurfacePipeline();
	void SetVolume(vtkImageData* v){this->mVolume=v;};
	void SetValue(double a){ this->mValue=a;};
	void SetDecimate(bool t){this->mDecimateSurface=t;};
	void SetSmooth(bool t){this->mSmoothSurface=t;};
	void SetConnectivity(bool t){this->mConnectivitySurface=t;};
	void SetEDSmooth(bool t){this->mEDSmooth=t;};

	void SetRange(double r[2]){ this->mRange[0]=r[0]; this->mRange[1]=r[1];};
	void Create();
	void UpdateSurface();
	vtkPolyData* GetSurface(){return mSurface;};
	HTKSetMacro(DecimateTargetReduction,double);
	HTKGetMacro(DecimateTargetReduction,double);

	HTKSetMacro(DecimateMaxError,double);
	HTKGetMacro(DecimateMaxError,double);

	HTKSetMacro(DecimateFeatureAngle,double);
	HTKGetMacro(DecimateFeatureAngle,double);

protected:
	vtkImageData* mVolume;
	vtkPolyData* mSurface;
	double mValue;
	bool mDecimateSurface;
	bool mSmoothSurface;
	bool mConnectivitySurface;
	bool mEDSmooth;
	vtkImageThreshold* mThreshold;
	vtkContourFilter *mContour; 

	vtkPolyDataNormals *mNormals;
	vtkDecimatePro* mDecimatePro;
	vtkCleanPolyData *mClean;
	vtkSmoothPolyDataFilter *mSmoothing;
	vtkStripper* mStripper;
	vtkPolyDataConnectivityFilter* mConnectivityFilter;
	double mDecimateTargetReduction;
	double mDecimateMaxError;
	double mDecimateFeatureAngle;
	double mRange[2];
};


#endif