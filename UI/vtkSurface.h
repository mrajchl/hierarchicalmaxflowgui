#ifndef _VTK_SURFACE_H_
#define _VTK_SURFACE_H_

#include <vector>
#include <iostream>
#include <qstring.h>
#include <string>

//!
/*!
 * This file contains 
 *
 *\author Usaf E. Aladl 2005
 *
 *
 *\b Revisions:
 * - 2006/05 first implementation
 * - 
 */

class vtkPolyData;
class vtkPolyDataNormals;
class vtkPolyDataMapper;
class vtkPolyDataReader;
class vtkActor;
class vtkImageData;
class vtkProbeFilter;
class vtkLookupTable;
//////////////////////////////////////////////////////////////////////////
class vtkSurface
{
public:
	vtkSurface();
	~vtkSurface();
	void SetPolyData(vtkPolyData* p){this->mSurface=p;};
	void ReadPolyDataFromFile(std::string str);
	void CreateActor();
	void CreatePropeActor(vtkImageData*);
	void CreatePropeActor(vtkImageData*,vtkLookupTable*);
	vtkPolyData* GetPolyData(){return this->mSurface;};
	vtkActor* GetActor(){return this->mActor;};
	void SetDiffuseColor(double* c){mDiffuseColor[0]=c[0];  mDiffuseColor[1]=c[1];  mDiffuseColor[2]=c[2];	};
	void SetDiffuseColor(double r, double g, double b){mDiffuseColor[0]=r;  mDiffuseColor[1]=g;  mDiffuseColor[2]=b;	};
	void SetLineWidth(int n){this->mLineWidth=n;};
	void SetOpacity(double a){this->mOpacity=a;};
	void UpdateColors();
protected:
	vtkPolyData* mSurface;
	vtkPolyDataNormals* mNormals;
	vtkPolyDataMapper* mMapper;
	vtkActor* mActor;
	vtkPolyDataReader* mReader;
	vtkImageData* mVolume;
	vtkProbeFilter* mProbe;
	double mDiffuseColor[3];
	double mSpecularColor[3];
	double mSpecular;
	double mSpecularPower;
	double mLineWidth;
	double mOpacity;

};



#endif