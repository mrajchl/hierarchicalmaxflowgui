#include "vtkSurface.h"

#include "vtkPolyDataReader.h"
#include "vtkPolyDataNormals.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkProperty.h"
#include "vtkProbeFilter.h"
#include "vtkImageData.h"
#include <vtkLookupTable.h>
#include <vtkImageMapToColors.h>
//////////////////////////////////////////////////////////////////////////
vtkSurface::vtkSurface()
{
mDiffuseColor[0]=1.0;
mDiffuseColor[1]=0.3882;
mDiffuseColor[2]=0.2784;

mSpecularColor[0]=1.0;
mSpecularColor[1]=0.9;
mSpecularColor[2]=0.9;

mSpecular=0.3;

mSpecularPower=30.0;

mLineWidth=1;
mOpacity=1.0;
mActor=NULL;
mSurface=NULL;
mReader=NULL;
}
//////////////////////////////////////////////////////////////////////////
vtkSurface::~vtkSurface()
{
if(mActor!=NULL)mActor->Delete();
if(mSurface!=NULL)mSurface->Delete();
}
//////////////////////////////////////////////////////////////////////////
void vtkSurface::CreateActor()
{
mNormals=vtkPolyDataNormals::New();
mNormals->SetInput(mSurface);
mNormals->ComputePointNormalsOn();
mNormals->ComputeCellNormalsOn();

mMapper=vtkPolyDataMapper::New();
mMapper->SetInput(mNormals->GetOutput());
mMapper->ScalarVisibilityOff();

mActor=vtkActor::New();
mActor->SetMapper(mMapper);
mActor->VisibilityOn();
}
//////////////////////////////////////////////////////////////////////////
void vtkSurface::ReadPolyDataFromFile(std::string str)
{
mReader=vtkPolyDataReader::New();
mReader->SetFileName(str.c_str());
mReader->Update();
mSurface=mReader->GetOutput();
}
//////////////////////////////////////////////////////////////////////////
void vtkSurface::CreatePropeActor(vtkImageData* vol)
{
this->mVolume=vol;

mNormals=vtkPolyDataNormals::New();
mNormals->SetInput(mSurface);
mNormals->ComputePointNormalsOn();
mNormals->ComputeCellNormalsOn();



mProbe = vtkProbeFilter::New();
mProbe->SetInput(mNormals->GetOutput());
mProbe->SetSource(mVolume);
mProbe->SpatialMatchOn();
mProbe->Update();



mMapper=vtkPolyDataMapper::New();
mMapper->SetInputConnection(mProbe->GetOutputPort());
mMapper->ScalarVisibilityOn();

mActor=vtkActor::New();
mActor->SetMapper(mMapper);
mActor->VisibilityOn();
}
//////////////////////////////////////////////////////////////////////////
void vtkSurface::CreatePropeActor(vtkImageData* vol, vtkLookupTable* table)
{
this->mVolume=vol;

mNormals=vtkPolyDataNormals::New();
mNormals->SetInput(mSurface);
mNormals->ComputePointNormalsOn();
mNormals->ComputeCellNormalsOn();


vtkImageMapToColors* mColorMapper= vtkImageMapToColors::New();
mColorMapper->SetOutputFormatToRGBA();
mColorMapper->PassAlphaToOutputOn();
mColorMapper->SetInput(mVolume);
mColorMapper->SetLookupTable(table);


mProbe = vtkProbeFilter::New();
mProbe->SetInput(mNormals->GetOutput());
mProbe->SetSource(mColorMapper->GetOutput());
mProbe->SpatialMatchOn();
mProbe->Update();



mMapper=vtkPolyDataMapper::New();
mMapper->SetInputConnection(mProbe->GetOutputPort());
mMapper->ScalarVisibilityOn();

mActor=vtkActor::New();
mActor->SetMapper(mMapper);
mActor->VisibilityOn();
}
//////////////////////////////////////////////////////////////////////////
void vtkSurface::UpdateColors()
{
mActor->GetProperty()->SetDiffuseColor(mDiffuseColor);
mActor->GetProperty()->SetSpecularColor(mSpecularColor);
mActor->GetProperty()->SetSpecular(mSpecular);
mActor->GetProperty()->SetSpecularPower(mSpecularPower);
mActor->GetProperty()->SetLineWidth(mLineWidth);
mActor->GetProperty()->SetOpacity(mOpacity);
mActor->GetProperty()->SetInterpolationToPhong();
mActor->GetProperty()->SetShading(true);
}