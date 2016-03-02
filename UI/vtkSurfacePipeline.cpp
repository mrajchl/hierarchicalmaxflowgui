#include "vtkSurfacePipeline.h"

#include <vtkCellArray.h>
#include <vtkImageData.h>
#include <vtkImageImport.h>
#include <vtkImageGaussianSmooth.h>
#include <vtkImageThreshold.h>
#include <vtkImageToStructuredPoints.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkStripper.h>
#include <vtkCallbackCommand.h>
#include <vtkContourFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkDecimatePro.h>
#include "vtkCleanPolyData.h"
#include <vtkMarchingCubes.h>
#include <vtkImageMarchingCubes.h>
#include <vtkPolyDataConnectivityFilter.h>
 #include <vtkContourGrid.h>
  
//////////////////////////////////////////////////////////////////////////
vtkSurfacePipeline::vtkSurfacePipeline()
{
mDecimateSurface=true;
mSmoothSurface=true;
mConnectivitySurface=true;
mEDSmooth=false;
mRange[0]=3.0;
mRange[1]=3.0;

mValue=mRange[0];

mDecimateTargetReduction=0.5;
mDecimateMaxError=VTK_DOUBLE_MAX;
mDecimateFeatureAngle=65.;



}
//////////////////////////////////////////////////////////////////////////
vtkSurfacePipeline::~vtkSurfacePipeline()
{

	
}
//////////////////////////////////////////////////////////////////////////
void vtkSurfacePipeline::Create()
{

mThreshold=vtkImageThreshold::New();
mThreshold->SetOutValue(0);
mThreshold->ReplaceOutOn();
mThreshold->ReplaceInOff();

mContour=vtkContourFilter::New();

//mContour->ReleaseDataFlagOn();
mContour->ComputeNormalsOff();
mContour->ComputeScalarsOff();
mContour->ComputeGradientsOff();
mContour->UseScalarTreeOn();
mContour->SetNumberOfContours(1);
mContour->SetValue(0,0.0f);

mConnectivityFilter=vtkPolyDataConnectivityFilter::New();
mConnectivityFilter->SetExtractionModeToLargestRegion();
//mConnectivityFilter->ScalarConnectivityOn();
//mConnectivityFilter->SetExtractionModeToAllRegions();

mClean=vtkCleanPolyData::New();
mClean->SetPointMerging(true);
mClean->SetTolerance(0.0);//It should be ZERO

mNormals= vtkPolyDataNormals::New();
mNormals->SplittingOff();
mNormals->ConsistencyOff();
mNormals->ReleaseDataFlagOn();
mNormals->ComputePointNormalsOn();
mNormals->ComputeCellNormalsOn();


mDecimatePro= vtkDecimatePro::New();
mDecimatePro->ReleaseDataFlagOn();


mSmoothing=vtkSmoothPolyDataFilter::New();
mSmoothing->ReleaseDataFlagOn();

mStripper= vtkStripper::New();
mStripper->ReleaseDataFlagOn();


}
//////////////////////////////////////////////////////////////////////////
void vtkSurfacePipeline::UpdateSurface()
{
vtkPolyData *pipePoly= NULL;

mThreshold->SetInput(mVolume);
mThreshold->ThresholdBetween(mRange[0], mRange[1]);
mThreshold->SetOutputScalarType(mVolume->GetScalarType());
mThreshold->Update();
double r[2];
mThreshold->GetOutput()->GetScalarRange(r);

//std::cout<<"PR1="<<r[0]<<" PR2="<<r[1]<<std::endl;
mContour->SetInput(mThreshold->GetOutput());




//mContour->SetInput(mVolume);

mContour->SetValue(0, mRange[0]);
//mContour->SetValue(0,mValue);
pipePoly=mContour->GetOutput();

if(mConnectivitySurface)
{
mConnectivityFilter->SetInput(pipePoly);
pipePoly=mConnectivityFilter->GetOutput();
int m=mConnectivityFilter->GetNumberOfExtractedRegions();
std::cout<<"Regions= "<<m<<std::endl;
}



if(mDecimateSurface)
{
mDecimatePro->SetInput(pipePoly);
mDecimatePro->SetTargetReduction(mDecimateTargetReduction); //0.5
mDecimatePro->PreserveTopologyOff();  //Off
mDecimatePro->SetFeatureAngle(mDecimateFeatureAngle); //50
mDecimatePro->SplittingOn();
mDecimatePro->BoundaryVertexDeletionOn();
mDecimatePro->SetMaximumError(mDecimateMaxError);
pipePoly= mDecimatePro->GetOutput();
}

mClean->SetInput(pipePoly);
pipePoly= mClean->GetOutput();	


if(mSmoothSurface)
{
mSmoothing->SetInput(pipePoly);
mSmoothing->SetNumberOfIterations(500);
mSmoothing->SetRelaxationFactor(0.95);  //0.75
mSmoothing->SetConvergence(0.05);//0.5
mSmoothing->SetFeatureAngle(60.0);//60.0
pipePoly=mSmoothing->GetOutput();
}


//mStripper->SetInput(pipePoly);
//pipePoly=mStripper->GetOutput();



mNormals->SetInput(pipePoly);
pipePoly= mNormals->GetOutput();



mSurface=pipePoly;
}
