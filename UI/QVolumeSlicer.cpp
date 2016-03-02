#include "QVolumeSlicer.h"
#include "vtkImageData.h"
#include "vtkImageMapToColors.h"
#include "vtkImageReslice.h"
#include "vtkLookupTable.h"
#include "vtkMath.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include <qimage.h>
//////////////////////////////////////////////////////////////////////////
QVolumeSlicer::QVolumeSlicer()
{
mPosition=0.0;
mVolume=NULL;
mPlaneOrientation=0;
mImage=new QImage(10,10,QImage::Format_RGB32);
}
//////////////////////////////////////////////////////////////////////////
QVolumeSlicer::~QVolumeSlicer()
{

}
//////////////////////////////////////////////////////////////////////////
void QVolumeSlicer::Create()
{
if(this->mVolume==NULL)return;
this->mReslice= vtkImageReslice::New();
this->mReslice->SetInput(this->mVolume);
this->mReslice->SetResliceTransform(vtkTransform::New());
this->mReslice->SetInterpolationModeToCubic();
this->mReslice->InterpolateOff();
this->mReslice->SetBackgroundLevel(0);
this->SetOutput(10.0);


this->mColorMapper= vtkImageMapToColors::New();
this->mColorMapper->SetOutputFormatToRGBA();
this->mColorMapper->PassAlphaToOutputOn();
this->mColorMapper->SetInputConnection(mReslice->GetOutputPort());
this->mColorMapper->SetLookupTable(this->mColorTable);
this->mColorMapper->Update();



}

//////////////////////////////////////////////////////////////////////////
void QVolumeSlicer::SetOutput(double slice)
{
 mPosition=slice;
mVolume->UpdateInformation();

int*    extent = mVolume->GetWholeExtent();
double*    origin = mVolume->GetOrigin();
double*    spacing = mVolume->GetSpacing();

mReslice->SetOutputSpacing(spacing);

double ox=-0.5*spacing[0]*extent[1];
double oy=-0.5*spacing[1]*extent[3];
double oz=-0.5*spacing[2]*extent[5];

if(this->mPlaneOrientation==0) //XY
{
//mReslice->SetOutputExtent(extent[0],extent[1],extent[2],extent[3],slice/spacing[2],slice/spacing[2]);
mReslice->SetOutputExtent(extent[0],extent[1],extent[2],extent[3],slice,slice);
mReslice->SetOutputOrigin(ox,oy,slice);
                   
}


if(this->mPlaneOrientation==1) //YZ
{
//mReslice->SetOutputExtent(slice/spacing[0],slice/spacing[0],extent[2],extent[3],extent[4],extent[5]);
mReslice->SetOutputExtent(slice,slice,extent[2],extent[3],extent[4],extent[5]);
mReslice->SetOutputOrigin(slice,oy,oz);
}



if(this->mPlaneOrientation==2) //ZX
{
//mReslice->SetOutputExtent(extent[0],extent[1],slice/spacing[1],slice/spacing[1],extent[4],extent[5]);
mReslice->SetOutputExtent(extent[0],extent[1],slice,slice,extent[4],extent[5]);
mReslice->SetOutputOrigin(ox,slice,oz);
}
mReslice->Update();

}
//////////////////////////////////////////////////////////////////////////
QImage* QVolumeSlicer::GetOutput()
{
mReslice->Update();
vtkImageData* out=this->mColorMapper->GetOutput();
unsigned char* buffer= (unsigned char*)out->GetScalarPointer();
int ext[6];
out->GetExtent(ext);
int w,h;
if(this->mPlaneOrientation==0) //XY
{
w=ext[1]+1;
h=ext[3]+1;
           
}


if(this->mPlaneOrientation==1) //YZ
{
w=ext[3]+1;
h=ext[5]+1;
}



if(this->mPlaneOrientation==2) //ZX
{
w=ext[1]+1;
h=ext[5]+1;
}

QImage im(buffer, w, h, QImage::Format_RGB32);
//mImage=new QImage(buffer, w, h, 32, NULL, 0, QImage::IgnoreEndian);
*mImage=im.copy();

return this->mImage;
}