#ifndef __Q_VOLUME_SLICER_H__
#define  __Q_VOLUME_SLICER_H__


class vtkImageData;
class vtkImageReslice;
class vtkImageMapToColors;
class vtkLookupTable;
class QImage;
//////////////////////////////////////////////////////////////////////////
class QVolumeSlicer
{
public:
	QVolumeSlicer();
	~QVolumeSlicer();
	void SetInput(vtkImageData* im){this->mVolume=im;};
	void SetColorTable(vtkLookupTable* ct){this->mColorTable=ct;};
	void SetPlaneOrientation(int i){this->mPlaneOrientation=i;};
	void SetOutput(double slice);
	double GetPosition(){return this->mPosition;};
	void Create();
	QImage* GetOutput();
protected:
	vtkImageData         *mVolume;
	vtkImageReslice      *mReslice;
	vtkImageMapToColors  *mColorMapper;
	vtkLookupTable       *mColorTable;
	QImage* mImage;
 	int mPlaneOrientation;
	double mPosition;
};


#endif