#ifndef __ORTHOGONAL_VIEW_H__
#define __ORTHOGONAL_VIEW_H__



#include <QtGui>
class QVTKWidget;
class vtkImageData;
class vtkLookupTable;
class QHBoxLayout;
class QSplitter;

class 	vtkRenderer;
class vtkRenderWindow; 
class vtkRenderWindowInteractor;
class vtkImagePlaneWidget;
class  vtkImageMapToColors;
class vtkImageActor;
class vtkIPWCallback;
class vtkSurface;
//////////////////////////////////////////////////////////////////////////
class OrthogonalView : public QWidget
{
	Q_OBJECT
public:
	OrthogonalView(QWidget *parent=0);
	virtual ~OrthogonalView( void );
	void SetInput(vtkImageData* s){this->mVolume=s;};
	void Create();
	void SetColorTable(vtkLookupTable* ct){this->mColorTable=ct;};
	vtkLookupTable* GetColorTable(){return this->mColorTable;};
	void SetBackgroundColor(double r, double g, double b);
	double* GetBackgroundColor(){return this->mColor;};
	void AddSliceControl(QWidget* w);
	void AddLights();
	void AddSurface(vtkSurface* sur);
	void Render();
	vtkRenderer* GetRenderer(){return this->mRenderer3D;};

protected:
	virtual void addSlices();
	virtual void addOutline();
	virtual void addAxes();
	virtual void addScalarBar();

protected:
	QHBoxLayout*  mLayout;
	vtkImageData* mVolume;
	vtkLookupTable* mColorTable;
	QSplitter*  mHSplitter;
	QSplitter*  mVSplitter;
	QVTKWidget*  mWidget3D;
	vtkRenderer       * mRenderer3D;
	QVTKWidget*  mWidget[3];
	vtkRenderer       * mRenderer[3];
	double  mColor[3];
	vtkImagePlaneWidget* mPlaneWidget[3];
	vtkImageMapToColors* mColorMap[3]; 
	vtkImageActor* mImageActor[3];
	vtkIPWCallback* cbk[3];
	std::vector<vtkSurface*> mSurfaces;
	QPushButton* mTButton;
public slots:
	void XChanged(int n);
	void YChanged(int n);
	void ZChanged(int n);	
	void ColorTableChanged(int n);
	void buttonClicked();

signals:
	void viewClicked(int);

};
#endif