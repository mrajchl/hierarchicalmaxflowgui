#include "OrthogonalView.h"
#include <qsplitter.h>
#include <qlayout.h>
#include "ui_xyzwidget.h"
#include <qspinbox.h>
#include <qslider.h>

#include "QVTKWidget.h"

#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkLookupTable.h"
#include "vtkImagePlaneWidget.h"
#include "vtkOutlineFilter.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkImageData.h"
#include "vtkCellPicker.h"
#include "vtkActor.h"
#include "vtkAxesActor.h"
#include "vtkTextProperty.h"
#include "vtkProperty.h"
#include "vtkOrientationMarkerWidget.h"
#include "vtkCaptionActor2D.h"
#include "vtkScalarBarActor.h"
#include "vtkScalarBarWidget.h"
#include "vtkCamera.h"
#include "vtkCommand.h"

#include "vtkImageMapToColors.h"
#include "vtkImageActor.h"


#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkInteractorStyleImage.h"
#include "vtkRenderer.h"
#include "vtkSurface.h"
#include "vtkLightKit.h"

//////////////////////////////////////////////////////////////////////////

// from kbarcode
static const char* remove_xpm[]=
{
    "16 16 15 1",
    " 	c None",
    ".	c #B7B7B7",
    "+	c #FFFFFF",
    "@	c #6E6E6E",
    "#	c #E9E9E9",
    "$	c #E4E4E4",
    "%	c #000000",
    "&	c #DEDEDE",
    "*	c #D9D9D9",
    "=	c #D4D4D4",
    "-	c #CECECE",
    ";	c #C9C9C9",
    ">	c #C3C3C3",
    ",	c #BEBEBE",
    "'	c #B9B9B9",

    "...............+",
    ".@@@@@@@@@@@@@@+",
    ".@+++++++++++.@+",
    ".@+          .@+",
    ".@+  %    %  .@+",
    ".@+ %%%  %%% .@+",
    ".@+  %%%%%%  .@+",
    ".@+   %%%%   .@+",
    ".@+   %%%%   .@+",
    ".@+  %%%%%%  .@+",
    ".@+ %%%  %%% .@+",
    ".@+  %    %  .@+",
    ".@+           @+",
    ".@............@+",
    ".@@@@@@@@@@@@@@+",
    "++++++++++++++++"
};


//////////////////////////////////////////////////////////////////////////
class vtkIPWCallback : public vtkCommand
{
public:
  static vtkIPWCallback *New()
    { return new vtkIPWCallback; }
  virtual void Execute(vtkObject *caller, unsigned long, void*)
    {
      vtkImagePlaneWidget *widget =
        reinterpret_cast<vtkImagePlaneWidget*>(caller);
      if(!widget) { return; }

      ColorMap->UpdateWholeExtent();   
      
      ImageActor->SetDisplayExtent(ImageActor->GetInput()->GetWholeExtent());
	//	Slider->setValue( widget->GetSliceIndex());
        RWindow->Render();         
    }

  vtkIPWCallback():Renderer(0),ImageActor(0),ColorMap(0){}

	vtkImageMapToColors* ColorMap;
	vtkImageActor* ImageActor;
	vtkRenderer           *Renderer;
	vtkRenderWindow* RWindow; 
QSlider* Slider;
};

//////////////////////////////////////////////////////////////////////////
OrthogonalView::OrthogonalView(QWidget *parent)
:QWidget(parent)
{
//UXX setBackgroundColor(Qt::black);	
setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);	
mLayout = new QHBoxLayout( this); 

mVolume=NULL;
mColorTable=NULL;

mPlaneWidget[0]=NULL;
mPlaneWidget[1]=NULL;
mPlaneWidget[2]=NULL;


QColor cc(25,51,102);
//UXX this->setBackgroundColor(cc);
mColor[0]=0.1;
mColor[1]=0.2;
mColor[2]=0.4;

mColor[0]=0.0;
mColor[1]=0.0;
mColor[2]=0.0;
//mColor[0]=0.674509;
//mColor[1]=0.6588;
//mColor[2]=0.6;
mHSplitter=new QSplitter(Qt::Horizontal,this);
//UXX mHSplitter->setBackgroundColor(cc);
mHSplitter->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
mHSplitter->show();
mLayout->addWidget(mHSplitter);

mWidget3D=new QVTKWidget(mHSplitter);
mWidget3D->show();
mWidget3D->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
 

mRenderer3D = vtkRenderer::New();
mWidget3D->GetRenderWindow()->AddRenderer(mRenderer3D);
mRenderer3D->Delete();
mRenderer3D->SetBackground(mColor[0],mColor[1],mColor[2]);


//mVSplitter=new QSplitter(QSplitter::Vertical,mHSplitter);
//mVSplitter->setBackgroundColor(cc);
//mVSplitter->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
//mVSplitter->show();



resize( QSize(300, 300).expandedTo(minimumSizeHint()) );



}
//////////////////////////////////////////////////////////////////////////
OrthogonalView::~OrthogonalView()
{

}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::Create()
{

if(mColorTable==NULL)
{
double l[2];
l[0]=mVolume->GetScalarTypeMin();
l[1]=mVolume->GetScalarTypeMax();
mColorTable=vtkLookupTable::New();
mColorTable->SetAlpha(1.0);
mColorTable->SetNumberOfColors(256);
mColorTable->SetRange(l);
mColorTable->Build();

double c;
 
for(int i=0; i<256; i++)
{
c=(double)i/255.0;	
mColorTable->SetTableValue(i,c,c,c);
}


}	


this->addOutline();
this->addAxes();
//this->addScalarBar();	
this->addSlices();
this->AddLights();
QSizePolicy sizePolicy( QSizePolicy::Fixed,QSizePolicy::Preferred);
    sizePolicy.setHorizontalStretch(0);
    sizePolicy.setVerticalStretch(0);
 QPixmap* pixx;
pixx= new QPixmap(remove_xpm);
mTButton=new QPushButton(mWidget3D);
mTButton->setIcon(QIcon(*pixx));
mTButton->setIconSize(QSize(22,22));
mTButton->resize(22, 22);
mTButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
connect( mTButton, SIGNAL( clicked() ), this, SLOT( buttonClicked() ) );
mTButton->show();


mWidget3D->GetRenderWindow()->Render();
mRenderer3D->GetActiveCamera()->SetFocalPoint(0,0,0);		
mRenderer3D->GetActiveCamera()->SetPosition(1,-1,1);		
mRenderer3D->GetActiveCamera()->SetViewUp(0,0,1);		
mRenderer3D->ResetCamera();
mWidget3D->GetRenderWindow()->Render();



}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::Render()
{
mWidget3D->GetRenderWindow()->Render();

}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::addOutline()
{
	if(mVolume==NULL)return;



  vtkOutlineFilter* outline = vtkOutlineFilter::New();
    outline->SetInput(mVolume);

  vtkPolyDataMapper* outlineMapper = vtkPolyDataMapper::New();
    outlineMapper->SetInputConnection(outline->GetOutputPort());

  vtkActor* outlineActor =  vtkActor::New();
    outlineActor->SetMapper( outlineMapper);



mRenderer3D->AddActor( outlineActor);



	
}

//////////////////////////////////////////////////////////////////////////
void OrthogonalView::addAxes()
{
vtkTextProperty* tprop1=vtkTextProperty::New(); 
tprop1->ItalicOn();
tprop1->ShadowOn();
tprop1->SetFontFamilyToTimes();


vtkAxesActor* axes=vtkAxesActor::New();
axes->SetShaftTypeToCylinder();
axes->SetXAxisLabelText("x");
axes->SetYAxisLabelText("y");
axes->SetZAxisLabelText("z");
axes->SetNormalizedShaftLength( 0.85, 0.85, 0.85 );
axes->SetNormalizedTipLength( 0.15, 0.15, 0.15 );
axes->SetTotalLength(10.5, 10.5, 10.5);
axes->GetXAxisCaptionActor2D()->SetCaptionTextProperty(tprop1);
axes->GetYAxisCaptionActor2D()->SetCaptionTextProperty(tprop1);
axes->GetZAxisCaptionActor2D()->SetCaptionTextProperty(tprop1);

tprop1->Delete();


vtkProperty* property = axes->GetXAxisTipProperty();
property->SetRepresentationToWireframe();
property->SetDiffuse(0);
property->SetAmbient(1);
property->SetColor( 1, 0, 1 );

property = axes->GetYAxisTipProperty();
property->SetRepresentationToWireframe();
property->SetDiffuse(0);
property->SetAmbient(1);
property->SetColor( 1, 1, 0 );

property = axes->GetZAxisTipProperty();
property->SetRepresentationToWireframe();
property->SetDiffuse(0);
property->SetAmbient(1);
property->SetColor( 0, 1, 1 );

vtkOrientationMarkerWidget* marker=vtkOrientationMarkerWidget::New();
marker->SetOutlineColor(0.93, 0.57, 0.13);
marker->SetOrientationMarker(axes);
marker->SetViewport(0.0, 0.0, 0.15, 0.3);
marker->SetInteractor(mWidget3D->GetInteractor());
marker->SetEnabled(1);
marker->InteractiveOff();

axes->Delete();
}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::addScalarBar()
{

vtkScalarBarWidget *scalarWidget=vtkScalarBarWidget::New();
scalarWidget->SetInteractor(mWidget3D->GetInteractor());

vtkScalarBarActor* scalarBar=scalarWidget->GetScalarBarActor();
scalarBar->SetLookupTable(mColorTable);
scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
scalarBar->GetPositionCoordinate()->SetValue(0.95,0.77);
scalarBar->SetOrientationToVertical();
scalarBar->SetWidth(0.05);
scalarBar->SetHeight(0.25);
scalarBar->SetNumberOfLabels(3);
scalarWidget->EnabledOn();
}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::addSlices()
{

int ext[6];
mVolume->GetWholeExtent(ext); 

int ss[3];
ss[0]=(ext[1]-ext[0])/2;
ss[1]=(ext[3]-ext[2])/2;
ss[2]=(ext[5]-ext[4])/2;

if(ext[5]==0)ss[0]=0;
if(ext[5]==0)ss[1]=0;

double c[3][3];
c[0][0]=1.0;
c[0][1]=0.0;
c[0][2]=0.0;

c[1][0]=1.0;
c[1][1]=1.0;
c[1][2]=0.0;

c[2][0]=0.0;
c[2][1]=0.0;
c[2][2]=1.0;

char k[3];
k[0]='x';
k[1]='y';
k[2]='z';
int i;
vtkCellPicker* sharedPicker = vtkCellPicker::New();
sharedPicker->SetTolerance(0.01);

for(i=0; i<3; i++)
{
	if(mPlaneWidget[i]!=NULL){
		mPlaneWidget[i]->EnabledOff();
		mPlaneWidget[i]->Delete();
	}
 vtkProperty* ipwProp = vtkProperty::New();

mPlaneWidget[i] = vtkImagePlaneWidget::New();
mPlaneWidget[i]->SetInteractor( mWidget3D->GetInteractor());
mPlaneWidget[i]->SetKeyPressActivationValue(k[i]);
mPlaneWidget[i]->SetPicker(sharedPicker);
mPlaneWidget[i]->GetPlaneProperty()->SetColor(c[i]);
mPlaneWidget[i]->SetTexturePlaneProperty(ipwProp);
mPlaneWidget[i]->TextureInterpolateOn();
mPlaneWidget[i]->SetResliceInterpolateToLinear();
mPlaneWidget[i]->SetInput(mVolume);
mPlaneWidget[i]->SetPlaneOrientation(i);
mPlaneWidget[i]->SetSliceIndex(ss[i]);

if(mVolume->GetNumberOfScalarComponents()==1)
{
mPlaneWidget[i]->UserControlledLookupTableOn(); 
mPlaneWidget[i]->SetLookupTable( mColorTable);
}else{
mPlaneWidget[i]->GetColorMap()->SetOutputFormatToRGB();
mPlaneWidget[i]->GetColorMap()->SetLookupTable(NULL);

}
mPlaneWidget[i]->DisplayTextOn();
mPlaneWidget[i]->RestrictPlaneToVolumeOn();
mPlaneWidget[i]->On();
mPlaneWidget[i]->EnabledOn();
mPlaneWidget[i]->GetPlaneProperty()->SetOpacity(1.0); 




}



	
}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::XChanged(int n)
{
if(mPlaneWidget[0]!=NULL)mPlaneWidget[0]->SetSliceIndex(n);
this->Render();
}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::YChanged(int n)
{
if(mPlaneWidget[1]!=NULL)mPlaneWidget[1]->SetSliceIndex(n);
this->Render();
}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::ZChanged(int n)
{
if(mPlaneWidget[2]!=NULL)mPlaneWidget[2]->SetSliceIndex(n);
this->Render();
}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::AddSliceControl(QWidget *w)
{
/*UXX
int ext[6];

mVolume->GetWholeExtent(ext); 
XYZWidget* xyzw;
if(w==0)xyzw=new XYZWidget(this,"Slices Controls", WType_TopLevel|WStyle_Tool|WDestructiveClose);
else xyzw=new XYZWidget(w,"Slices Controls");

QRect rect3(10,10,180,100);
if(w==mWidget3D)xyzw->setGeometry(rect3);
xyzw->setPalette (QPalette(QColor(118,118,118))) ;

xyzw->xSlider->setMaxValue( ext[1] );
xyzw->xSpinBox->setMaxValue(ext[1]);
xyzw->xSlider->setValue( mPlaneWidget[0]->GetSliceIndex());
cbk[0]->Slider=xyzw->xSlider;
connect( xyzw->xSlider, SIGNAL( valueChanged(int) ), this, SLOT( XChanged(int) ) );

xyzw->ySlider->setMaxValue( ext[3] );
xyzw->ySpinBox->setMaxValue(ext[3]);
xyzw->ySlider->setValue( mPlaneWidget[1]->GetSliceIndex());
cbk[1]->Slider=xyzw->ySlider;
connect( xyzw->ySlider, SIGNAL( valueChanged(int) ), this, SLOT( YChanged(int) ) );


xyzw->zSlider->setMaxValue( ext[5]);
xyzw->zSpinBox->setMaxValue(ext[5]);
xyzw->zSlider->setValue( mPlaneWidget[2]->GetSliceIndex() );
cbk[2]->Slider=xyzw->zSlider;
connect( xyzw->zSlider, SIGNAL( valueChanged(int) ), this, SLOT( ZChanged(int) ) );

xyzw->show();
*/
}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::SetBackgroundColor(double r, double g, double b)
{
mColor[0]=r;
mColor[1]=g;
mColor[2]=b;

mRenderer3D->SetBackground(mColor[0],mColor[1],mColor[2]);
mRenderer[0]->SetBackground(mColor[0],mColor[1],mColor[2]);
mRenderer[1]->SetBackground(mColor[0],mColor[1],mColor[2]);
mRenderer[2]->SetBackground(mColor[0],mColor[1],mColor[2]);

}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::ColorTableChanged(int n)
{
//mColorTable->SetColorTableNumber(n);
mColorTable->Build();
this->Render();
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::AddLights()
{

vtkLightKit* lightKit=vtkLightKit::New();
lightKit->AddLightsToRenderer(mRenderer3D);


}
//////////////////////////////////////////////////////////////////////////
void OrthogonalView::AddSurface(vtkSurface* sur)
{
int val=mSurfaces.size();
mSurfaces.push_back(sur);
vtkActor* actor=sur->GetActor();
mRenderer3D->AddActor(actor);
	

}

//////////////////////////////////////////////////////////////////////////
void OrthogonalView::buttonClicked()
{
emit viewClicked(3);
}
