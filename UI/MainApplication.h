#ifndef _MAIN_APPLICATION_H
#define _MAIN_APPLICATION_H


#include <QtCore>
#include <QtGui>
#include "iambusy.h"
#include "SliceViewer.h"
#include <iostream>
#include <string>
#include "vtkImageData.h"
#include "vtkMetaImageReader.h"
#include "vtkLookupTable.h"
#include "OrthogonalView.h"
#include "QVTKWidget.h"
#include "qcolortablewidget.h"
#include "QRangeSlider.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkRenderer.h"
#include "vtkImageGaussianSmooth.h"
#include "vtkCamera.h"
#include "vtkMetaImageWriter.h"
#include "vtkImageClip.h"
#include "ui_xyzwidget.h"
#include "vtkImageShrink3D.h"
#include "vtkImageShiftScale.h"
#include "vtkImageIdealHighPass.h"
#include "vtkImageExtractComponents.h"
#include "vtkImageAnisotropicDiffusion3D.h"
#include "vtkImageLaplacian.h"
#include "vtkImageCast.h"
#include "vtkImageMathematics.h"
#include "vtkImageGradientMagnitude.h"
#include "vtkImageGradient.h"
#include "vtkImageMagnitude.h"
#include "vtkImageNonMaximumSuppression.h"
#include "vtkImageSkeleton2D.h"
#include "vtkImageSobel3D.h"
#include "vtkImageVariance3D.h"
#include "vtkSurfacePipeline.h"
#include "vtkSurface.h"



#include "runMaxFlow.h"
#include "vtkImageShiftScale.h"

#include "vtkContourFilter.h"
//////////////////////////////////////////////////////////////////////////
class MColor
{
public:
  MColor(double _r=0.0, double _g=0.0, double _b=0.0):r(_r),g(_g), b(_b){};
  double r, g, b;
} ;
//////////////////////////////////////////////////////////////////////////
class MainApplication : public QMainWindow
{
	
  Q_OBJECT
  public:
  MainApplication(QWidget *parent = 0);
protected:
  void closeEvent(QCloseEvent *event);
  bool isSaveToClose();
  void createMenus();
  void createActions();
  void createLeftDockWindow( );
  void createRightDockWindow( );
  void createControls();
  void connectControls();
  void createComboLabels();
  void createBin();
  void createLabels();
  void updateRange();
  void applyMask();
protected:
  QStackedWidget* mStackedWidget;
  QSplitter*  mSplitter;
  QSplitter*  mLeftSplitter;
  QSplitter*  mRightSplitter;
  QColor mBackColor;
  QDockWidget *mLeftDock;
  QDockWidget *mSmoothDock;
  QDockWidget *mRightDock;
  std::string mFileName;
  vtkImageData* mVolume;
  vtkImageData* mBin;
  vtkImageData* mCut;
  vtkLookupTable* mLabels;
  vtkLookupTable* mLabels2;
  SliceViewer* mViewer[3];
  OrthogonalView* mOView;
  int mSliceNumber[3];
  QComboBox* mLabelCombo;
  QSlider* mAlphaSlider;
  QSpinBox* mRadiusSpinBox;
  QDoubleSpinBox* mTValphaSpinBox11;
  QDoubleSpinBox* mTValphaSpinBox12;
  QDoubleSpinBox* mTValphaSpinBox21;
  QDoubleSpinBox* mTValphaSpinBox22;
  std::vector<MColor> mColors;
  std::vector<QString> mTextLabels;
  Ui::XYZWidget* mXYZWidget;
  QRangeSlider* mRangeSlider;
  QColorTableWidget* mColorWidget;
  QProgressBar *mPb;
  std::vector<double> mValues;
  vtkSurfacePipeline * mSPP;
  QAction* mAboutAct;
protected:
  QMenu *mHelpMenu;
  QMenu *mFileMenu;
  QMenu *saveAsMenu;
  QSpinBox* mSliceSpinBox;
  QButtonGroup* mOrientationGroup; 
  QPushButton* mViewBin;
  QPushButton* mSmooth;
  QPushButton *mMaxFlow;
  QPushButton* mISO;
  QPushButton* mClearBinAndCut;
	
public slots:
  void about();
  void loadVolume();
  void display();
  void rangeSlot(int l, int u);
  void colorSlot(int);
  void saveBin();
  void saveCut();
  void loadBin();
  void Refresh();
  void smoothVolume();
  void calcMaxFlow();
  vtkImageData*  mashSmooth(vtkImageData* vol,int itr, double lam );
  void isoView();
  void cleanStack();
  void viewAll();
  void stackIndex(int);
  void saveVolume();
  void maskVolume();
  void clearBinAndCut();
};

#endif
