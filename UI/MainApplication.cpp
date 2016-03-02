#include "MainApplication.h"

vtkImageData* doQuat(double t);
//////////////////////////////////////////////////////////////////////////
MainApplication::MainApplication(QWidget *parent) : QMainWindow(parent){
  mVolume=NULL;
  mViewer[0]=NULL;
  mViewer[1]=NULL;
  mViewer[2]=NULL;
  mOView=NULL;

  mBackColor.setRgb(172,168,153);

  mStackedWidget=new QStackedWidget;
  setCentralWidget(mStackedWidget); 


  mSplitter=new QSplitter(Qt::Horizontal,this);
  mSplitter->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
  //mSplitter->show();
  //setCentralWidget( mSplitter );

  mStackedWidget->addWidget(mSplitter);

  QPalette palette;
  palette.setColor(mSplitter->backgroundRole(), mBackColor);
  mSplitter->setPalette(palette);
  mLeftSplitter=new QSplitter(Qt::Vertical,mSplitter);
  mLeftSplitter->setFrameStyle(QFrame::NoFrame);
  mLeftSplitter->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
  mLeftSplitter->show();

  mRightSplitter=new QSplitter(Qt::Vertical, mSplitter);
  mRightSplitter->setFrameStyle(QFrame::NoFrame);
  mRightSplitter->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
  mRightSplitter->show();


  createLabels();
  createControls();
  createMenus();
  createActions();
  resize(800,600);
  mPb= new QProgressBar(statusBar());
  mPb->setTextVisible(false);
  mPb->hide();
  statusBar()->addPermanentWidget(mPb);
  statusBar()->showMessage( "Ready", 2000 );

};

//////////////////////////////////////////////////////////////////////////
bool MainApplication::isSaveToClose()
{
  
  int ret = QMessageBox::warning(this, tr("MaxFlowGUI"),
				 tr("Do you want to exit?"),
				 QMessageBox::Yes | QMessageBox::Default,   QMessageBox::No|QMessageBox::Escape);
  if (ret == QMessageBox::Yes)
    return true;
  else if (ret == QMessageBox::No)
    return false;
  
  return true;
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::closeEvent(QCloseEvent *event)
{
  event->accept();

  if (isSaveToClose()) 
    {
      event->accept();
    } else {
    event->ignore();
  }
}

//////////////////////////////////////////////////////////////////////////
void MainApplication::about()
{

  QMessageBox msg(this);
  msg.setIconPixmap(QIcon(":/images/ss.svg").pixmap(QSize(128,128)));


  QString str =
    "<h2><center>MaxFlowGUI </center></h2>"
    "<p><h3>Martin Rajchl (2011)</h3>"
    "<p><h3>Usaf Aladl    (2011)</h3>"
    "<p>Email: mrajchl@imaging.robarts.ca"
    "<p>A graphical user interface for interactive continuous max-flow graph cut segmentation."
    "<p>Supported input formats:"
    "<blockquote>";
  str+="<p>MetaImage ";
  str += "</blockquote>";
	
  msg.setText(str); 
  msg.exec();

}
//////////////////////////////////////////////////////////////////////////
void MainApplication::createMenus()
{
  mFileMenu = menuBar()->addMenu(tr("&File"));
  mHelpMenu = menuBar()->addMenu(tr("&Help"));

}

//////////////////////////////////////////////////////////////////////////
void MainApplication::createActions()
{
  QAction* act;

  act = new QAction(tr("About"), this);
  act->setShortcut(tr("F1"));
  connect(act, SIGNAL(triggered()), this, SLOT(about()));
  mHelpMenu->addAction(act);

  act = new QAction(tr("Load Volume"), this);
  act->setStatusTip(tr("Load volume file"));
  connect(act , SIGNAL(activated()), this, SLOT(loadVolume()));
  mFileMenu->addAction(act);
  mFileMenu->addSeparator();

  act = new QAction(tr("Load Bin"), this);
  act ->setStatusTip(tr("Load volume bin file"));
  connect(act , SIGNAL(activated()), this, SLOT(loadBin()));
  mFileMenu->addAction(act);
  mFileMenu->addSeparator();


  saveAsMenu= new QMenu(tr("Save"),this);
  mFileMenu->addMenu(saveAsMenu);
  mFileMenu->addSeparator();

  act = new QAction(tr("Volume"), this);
  act->setStatusTip(tr("Save volume..."));
  connect(act, SIGNAL(activated()), this, SLOT(saveVolume()));
  saveAsMenu->addAction(act);

  act = new QAction(tr("Bin Volume"), this);
  act->setStatusTip(tr("Save binary volume..."));
  connect(act, SIGNAL(activated()), this, SLOT(saveBin()));
  saveAsMenu->addAction(act);
  act = new QAction(tr("Cut Volume"), this);
  act->setStatusTip(tr("Save cut volume..."));
  connect(act, SIGNAL(activated()), this, SLOT(saveCut()));
  saveAsMenu->addAction(act);
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::createLeftDockWindow( )
{
  mLeftDock = new QDockWidget( tr("Controls"), this);
  mLeftDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  mLeftDock->setFeatures(QDockWidget::DockWidgetMovable| QDockWidget::DockWidgetFloatable);

  QPalette p=mLeftDock->style()->standardPalette();
  p.setColor(QPalette::Background, QColor(191, 215, 191));
  p.setColor(QPalette::Background, mBackColor);
  //p.setColor(QPalette::Background, QColor(191, 215, 191));
  mLeftDock->setBackgroundRole(QPalette::Window);
  mLeftDock->setAutoFillBackground(true);
  mLeftDock->setPalette(p);

  addDockWidget(Qt::LeftDockWidgetArea, mLeftDock);
}

//////////////////////////////////////////////////////////////////////////
void MainApplication::createRightDockWindow( )
{

}
//////////////////////////////////////////////////////////////////////////
void MainApplication::createControls()
{
  createLeftDockWindow( );

  QFrame* w0=new QFrame;
  w0->setFrameStyle(QFrame::WinPanel | QFrame::Sunken);
  w0->setLineWidth(2);
  QVBoxLayout* l= new QVBoxLayout;
  w0->setLayout(l);

  mLeftDock->setWidget(w0);

  mColorWidget=new QColorTableWidget;
  connect(mColorWidget, SIGNAL( activated(int) ), this, SLOT( colorSlot(int) ) );
  l->addWidget(mColorWidget);


  mRangeSlider=new QRangeSlider;
  mRangeSlider->setMinValue(0);
  mRangeSlider->setMaxValue(255);
  mRangeSlider->setLower(0);
  mRangeSlider->setUpper(255);
  connect(mRangeSlider, SIGNAL( valueChanged(int,int) ), this, SLOT( rangeSlot(int,int) ) );

  l->addWidget(mRangeSlider);

  mSmooth = new QPushButton;
  mSmooth->setText( tr( "Smooth Volume" ) );
  l->addWidget(mSmooth);
  connect( mSmooth, SIGNAL( clicked() ), this, SLOT( smoothVolume() ) );


  mAlphaSlider = new QSlider;
  mAlphaSlider->setMaximum(100);
  mAlphaSlider->setValue( 50 );
  mAlphaSlider->setOrientation( Qt::Horizontal );
  mAlphaSlider->setTickPosition(QSlider::TicksBothSides);
  l->addWidget( mAlphaSlider );

  QHBoxLayout*   layout2 = new QHBoxLayout; 

  QLabel*    textLabel2 = new QLabel;
  textLabel2->setText( tr( "Pen Size" ) );
  layout2->addWidget( textLabel2 );

  mRadiusSpinBox = new QSpinBox;
  mRadiusSpinBox ->setMaximum(500);
  mRadiusSpinBox ->setMinimum(0);
  mRadiusSpinBox ->setValue( 3);
  layout2->addWidget( mRadiusSpinBox );
  l->addLayout( layout2 );

  createComboLabels();

  mViewBin = new QPushButton;
  mViewBin->setText( tr( "Region Grow" ) );
  mViewBin->setCheckable(true);
  l->addWidget(mViewBin);

  mMaxFlow = new QPushButton;
  mMaxFlow->setText( tr( "Run MaxFlow" ) );
  l->addWidget(mMaxFlow);
  connect( mMaxFlow, SIGNAL( clicked() ), this, SLOT( calcMaxFlow() ) );

  QLabel* textLabel3 = new QLabel;
  textLabel3->setText( tr( "TV lvl1: " ) );
  layout2->addWidget( textLabel3 );

  mTValphaSpinBox11 = new QDoubleSpinBox;
  mTValphaSpinBox11 ->setMaximum(1.5);
  mTValphaSpinBox11 ->setMinimum(0);
  mTValphaSpinBox11 ->setSingleStep(0.01);
  mTValphaSpinBox11 ->setValue( 0.35 );
  layout2->addWidget( mTValphaSpinBox11 );
  l->addLayout( layout2 );

  mTValphaSpinBox12 = new QDoubleSpinBox;
  mTValphaSpinBox12 ->setMaximum(5.0);
  mTValphaSpinBox12 ->setMinimum(0);
  mTValphaSpinBox12 ->setSingleStep(0.1);
  mTValphaSpinBox12 ->setValue( 0.5 );
  layout2->addWidget( mTValphaSpinBox12 );
  l->addLayout( layout2 );

  QLabel* textLabel4 = new QLabel;
  textLabel4->setText( tr( "TV lvl2: ") );
  layout2->addWidget( textLabel4 );

  mTValphaSpinBox21 = new QDoubleSpinBox;
  mTValphaSpinBox21 ->setMaximum(1.5);
  mTValphaSpinBox21 ->setMinimum(0);
  mTValphaSpinBox21 ->setSingleStep(0.005);
  mTValphaSpinBox21 ->setValue( 0.15 );
  layout2->addWidget( mTValphaSpinBox21 );
  l->addLayout( layout2 );

  mTValphaSpinBox22 = new QDoubleSpinBox;
  mTValphaSpinBox22 ->setMaximum(5.0);
  mTValphaSpinBox22 ->setMinimum(0);
  mTValphaSpinBox22 ->setSingleStep(0.1);
  mTValphaSpinBox22 ->setValue( 0.5 );
  layout2->addWidget( mTValphaSpinBox22 );
  l->addLayout( layout2 );





  QWidget* dummy=new QWidget;

  l->addWidget(dummy);
  l->setStretchFactor(dummy,1);

  mClearBinAndCut= new QPushButton;
  mClearBinAndCut->setText( tr( "Clear Bin & Cut" ) );
  l->addWidget(mClearBinAndCut);
  connect( mClearBinAndCut, SIGNAL( clicked() ), this, SLOT( clearBinAndCut() ) );

  mISO= new QPushButton;
  mISO->setText( tr( "Update Surface" ) );
  l->addWidget(mISO);
  connect( mISO, SIGNAL( clicked() ), this, SLOT( isoView() ) );

  QPushButton* b= new QPushButton;
  b->setText( tr( "Mask Volume" ) );
  l->addWidget(b);
  connect( b, SIGNAL( clicked() ), this, SLOT( maskVolume() ) );


}
//////////////////////////////////////////////////////////////////////////
void MainApplication::rangeSlot(int l, int u)
{
  mColorWidget->GetLookupTable()->SetRange((double)l, (double)u);
  Refresh();
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::colorSlot(int n)
{

  Refresh();
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::Refresh()
{
  if(mVolume==NULL)return;	
  mViewer[0]->Render();
  mViewer[1]->Render();
  mViewer[2]->Render();
  mOView->Render();
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::connectControls()
{
  int ext[6];
  mVolume->GetExtent(ext);

  mSliceNumber[2]=(ext[1]-ext[0])/2;
  mSliceNumber[1]=(ext[3]-ext[2])/2;
  mSliceNumber[0]=(ext[5]-ext[4])/2;
  connect(mViewer[2], SIGNAL( sliceChanged(int) ), mOView, SLOT( XChanged(int) ) );
  connect(mViewer[1], SIGNAL( sliceChanged(int) ), mOView, SLOT( YChanged(int) ) );
  connect(mViewer[0], SIGNAL( sliceChanged(int) ), mOView, SLOT( ZChanged(int) ) );


  connect( mRadiusSpinBox, SIGNAL( valueChanged(int) ), mViewer[0], SLOT( radiusSlot(int) ) );
  connect( mRadiusSpinBox, SIGNAL( valueChanged(int) ), mViewer[1], SLOT( radiusSlot(int) ) );
  connect( mRadiusSpinBox, SIGNAL( valueChanged(int) ), mViewer[2], SLOT( radiusSlot(int) ) );
  connect( mAlphaSlider, SIGNAL( valueChanged(int) ), mViewer[0], SLOT( alphaSlot(int) ) );
  connect( mAlphaSlider, SIGNAL( valueChanged(int) ), mViewer[1], SLOT( alphaSlot(int) ) );
  connect( mAlphaSlider, SIGNAL( valueChanged(int) ), mViewer[2], SLOT( alphaSlot(int) ) );

  connect( mLabelCombo, SIGNAL( activated(int) ), mViewer[0], SLOT( labelSlot(int) ) );
  connect( mLabelCombo, SIGNAL( activated(int) ), mViewer[1], SLOT( labelSlot(int) ) );
  connect( mLabelCombo, SIGNAL( activated(int) ), mViewer[2], SLOT( labelSlot(int) ) );

  connect( mViewBin, SIGNAL(toggled(bool)), mViewer[0], SLOT(useRegionGrow(bool)) );
  connect( mViewBin, SIGNAL(toggled(bool)), mViewer[1], SLOT(useRegionGrow(bool)) );
  connect( mViewBin, SIGNAL(toggled(bool)), mViewer[2], SLOT(useRegionGrow(bool)) );

  connect(mViewer[0], SIGNAL(viewClicked(int)),  this, SLOT(stackIndex(int)));
  connect(mViewer[1], SIGNAL(viewClicked(int)),  this, SLOT(stackIndex(int)));
  connect(mViewer[2], SIGNAL(viewClicked(int)),  this, SLOT(stackIndex(int)));
  connect(mOView, SIGNAL(viewClicked(int)),  this, SLOT(stackIndex(int)));

}

//////////////////////////////////////////////////////////////////////////
void MainApplication::loadVolume()
{


  QString s = QFileDialog::getOpenFileName(this, QString("Volume"),  QString(),  QString("MetaImage (*.mhd *.mha)"));

  
  if ( s.isEmpty() ) return; 

  mFileName=std::string(s.toStdString().c_str());

  IAMBusy busy;

  vtkMetaImageReader* reader=vtkMetaImageReader::New();
  reader->SetFileName(mFileName.c_str());
  reader->Update();
  int xyz[3];

  xyz[0]=2;
  xyz[1]=2;
  xyz[2]=2;
  /*
    vtkImageShrink3D* mask = vtkImageShrink3D::New();
    mask->SetInput(reader->GetOutput());
    mask->SetShrinkFactors(xyz);
    mask->AveragingOn();
    mask->Update();
  */

  if(mVolume!=NULL)mVolume->Delete();
  mVolume=reader->GetOutput();
  //mVolume=mask->GetOutput();
  //volumePermute();
  this->createBin();
  display();


}

//////////////////////////////////////////////////////////////////////////
void MainApplication::display()
{
  vtkLookupTable* ct=mColorWidget->GetLookupTable();
  double range[2];
  mVolume->GetScalarRange(range);
  ct->SetRange(range);

  mRangeSlider->setMinValue(range[0]);
  mRangeSlider->setMaxValue(range[1]);
  mRangeSlider->setLower(range[0]);
  mRangeSlider->setUpper(range[1]);


  if(mViewer[0]!=NULL)
    {
      mViewer[0]->hide();
      delete mViewer[0];
    }

  if(mViewer[1]!=NULL)
    {
      mViewer[1]->hide();
      delete mViewer[1];
    }


  if(mViewer[2]!=NULL)
    {
      mViewer[2]->hide();
      delete mViewer[2];
    }


  if(mOView!=NULL)
    {
      mOView->hide();
      delete mOView;
    }

  mViewer[0]=new SliceViewer(mLeftSplitter);
  mViewer[0]->SetVolume(mVolume);
  mViewer[0]->SetLookupTable(ct);
  mViewer[0]->SetBin(mBin);
  mViewer[0]->SetCut(mCut);
  mViewer[0]->SetBinaryColorTable(mLabels);
  mViewer[0]->show();
  mViewer[0]->Create();


  mViewer[1]=new SliceViewer(mLeftSplitter);
  mViewer[1]->SetVolume(mVolume);
  mViewer[1]->SetLookupTable(ct);
  mViewer[1]->SetBin(mBin);
  mViewer[1]->SetCut(mCut);
  mViewer[1]->SetBinaryColorTable(mLabels);
  mViewer[1]->SetOrientation(1);
  mViewer[1]->show();
  mViewer[1]->Create();

  mViewer[2]=new SliceViewer(mRightSplitter);
  mViewer[2]->SetVolume(mVolume);
  mViewer[2]->SetLookupTable(ct);
  mViewer[2]->SetBin(mBin);
  mViewer[2]->SetCut(mCut);
  mViewer[2]->SetBinaryColorTable(mLabels);
  mViewer[2]->SetOrientation(0);
  mViewer[2]->show();
  mViewer[2]->Create();

  mOView=new OrthogonalView(mRightSplitter);
  //mOView->SetInput(mBin);
  mOView->SetInput(mVolume);
  //mOView->SetColorTable(mLabels2);
  mOView->SetColorTable(ct);
  mOView->Create();
  mOView->Render();
  mOView->show();


  connectControls();
}

//////////////////////////////////////////////////////////////////////////
void MainApplication::createComboLabels()
{


  mLabelCombo=new QComboBox;

  mLabelCombo->setIconSize(QSize(30,15));

  int s=mColors.size();

  int i;
  MColor ll;
  int r,g,b;
  for( i=0; i<s; i++)
    {
      ll=mColors[i];
      r=255*ll.r;
      g=255*ll.g;
      b=255*ll.b;


      QPixmap cpix(30, 15);
      cpix.fill(QColor(r,g,b));

      QIcon ic(cpix);
      mLabelCombo->insertItem(i,ic, mTextLabels[i] );


    }
  //mLabelCombo->resize(100,20);
  //mLeftDock->addWidget(mLabelCombo);
  QWidget* w=mLeftDock->widget();
  w->layout()->addWidget(mLabelCombo);

}
//////////////////////////////////////////////////////////////////////////
void MainApplication::loadBin()
{

  
  QString s = QFileDialog::getOpenFileName(this, QString("Volume"),  QString(),  QString("MetaImage (*.mhd *.mha)"));


  if ( s.isEmpty() ) return; 

  vtkMetaImageReader* reader=vtkMetaImageReader::New();
  reader->SetFileName(s. toLatin1());
  reader->Update();
  mBin=reader->GetOutput();
  display();
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::saveBin()
{
  if(mVolume==NULL)return;

 
  QString fileName = QFileDialog::getSaveFileName(this, QString("Save File"), QString(),QString( "MetaImage (*.mhd)" ));
  if ( fileName.isEmpty() ) return; 

  QString ext(".mhd");
  QString f(fileName);
  f.append(ext);


  vtkMetaImageWriter* r=vtkMetaImageWriter::New();
  r->SetFileName(f.toLatin1());
  r->SetInput(mBin);
  r->Write();

  r->Delete();
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::saveCut()
{
  if(mVolume==NULL)return;


  QString fileName = QFileDialog::getSaveFileName(this, QString("Save File"), QString(),QString( "MetaImage (*.mhd)" ));
  if ( fileName.isEmpty() ) return;

 // QString ext(".mhd");
  QString f(fileName);
  //f.append(ext);


  vtkMetaImageWriter* r=vtkMetaImageWriter::New();
  r->SetFileName(f.toLatin1());
  r->SetInput(mCut);
  r->Write();

  r->Delete();
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::createBin()
{
  double spacing[3];
  int ext[6];
  int dim[3];
  double orig[3];
  mVolume->GetExtent(ext);
  mVolume->GetSpacing(spacing);
  mVolume->GetDimensions(dim);
  mVolume->GetOrigin(orig);

  mBin=vtkImageData::New();
  mBin->SetSpacing(spacing);
  mBin->SetExtent(ext);
  mBin->SetDimensions(dim);
  mBin->SetOrigin(orig);
  mBin->SetScalarTypeToUnsignedChar();
  mBin->SetNumberOfScalarComponents(1);
  mBin->AllocateScalars();
  mBin->Update();

  mCut=vtkImageData::New();
  mCut->SetSpacing(spacing);
  mCut->SetExtent(ext);
  mCut->SetDimensions(dim);
  mCut->SetOrigin(orig);
  mCut->SetScalarTypeToUnsignedChar();
  mCut->SetNumberOfScalarComponents(1);
  mCut->AllocateScalars();
  mCut->Update();

 // for(int iz=0; iz<dim[2]; iz++){
 //   for(int iy=0; iy<dim[1]; iy++){
 //     for(int ix=0; ix<dim[0]; ix++){
	//
	//mBin->SetScalarComponentFromFloat(ix,iy,iz,0,0.0f);
	//mCut->SetScalarComponentFromFloat(ix,iy,iz,0,0.0f);
 //     }
 //   }
 // }

  vtkstd::fill_n((unsigned char*)mBin->GetScalarPointer(),dim[0]*dim[1]*dim[2], static_cast<unsigned char>(0));
  vtkstd::fill_n((unsigned char*)mCut->GetScalarPointer(),dim[0]*dim[1]*dim[2], static_cast<unsigned char>(0));
  
  
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::createLabels()
{
  mTextLabels.push_back(QString(" Delete"));
  mTextLabels.push_back(QString(" Thorax"));
  
  mTextLabels.push_back(QString(" Myocardium"));
  mTextLabels.push_back(QString(" Blood "));
  mTextLabels.push_back(QString(" Scar"));

  mColors.push_back(MColor(0.0,0.0,0.0)); 
  mColors.push_back(MColor(0.5,0.5,0.5)); 
  mColors.push_back(MColor(0.0, 1.0, 1.0)); // cyan
  mColors.push_back(MColor(1.0, 0.0, 1.0)); // magenta
  mColors.push_back(MColor(1.0, 1.0, 0.0)); // yellow


  int s=mColors.size();
  mLabels=vtkLookupTable::New();
  mLabels->SetNumberOfColors(s);
  mLabels->SetRange(0.0, s-1);
  mLabels->Build();
  MColor l;

  mValues.clear();
  for(int i=0; i<s; i++)
    {
      l=mColors[i];
      mLabels->SetTableValue(i,l.r,l.g,l.b,1.0);
      if(i!=0)mValues.push_back((double)i);
    }

  mLabels->SetTableValue(0,0.0,0.0,0.0,0.0);
  mLabels2=vtkLookupTable::New();	

  mLabels2->DeepCopy(mLabels);
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::smoothVolume()
{

  if(mVolume==NULL)return;
  IAMBusy busy;
  vtkImageGaussianSmooth* smooth=vtkImageGaussianSmooth::New();
  smooth->SetInput(mVolume);
  smooth->SetDimensionality(3);
  smooth->SetRadiusFactors(2,2,2);
  smooth->SetStandardDeviation(1.0,1.0,1.0);
  smooth->Update();  

  mVolume->DeepCopy(smooth->GetOutput());
  mVolume->UpdateInformation();
  smooth->Delete();
  updateRange();
  Refresh();
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::calcMaxFlow()
{
  IAMBusy busy; 

  float *penalty1 =  new float[3];
  float *penalty2 =  new float[3];
  penalty1[0] =(float)mTValphaSpinBox11->value();
  penalty1[1] =(float)mTValphaSpinBox12->value();
  penalty1[2] = 10.0f;

  penalty2[0] =(float)mTValphaSpinBox21->value();
  penalty2[1] =(float)mTValphaSpinBox22->value();
  penalty2[2] = 10.0f;

  //runMaxFlow(mVolume, mBin, mCut);
  
  vtkMetaImageWriter* r=vtkMetaImageWriter::New();
  r->SetFileName("LastBin.mhd");
  r->SetInput(mBin);
  r->SetCompression(false);
  r->Write();

  runMaxFlow(mVolume, mBin, mCut, penalty1, penalty2);
  
  r->Delete();
  Refresh();
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::updateRange()
{
  vtkLookupTable* ct=mColorWidget->GetLookupTable();
  double range[2];
  mVolume->GetScalarRange(range);
  ct->SetRange(range);

  mRangeSlider->setMinValue(range[0]);
  mRangeSlider->setMaxValue(range[1]);
  mRangeSlider->setLower(range[0]);
  mRangeSlider->setUpper(range[1]);


}
//////////////////////////////////////////////////////////////////////////
double    GradientThreshold(vtkImageData* vol, vtkImageData* gm)
{
  double out;
  int dim[3];
  vol->GetDimensions(dim);

  double gsum=0.0;
  double gisum=0.0;
  double val,gval;

  for( int iz=0; iz<dim[2]; iz++)
    for( int iy=0; iy<dim[1]; iy++)
      for( int ix=0; ix<dim[0]; ix++)
	{
	  val=vol->GetScalarComponentAsDouble(ix,iy,iz,0);
	  gval=gm->GetScalarComponentAsDouble(ix,iy,iz,0);
	  gisum+=gval*val;
	  gsum+=gval;
	}

  out=gisum/gsum;

  return out;
}
//////////////////////////////////////////////////////////////////////////

vtkImageData*  MainApplication::mashSmooth(vtkImageData* vol,int itr, double lam )
{
  vtkImageData* out=vtkImageData::New();
  double spacing[3];
  int ext[6];
  int dim[3];
  double orig[3];

  vol->GetExtent(ext);
  vol->GetSpacing(spacing);
  vol->GetDimensions(dim);
  vol->GetOrigin(orig);


  out->SetSpacing(spacing);
  out->SetExtent(ext);
  out->SetDimensions(dim);
  out->SetOrigin(orig);
  out->SetScalarType(vol->GetScalarType());
  out->SetNumberOfScalarComponents(1);
  out->AllocateScalars();
  out->Update();


  double val;
  double diff;


  int nbr8map[8][2] = {-1,0, 1,0,  0,-1,  0,1 , -1,-1, 1,1,  -1,1, 1,-1 };
  int x,y;
  int iz,iy,ix, n;
  int j;
  double q;
  for(n=0; n<itr; n++)
    {

      for(iz=0; iz<dim[2]; iz++)
	for(iy=0; iy<dim[1]; iy++)
	  for(ix=0; ix<dim[0]; ix++)
	    {

	      val=vol->GetScalarComponentAsDouble(ix,iy,iz,0);
	      diff=0.0;
	      for (j=0; j<8; j++)
		{

		  x= ix+nbr8map[j][0];
		  y= iy+nbr8map[j][1];
		
		  if((x>=ext[0]) && (y>=ext[2]) && (x<=ext[1]) && (y<=ext[3]) )
		    {
		      q=vol->GetScalarComponentAsDouble(x,y,iz,0);
		      diff+=q-val;
		    }

		}
	 

	      val+=lam*diff;
	      out->SetScalarComponentFromDouble(ix,iy,iz,0,val);
	    }
      vol->DeepCopy(out);
      std::cout<<"itr="<<n<<std::endl;
    }
  return out;


}
//////////////////////////////////////////////////////////////////////////
void MainApplication::isoView()
{

  if(mVolume==NULL)return;



  mSPP=new vtkSurfacePipeline();
  mSPP->SetVolume(mCut);
  //mSPP->SetValues(mValues);
  mSPP->SetValue(1);
  mSPP->Create();
  //mSPP->UpdateSurface();

  double rgb[3];
  int n=mValues.size();
  double r[2];

  for(int i=4; i<=4; i++)
  {
  mSPP=new vtkSurfacePipeline();
  mSPP->SetVolume(mCut);
  //mSPP->SetValues(mValues);
  mSPP->SetValue(4);
  mSPP->Create();
	  r[0]=(double)i;
	  r[1]=(double)i;
	mSPP->SetRange(r);
	mSPP->UpdateSurface();


    mLabels->GetColor(mValues[i-1],rgb);
    vtkSurface* sur=new vtkSurface();
    sur->SetPolyData(mSPP->GetSurface());
    sur->CreateActor();
	sur->SetOpacity((double)1.0);
    sur->SetDiffuseColor(rgb);
    sur->UpdateColors();
    mOView->AddSurface(sur);
    //std::cout<<"iii="<<i<<std::endl;
  }

  mOView->Render();
}

//////////////////////////////////////////////////////////////////////////
void MainApplication::cleanStack()
{
  int m=mStackedWidget->count();
  for(int i=0;i<m; i++)mStackedWidget->removeWidget(mStackedWidget->widget(i));

}
//////////////////////////////////////////////////////////////////////////
void MainApplication::viewAll()
{
  mLeftSplitter->addWidget(mViewer[0]);
  mLeftSplitter->addWidget(mViewer[1]);
  mRightSplitter->addWidget(mViewer[2]);
  mRightSplitter->addWidget(mOView);

  mStackedWidget->addWidget(mSplitter);
  mViewer[0]->show();
  mViewer[1]->show();
  mViewer[2]->show();	
  mOView->show();
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::stackIndex(int n)
{//Z=2, Y=1, X=0
  QWidget* w=mStackedWidget->widget(mStackedWidget->currentIndex());

  if(n==0)
    {
      cleanStack();
      if(w==mViewer[2])viewAll();
      else
	mStackedWidget->addWidget(mViewer[2]);
      return;
    }


  if(n==1)
    {
      cleanStack();
      if(w==mViewer[1])viewAll();
      else
	mStackedWidget->addWidget(mViewer[1]);
      return;
    }

  if(n==2)
    {
      cleanStack();
      if(w==mViewer[0])viewAll();
      else
	mStackedWidget->addWidget(mViewer[0]);
      return;
    }


  if(n==3)
    {
      cleanStack();
      if(w==mOView)viewAll();
      else
	mStackedWidget->addWidget(mOView);
      return;
    }
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::applyMask()
{

  int dim[3];
  mVolume->GetDimensions(dim);
  int nc=mVolume->GetNumberOfScalarComponents();
  int i;
  double val;

  for( int iz=0; iz<dim[2]; iz++)
    for( int iy=0; iy<dim[1]; iy++)
      for( int ix=0; ix<dim[0]; ix++)
	{
	  val=mBin->GetScalarComponentAsDouble (ix, iy, iz, 0);
	  if(val!=0)
	    {
	      for(i=0; i<nc; i++)mVolume->SetScalarComponentFromDouble(ix,iy,iz,i,0.0);
	      mBin->SetScalarComponentFromDouble(ix,iy,iz,0,0.0);
	    }

	}

}
//////////////////////////////////////////////////////////////////////////
void MainApplication::saveVolume()
{
  if(mVolume==NULL)return;

 
  QString fileName = QFileDialog::getSaveFileName(this, QString("Save File"), QString(),QString( "MetaImage (*.mhd)" ));
  if ( fileName.isEmpty() ) return; 

  //QString ext(".mhd");
  //QString f(fileName);
  //f.append(ext);

  IAMBusy busy;
  vtkMetaImageWriter* r=vtkMetaImageWriter::New();
  r->SetFileName(fileName.toLatin1());
  r->SetInput(mVolume);
  r->SetCompression(false);
  r->Write();

  r->Delete();
}
//////////////////////////////////////////////////////////////////////////
void MainApplication::maskVolume()
{
  if(mVolume==NULL)return;

  IAMBusy busy;

  applyMask();

  Refresh();
}


void MainApplication::clearBinAndCut()
{
  if(mVolume==NULL)return;

  IAMBusy busy;

  
	
  vtkstd::fill_n((unsigned char*)mBin->GetScalarPointer(0,0,0),mBin->GetDimensions()[0]*mBin->GetDimensions()[1]*mBin->GetDimensions()[2], static_cast<unsigned char>(0) );
  vtkstd::fill_n((unsigned char*)mCut->GetScalarPointer(0,0,0),mBin->GetDimensions()[0]*mBin->GetDimensions()[1]*mBin->GetDimensions()[2], static_cast<unsigned char>(0) );
  
  Refresh();
}
