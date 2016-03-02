/***************************************************************************/
/* Name:      runMaxFlow                  

   Authors:
   Martin Rajchl   mrajchl@imaging.robarts.ca

   Description:
   Parallelized Continuous MaxFlow GraphCuts using CUDA

   For more details, see the report:
   Jing Yuan et al.
   "A Study on Continuous Max-Flow and Min-Cut Approaches"
   CVPR, 2010

   Jing Yuan, et al.
   "A Study on Continuous Max-Flow and Min-Cut Approaches: Part I: Binary Labeling"
   UCLA CAM, 2010

   Date: 2011/09/29

*/
/***************************************************************************/

#include <iostream>
#include <string>
#include <sstream>
#include <math.h>

#include "cmf3DCut_std.h"

// ITK
#include <itkImage.h>


#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkCastImageFilter.h>
#include <itkStatisticsImageFilter.h>

#include "itkBinaryThresholdImageFilter.h"


// VTK
#include "vtkMetaImageReader.h"
#include "vtkMetaImageWriter.h"
#include "vtkSmartPointer.h"
#include "vtkImageCast.h"

#include "itkImageToVTKImageFilter.h"
#include "itkVTKImageToImageFilter.h"



// Global vars:
static float alpha = 0.01f; //0-0.2
static float cc = 0.3; //0.2-3.0
static float steps = 0.09; //0.08-0.15

// Typedefs
typedef float FloatPixelType;
typedef unsigned char UCharPixelType;
typedef itk::Image <UCharPixelType, 3> UCharImageType;
typedef itk::Image <FloatPixelType, 3> FloatImageType;
typedef itk::ImageFileReader <FloatImageType> ReaderType;

// Prototypes
void calculateHistogram(float *bin, float *sample, int szSample, int histBinSize);

// Templates
template<typename TImage>
void DeepCopy(typename TImage::Pointer input, typename TImage::Pointer output)
{
  output->SetRegions(input->GetLargestPossibleRegion());
  output->Allocate();
 
  itk::ImageRegionConstIterator<TImage> inputIterator(input, input->GetLargestPossibleRegion());
  itk::ImageRegionIterator<TImage> outputIterator(output, output->GetLargestPossibleRegion());
 
  while(!inputIterator.IsAtEnd())
    {
      outputIterator.Set(inputIterator.Get());
      ++inputIterator;
      ++outputIterator;
    }
}

template<typename TImage>
void normalizeImage(typename TImage::Pointer input)
{
  
  typedef itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;
  StatisticsFilterType::Pointer stats = StatisticsFilterType::New();
  stats->SetInput( input );
  stats->Update();

  itk::ImageRegionIterator<TImage> inputIterator(input, input->GetLargestPossibleRegion());
 
  while(!inputIterator.IsAtEnd())
    {
      inputIterator.Set( (inputIterator.Get()-stats->GetMinimum()) / (stats->GetMaximum()-stats->GetMinimum()) );
      ++inputIterator;
    }
}


template<typename TImage>
void calcDataTerm2(typename TImage::Pointer input,
		   typename TImage::Pointer label,
		   typename TImage::Pointer fCs,
		   typename TImage::Pointer fCt){
 
  int  histBinSize = 16;

  typedef itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;
  StatisticsFilterType::Pointer stats = StatisticsFilterType::New();
  stats->SetInput( input );
  stats->Update();

  fCs->SetRegions(input->GetLargestPossibleRegion());
  fCs->Allocate();

  fCt->SetRegions(input->GetLargestPossibleRegion());
  fCt->Allocate();
 
  itk::ImageRegionIterator<TImage> inputIterator(input, input->GetLargestPossibleRegion());
  itk::ImageRegionIterator<TImage> labelIterator(label, label->GetLargestPossibleRegion());
  itk::ImageRegionIterator<TImage> fCsIterator(fCs, fCs->GetLargestPossibleRegion());
  itk::ImageRegionIterator<TImage> fCtIterator(fCt, fCt->GetLargestPossibleRegion());
  

  int countN1 = 0;
  int countN2 = 0;

  while(!inputIterator.IsAtEnd())
    {
      // Normalize to 8 bit grayscale
      inputIterator.Set( (inputIterator.Get()-stats->GetMinimum()) / (stats->GetMaximum()-stats->GetMinimum()) * 255.0f);

      // Count voxels from FG and BG labels
      if ( labelIterator.Get() == 1.0 ){
	countN1++;
      }
      else if ( labelIterator.Get() == 2.0 ){
	countN2++; 
      }
      ++inputIterator;
      ++labelIterator;
    }


  std::cout << countN1 << " " << countN2 << std::endl;

  int szSampleN1 = countN1;
  int szSampleN2 = countN2;

  if ( szSampleN1 == 0 || szSampleN2 == 0)
    return;

  float *N1 = new float[szSampleN1];
  float *N2 = new float[szSampleN2];

  std::fill_n(N1, szSampleN1 , 0.0f);
  std::fill_n(N2, szSampleN2 , 0.0f);
  
  countN1 = 0;
  countN2 = 0;
  
  labelIterator.GoToBegin();
  inputIterator.GoToBegin();

  while(!inputIterator.IsAtEnd())
    {
      if ( labelIterator.Get() == 1.0 ){
	N1[countN1] = inputIterator.Get();
	countN1++;
      }
      else if ( labelIterator.Get() == 2.0 ){
	N2[countN2] = inputIterator.Get();
	countN2++;
      }
      ++inputIterator;
      ++labelIterator;
    }


  float *histN1 = new float[histBinSize];
  float *histN2 = new float[histBinSize];

  

  
  calculateHistogram(histN1, N1, szSampleN1, histBinSize);
  calculateHistogram(histN2, N2, szSampleN2, histBinSize);


  labelIterator.GoToBegin();
  inputIterator.GoToBegin();

  while(!inputIterator.IsAtEnd())
    {

      if (labelIterator.Get() == 1){
	fCsIterator.Set( -log(1e-8) ); 
	fCtIterator.Set( -log(1 ) );
      }
      else if (labelIterator.Get() == 2){
	fCsIterator.Set( -log(1) ); 
	fCtIterator.Set( -log(1e-8) );
      }
      else{
	fCsIterator.Set( -log(histN2[(int)(inputIterator.Get()/(256/histBinSize))]) ); 
	fCtIterator.Set( -log(histN1[(int)(inputIterator.Get()/(256/histBinSize))]) );
      }
      ++inputIterator;
      ++labelIterator;
      ++fCsIterator;
      ++fCtIterator;
    }
 

  StatisticsFilterType::Pointer statsfCs = StatisticsFilterType::New();
  statsfCs->SetInput( fCs );
  statsfCs->Update();
  std::cout << "fCs: min=" << statsfCs->GetMinimum() << " max=" << statsfCs->GetMaximum() << " mean=" << statsfCs->GetMean() << std::endl;

  StatisticsFilterType::Pointer statsfCt = StatisticsFilterType::New();
  statsfCt->SetInput( fCt );
  statsfCt->Update();
  std::cout << "fCt: min=" <<  statsfCt->GetMinimum() << " max=" << statsfCt->GetMaximum() << " mean=" << statsfCt->GetMean() << std::endl;

  float maxDataCost, minDataCost;

  if ( statsfCs->GetMinimum() <= statsfCt->GetMinimum() )
    minDataCost = statsfCs->GetMinimum();
  else 
    minDataCost = statsfCt->GetMinimum();

  if ( statsfCs->GetMaximum() >= statsfCt->GetMaximum() )
    maxDataCost = statsfCs->GetMaximum();
  else 
    maxDataCost = statsfCt->GetMaximum();

  

  fCsIterator.GoToBegin();
  fCtIterator.GoToBegin();

   while(!fCsIterator.IsAtEnd())
    {
      fCsIterator.Set( fCsIterator.Get() / (maxDataCost - minDataCost) ); 
      fCtIterator.Set( fCtIterator.Get() / (maxDataCost - minDataCost) );


      ++fCsIterator;
      ++fCtIterator;
    }


  StatisticsFilterType::Pointer statsfCs2 = StatisticsFilterType::New();
  statsfCs2->SetInput( fCs );
  statsfCs2->Update();
  std::cout << "fCs: min=" << statsfCs2->GetMinimum() << " max=" << statsfCs2->GetMaximum() << " mean=" << statsfCs2->GetMean() << std::endl;

  StatisticsFilterType::Pointer statsfCt2 = StatisticsFilterType::New();
  statsfCt2->SetInput( fCt );
  statsfCt2->Update();
  std::cout << "fCt: min=" <<  statsfCt2->GetMinimum() << " max=" << statsfCt2->GetMaximum() << " mean=" << statsfCt2->GetMean() << std::endl;


}



template<typename TImage>
void calcDataTerm(typename TImage::Pointer input,
		  typename TImage::Pointer label,
		  typename TImage::Pointer fCs,
		  typename TImage::Pointer fCt){

  fCs->SetRegions(input->GetLargestPossibleRegion());
  fCs->Allocate();

  fCt->SetRegions(input->GetLargestPossibleRegion());
  fCt->Allocate();
 
  itk::ImageRegionConstIterator<TImage> inputIterator(input, input->GetLargestPossibleRegion());
  itk::ImageRegionIterator<TImage> labelIterator(label, label->GetLargestPossibleRegion());
  itk::ImageRegionIterator<TImage> fCsIterator(fCs, fCs->GetLargestPossibleRegion());
  itk::ImageRegionIterator<TImage> fCtIterator(fCt, fCt->GetLargestPossibleRegion());
 
  float meanFG = 0;
  float meanBG = 0;
  float countFG = 0;
  float countBG = 0;


  while(!inputIterator.IsAtEnd())
    {
      if ( labelIterator.Get() == 1.0 ){
	meanFG += inputIterator.Get();
	countFG++;
      }
      else if ( labelIterator.Get() == 2.0 ){
	meanBG += inputIterator.Get();
	countBG++;
      }
      ++inputIterator;
      ++labelIterator;
    }


  meanFG = meanFG / countFG;
  meanBG = meanBG / countBG;

  labelIterator.GoToBegin();
  inputIterator.GoToBegin();

  while(!inputIterator.IsAtEnd())
    {
      
      fCsIterator.Set( fabs(inputIterator.Get() - meanFG ) );
      fCtIterator.Set( fabs(inputIterator.Get() - meanBG ) );
      ++inputIterator;
      ++fCsIterator;
      ++fCtIterator;

    }



}





FloatImageType::Pointer arrayToImage(int nx, int ny, int nz,
				     float dx, float dy, float dz,
				     float ox, float oy , float oz,
				     float * buffer)
{
  FloatImageType::Pointer image = FloatImageType::New();
  FloatImageType::SizeType size;
  size[0] = nx;
  size[1] = ny;
  size[2] = nz;

  FloatImageType::IndexType start;
  start[0] = 0;
  start[1] = 0;
  start[2] = 0;
  
  FloatImageType::RegionType region;
  region.SetSize( size );
  region.SetIndex( start );

  image->SetRegions( region );
  image->Allocate();

  double spacing[3];
  spacing[0] = dx;
  spacing[1] = dy;
  spacing[2] = dz;

  image->SetSpacing( spacing );

  double origin[3];
  origin[0] = ox;
  origin[1] = oy;
  origin[2] = oz;

  image->SetOrigin( origin );

  typedef itk::ImageRegionIterator< FloatImageType > IteratorType;
  IteratorType it( image, region );
  it.GoToBegin();

  float * data = buffer;
  while( ! it.IsAtEnd() )
    {
      it.Set( *data );
      ++it;
      ++data;
    }

  return image;
}





void runMaxFlow (vtkImageData *img, vtkImageData *label, vtkImageData *cut){


  // vtkSmartPointer<vtkMetaImageReader> reader1 =   vtkSmartPointer<vtkMetaImageReader>::New();
  // reader1->SetFileName("delayed_contrast_MRIcropped.mhd");

  // vtkSmartPointer<vtkMetaImageReader> reader2 = vtkSmartPointer<vtkMetaImageReader>::New();
  // reader2->SetFileName("delayed_contrast_MRIcropped_testLabel2.mhd");

  vtkSmartPointer<vtkImageCast> cast1 = vtkSmartPointer<vtkImageCast>::New();
  vtkSmartPointer<vtkImageCast> cast2 = vtkSmartPointer<vtkImageCast>::New();
  cast1->SetOutputScalarTypeToFloat();
  cast1->SetInput( img );
  cast1->Update();
  cast2->SetOutputScalarTypeToFloat();
  cast2->SetInput( label );
  cast2->Update();

  typedef itk::VTKImageToImageFilter<FloatImageType> ImportConnectorType;
  ImportConnectorType::Pointer connector1 = ImportConnectorType::New();
  connector1->SetInput( cast1->GetOutput() );
  connector1->Update();
  ImportConnectorType::Pointer connector2 = ImportConnectorType::New();
  connector2->SetInput( cast2->GetOutput() );
  connector2->Update();



  FloatImageType::Pointer inputImage = FloatImageType::New();
  FloatImageType::Pointer inputLabel = FloatImageType::New();
  FloatImageType::Pointer fCs = FloatImageType::New();
  FloatImageType::Pointer fCt = FloatImageType::New();

  inputImage = connector1->GetImporter()->GetOutput();
  inputLabel = connector2->GetImporter()->GetOutput();

  //normalizeImage < FloatImageType > ( inputImage );
  //calcDataTerm < FloatImageType > ( inputImage, inputLabel, fCs, fCt );

  calcDataTerm2 < FloatImageType > ( inputImage, inputLabel, fCs, fCt );


  // Set the parameters
  FloatImageType::SizeType imageSize = inputImage->GetLargestPossibleRegion().GetSize();
  
  /*
   *pfVecParameters Setting
   * [0] : x
   * [1] : y
   * [2] : z
   * [3] : penalty parameter alpha
   * [4] : total iteration number
   * [5] : error criterion
   * [6] : cc for ALM (0.2 <= cc <= 2)
   * [7] : steps for each iteration (0.1 <= Steps <= 0.19)
   * [8] :
   * [9] :
   */
  float pars[8] = {imageSize[0],
		   imageSize[1],
		   imageSize[2],
		   alpha, 
		   200,
		   0.00005,
		   cc,
		   steps};


  float *lambda = new float[ imageSize[0] * imageSize[1] * imageSize[2] ];

  std::fill_n(lambda, imageSize[0] * imageSize[1] * imageSize[2] , 0.0f);

  
  cmf3DCut_std(lambda,
	       fCs->GetBufferPointer(),
	       fCt->GetBufferPointer(),
	       pars);
  
   
   
  FloatImageType::Pointer outputImage = arrayToImage(inputImage->GetLargestPossibleRegion().GetSize()[0],
  						     inputImage->GetLargestPossibleRegion().GetSize()[1],
  						     inputImage->GetLargestPossibleRegion().GetSize()[2],
  						     inputImage->GetSpacing()[0],
  						     inputImage->GetSpacing()[1],
  						     inputImage->GetSpacing()[2],
  						     inputImage->GetOrigin()[0],
  						     inputImage->GetOrigin()[1],
  						     inputImage->GetOrigin()[2],
  						     lambda
  						     );
  

  typedef itk::BinaryThresholdImageFilter <FloatImageType, FloatImageType> BinaryThresholdImageFilterType;
 
  BinaryThresholdImageFilterType::Pointer thresholdFilter = BinaryThresholdImageFilterType::New();
  thresholdFilter->SetInput( outputImage );
  thresholdFilter->SetLowerThreshold(0.5f);
  thresholdFilter->SetUpperThreshold(1.0f);
  thresholdFilter->SetInsideValue(1.0);
  thresholdFilter->SetOutsideValue(0.0);

  typedef itk::ImageToVTKImageFilter< FloatImageType> ExportConnectorType;
  ExportConnectorType::Pointer connector3 = ExportConnectorType::New();
  connector3->SetInput( thresholdFilter->GetOutput() );
  connector3->Update();

  vtkSmartPointer<vtkImageCast> cast3 = vtkSmartPointer<vtkImageCast>::New();
  cast3->SetInput( connector3->GetOutput() );
  cast3->SetOutputScalarTypeToUnsignedChar();
  cast3->Update();
  cut->DeepCopy( cast3->GetOutput() );

  // vtkSmartPointer<vtkMetaImageWriter> writer = vtkSmartPointer<vtkMetaImageWriter>::New();
  // writer->SetInput( cast3->GetOutput() );
  // writer->SetFileName( "output.mhd" );
  // writer->Write();


  delete[] lambda;
 
}


void calculateHistogram(float *bin, float *sample, int szSample, int histBinSize){

  for (int i =0; i < szSample; i++){
    
    float data = sample[i];
    
   
    if (histBinSize == 16){
     
      if (data >=0 && data <16) {bin[0]++; }
      else if (data >=16 && data <32) {bin[1]++; }
      else if (data >=32 && data <48) {bin[2]++; }
      else if (data >=48 && data <64) {bin[3]++; }
      else if (data >=64 && data <80) {bin[4]++; }
      else if (data >=80 && data <96) {bin[5]++; }
      else if (data >=96 && data <112) {bin[6]++; }
      else if (data >=112 && data <128) {bin[7]++; }
      else if (data >=128 && data <144) {bin[8]++; }
      else if (data >=144 && data <160) {bin[9]++; }
      else if (data >=160 && data <176) {bin[10]++; }
      else if (data >=176 && data <192) {bin[11]++; }
      else if (data >=192 && data <208) {bin[12]++; }
      else if (data >=208 && data <224) {bin[13]++; }
      else if (data >=224 && data <240) {bin[14]++; }
      else if (data >=240 && data <256) {bin[15]++; }

    }

    else if (histBinSize == 8){
     
      if (data >=0 && data <32) {bin[0]++; }
      else if (data >=32 && data <64) {bin[1]++; }
      else if (data >=64 && data <96) {bin[2]++; }
      else if (data >=96 && data <128) {bin[3]++; }
      else if (data >=128 && data <160) {bin[4]++; }
      else if (data >=160 && data <192) {bin[5]++; }
      else if (data >=192 && data <224) {bin[6]++; }
      else if (data >=224 && data <256) {bin[7]++; }

    }
   
    else {
      std::cout << "Error: Incorrect bin size. " << std::endl;
      return;
    }
  }



  for (int i = 0; i < histBinSize; i++){
    bin[i] =  bin[i]/szSample + 1.0e-8;
    std::cout << "Bin[" << i << "]: " << bin[i] << std::endl; 
  }
  std::cout << std::endl;

}
