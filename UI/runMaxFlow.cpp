/***************************************************************************/
/* Name:      runMaxFlow.cpp

Authors:
Martin Rajchl   mrajchl@imaging.robarts.ca


Description:
Cost functions for Parallelized Continuous MaxFlow GraphCuts using CUDA

Date: 2012/02/17

*/
/***************************************************************************/

#include <iostream>
#include <string>
#include <sstream>
#include <math.h>

#include "cmf3DHybridPottsCut.h"

// ITK
#include <itkImage.h>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkCastImageFilter.h>
#include <itkStatisticsImageFilter.h>

#include <itkConnectedComponentImageFilter.h>
#include <itkGradientMagnitudeRecursiveGaussianImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>

// VTK
#include "vtkMetaImageReader.h"
#include "vtkMetaImageWriter.h"
#include "vtkSmartPointer.h"
#include "vtkImageCast.h"

#include "itkImageToVTKImageFilter.h"
#include "itkVTKImageToImageFilter.h"



// Global vars:
static float cc = 0.25; //0.2-3.0
static float steps = 0.11; //0.08-0.15

// Typedefs
typedef float FloatPixelType;
typedef unsigned char UCharPixelType;
typedef unsigned char UIntPixelType;
typedef itk::Image <UCharPixelType, 3> UCharImageType;
typedef itk::Image <UIntPixelType, 3> UIntImageType;
typedef itk::Image <FloatPixelType, 3> FloatImageType;
typedef itk::ImageFileReader <FloatImageType> ReaderType;

// Prototypes
void calculateHistogram(float *bin, float *sample, int szSample, int histBinSize);


// Templates
template<typename TImage>void DeepCopy(typename TImage::Pointer input, typename TImage::Pointer output){
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



template<typename TImage>void calcTVRegTerm(typename TImage::Pointer input, typename TImage::Pointer penalty1, typename TImage::Pointer penalty2, float *penaltyParameters1, float *penaltyParameters2){

	FloatImageType::IndexType start;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;

	FloatImageType::RegionType region;
	region.SetSize( input->GetLargestPossibleRegion().GetSize() );
	region.SetIndex( start );

	penalty1->SetRegions( region );
	penalty1->Allocate();
	penalty1->SetSpacing( input->GetSpacing() );
	penalty1->SetOrigin( input->GetOrigin() );

	penalty2->SetRegions( region );
	penalty2->Allocate();
	penalty2->SetSpacing( input->GetSpacing() );
	penalty2->SetOrigin( input->GetOrigin() );

	FloatImageType::Pointer gradient = FloatImageType::New();
	gradient->SetRegions( region );
	gradient->Allocate();
	gradient->SetSpacing( input->GetSpacing() );
	gradient->SetOrigin( input->GetOrigin() );


	typedef itk::GradientMagnitudeRecursiveGaussianImageFilter< FloatImageType, FloatImageType >  GradientMagnitudeRecursiveGaussianImageFilterType;
	GradientMagnitudeRecursiveGaussianImageFilterType::Pointer gradientFilter = GradientMagnitudeRecursiveGaussianImageFilterType::New();
	gradientFilter->SetInput( input );
	gradientFilter->Update();


	gradient = gradientFilter->GetOutput();
	gradient->Update();

	typedef itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;
	StatisticsFilterType::Pointer stats = StatisticsFilterType::New();
	stats->SetInput( gradient );
	stats->Update();

	//std::cout << "Gradient Int(min,max): " << stats->GetMinimum() << " - " << stats->GetMaximum() << std::endl;
	//std::cout << "pPars1: " << penaltyParameters1[0] << ", " << penaltyParameters1[1] << ", " << penaltyParameters1[2] << std::endl;
	//std::cout << "pPars2: " << penaltyParameters2[0] << ", " << penaltyParameters2[1] << ", " << penaltyParameters2[2] << std::endl;
	//std::cout.flush();

	itk::ImageRegionConstIterator<FloatImageType> inputIterator(input, input->GetLargestPossibleRegion());
	itk::ImageRegionIterator<FloatImageType> p1Iterator(penalty1, penalty1->GetLargestPossibleRegion());
	itk::ImageRegionIterator<FloatImageType> p2Iterator(penalty2, penalty2->GetLargestPossibleRegion());
	itk::ImageRegionIterator<FloatImageType> gradientIterator(gradient, gradient->GetLargestPossibleRegion());

	inputIterator.GoToBegin();
	p1Iterator.GoToBegin();
	p2Iterator.GoToBegin();
	gradientIterator.GoToBegin();


	while( ! inputIterator.IsAtEnd() )
	{
		p1Iterator.Set( penaltyParameters1[0] + penaltyParameters1[1] * exp (-penaltyParameters1[2]*gradientIterator.Get()/stats->GetMaximum() ) );
		p2Iterator.Set( penaltyParameters2[0] + penaltyParameters2[1] * exp (-penaltyParameters2[2]*gradientIterator.Get()/stats->GetMaximum() ) );

		++inputIterator;
		++p1Iterator;
		++p2Iterator;
		++gradientIterator;
	}


}



template<typename TImage>void calcMaxLikelihoodMultiLabelDataTerm(typename TImage::Pointer input, typename TImage::Pointer label, float *fCt, int nLab){

	int  histBinSize = 64;

	typedef itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;
	StatisticsFilterType::Pointer stats = StatisticsFilterType::New();
	stats->SetInput( input );
	stats->Update();

	itk::ImageRegionIterator<TImage> inputIterator(input, input->GetLargestPossibleRegion());
	itk::ImageRegionIterator<TImage> labelIterator(label, label->GetLargestPossibleRegion());

	int countN1 = 0;
	int countN2 = 0;
	int countN3 = 0;
	int countN4 = 0;
	

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
		else if ( labelIterator.Get() == 3.0 ){
			countN3++; 
		}
		else if ( labelIterator.Get() == 4.0 ){
			countN4++; 
		}
	
		++inputIterator;
		++labelIterator;
	}



	typedef itk::StatisticsImageFilter< FloatImageType > StatisticsFilterType;
	StatisticsFilterType::Pointer stats2 = StatisticsFilterType::New();
	stats2->SetInput( input );
	stats2->Update();

	std::cout << "Initial Int(min,max): " << stats->GetMinimum() << " - " << stats->GetMaximum() << std::endl;
	std::cout << "Normalized Int(min,max): " << stats2->GetMinimum() << " - " << stats2->GetMaximum() << std::endl;

	std::cout << "Bin Count(N1,N2,N3,N4): " << countN1 << ", " << countN2 << ", " << countN3 << ", " << countN4 <<std::endl;

	int szSampleN1 = countN1;
	int szSampleN2 = countN2;
	int szSampleN3 = countN3;
	int szSampleN4 = countN4;

	float *N1 = new float[szSampleN1]; std::fill_n(N1, szSampleN1 , 0.0f);
	float *N2 = new float[szSampleN2]; std::fill_n(N2, szSampleN2 , 0.0f);
	float *N3 = new float[szSampleN3]; std::fill_n(N3, szSampleN3 , 0.0f);
	float *N4 = new float[szSampleN4]; std::fill_n(N4, szSampleN4 , 0.0f);

	countN1 = 0;
	countN2 = 0;
	countN3 = 0;
	countN4 = 0;

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
		else if ( labelIterator.Get() == 3.0 ){
			N3[countN3] = inputIterator.Get();
			countN3++;
		}
		else if ( labelIterator.Get() == 4.0 ){
			N4[countN4] = inputIterator.Get();
			countN4++;
		}
 
		++inputIterator;
		++labelIterator;
	}

	// Create histograms for each label
	float *histN1 = new float[histBinSize];
	float *histN2 = new float[histBinSize];
	float *histN3 = new float[histBinSize];
	float *histN4 = new float[histBinSize];
	
	// Calculate the normalized histograms for each label
	calculateHistogram(histN1, N1, szSampleN1, histBinSize);
	calculateHistogram(histN2, N2, szSampleN2, histBinSize);
	calculateHistogram(histN3, N3, szSampleN3, histBinSize);
	calculateHistogram(histN4, N4, szSampleN4, histBinSize);
	

	int idx = 0;

	FloatImageType::SizeType dims = input->GetLargestPossibleRegion().GetSize();
	int imageSize = dims[0]*dims[1]*dims[2];

	for(int i = 1; i<=nLab; i++){

		labelIterator.GoToBegin();
		inputIterator.GoToBegin();

		while(!inputIterator.IsAtEnd())
		{


			if(i == 1){ // Heart label 
				fCt[idx] =  0.0f; 
			}
			else if(i == 2){ // Thorax label
				if(labelIterator.Get() == 1 ){
					fCt[idx] = 0.0f;
					fCt[idx+imageSize] = -log(1.0e-8f)/8.0f; // max cost
					fCt[idx+imageSize*2] = -log(1.0e-8f)/8.0f; // max cost
					fCt[idx+imageSize*3] = -log(1.0e-8f)/8.0f; // max cost
				}
				else{
					fCt[idx] = (-log(histN1[(int)(inputIterator.Get()/(256/histBinSize))]))/8.0f ;
				}
			}
			else if(i == 3){
				if(labelIterator.Get() == 2 ){
					fCt[idx] = 0.0f;
					fCt[idx-imageSize] = -log(1.0e-8f)/8.0f; // max cost
					fCt[idx+imageSize] = -log(1.0e-8f)/8.0f; // max cost
					fCt[idx+imageSize*2] = -log(1.0e-8f)/8.0f; // max cost
				}
				else{
					fCt[idx] =  (-log(histN2[(int)(inputIterator.Get()/(256/histBinSize))]))/8.0f ;
				}
			}
			else if(i == 4){
				if(labelIterator.Get() == 3 ){
					fCt[idx] = 0.0f;
					fCt[idx-imageSize] = -log(1.0e-8f)/8.0f; // max cost
					fCt[idx-imageSize*2] = -log(1.0e-8f)/8.0f; // max cost
					fCt[idx+imageSize] = -log(1.0e-8f)/8.0f; // max cost
				}
				else{
					fCt[idx] =  (-log(histN3[(int)(inputIterator.Get()/(256/histBinSize))]))/8.0f ;
				}
			}
			else if(i == 5){
				if(labelIterator.Get() == 4 ){
					fCt[idx] = 0.0f;
					fCt[idx-imageSize] = -log(1.0e-8f)/8.0f; // max cost
					fCt[idx-imageSize*2] = -log(1.0e-8f)/8.0f; // max cost
					fCt[idx-imageSize*3] = -log(1.0e-8f)/8.0f; // max cost
				}
				else{
					fCt[idx] =  (-log(histN4[(int)(inputIterator.Get()/(256/histBinSize))]))/8.0f ;
				}
			}


			++inputIterator;
			++labelIterator;
			++idx;	
		}


	}

	delete[] N1;
	delete[] N2;
	delete[] N3;
	delete[] N4;
	delete[] histN1;
	delete[] histN2;
	delete[] histN3;
	delete[] histN4;

}


FloatImageType::Pointer multiArrayToImageAndThreshold2(int nx, int ny, int nz, float dx, float dy, float dz, float ox, float oy , float oz, float * buffer, float nLab){
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

	while( ! it.IsAtEnd() )
	{
		it.Set( 0.0f );
		++it;
	}




	float * data = buffer;

	for(int i = 0; i < 5; i++){
		it.GoToBegin();
		while( ! it.IsAtEnd() )
		{   

			float currVal = *data ;

			if(currVal > 0.5 ){
				currVal = 1.0f;
			}
			else{
				currVal = 0.0f;
			}


			it.Set( it.Get() + currVal*i );


			++it;
			++data;
		}
	}





	return image;
}


template<typename TImage>void connectedSeedComponentFilter(typename TImage::Pointer cut, typename TImage::Pointer label){ 

	FloatImageType::IndexType start;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;

	FloatImageType::RegionType region;
	region.SetSize( cut->GetLargestPossibleRegion().GetSize() );
	region.SetIndex( start );

	UIntImageType::Pointer tempCut = UIntImageType::New();
	tempCut->SetRegions( region );
	tempCut->Allocate();
	tempCut->SetSpacing( cut->GetSpacing() );
	tempCut->SetOrigin( cut->GetOrigin() );
	tempCut->Update();

	//UIntImageType::Pointer tempLabel = UIntImageType::New();
	FloatImageType::Pointer tempLabel = FloatImageType::New();
	tempLabel->SetRegions( region );
	tempLabel->Allocate();
	tempLabel->SetSpacing( cut->GetSpacing() );
	tempLabel->SetOrigin( cut->GetOrigin() );
	tempLabel->Update();
	
	typedef itk::BinaryThresholdImageFilter<FloatImageType,FloatImageType> BinaryThresholdImageFilterType;
	typedef itk::ConnectedComponentImageFilter<FloatImageType, UIntImageType > ConnectedComponentImageFilterType;

	// calculate cc for label 4 (scar tissue) of the max flow result
	BinaryThresholdImageFilterType::Pointer thresholdFilter1 = BinaryThresholdImageFilterType::New();
	thresholdFilter1->SetInput(cut);
	thresholdFilter1->SetLowerThreshold(4.0f);
	thresholdFilter1->SetUpperThreshold(4.0f);
	thresholdFilter1->Update();
	

	ConnectedComponentImageFilterType::Pointer cCompFilter1 = ConnectedComponentImageFilterType::New ();
	cCompFilter1->SetInput( thresholdFilter1->GetOutput() );
	cCompFilter1->SetNumberOfThreads(1);
	cCompFilter1->Update();
	tempCut = cCompFilter1->GetOutput();
	int numberCutObjects = cCompFilter1->GetObjectCount();

	// calculate cc for label 4 (scar tissue) of the label
	BinaryThresholdImageFilterType::Pointer thresholdFilter2 = BinaryThresholdImageFilterType::New();
	thresholdFilter2->SetInput(label);
	thresholdFilter2->SetLowerThreshold(4.0f);
	thresholdFilter2->SetUpperThreshold(4.0f);
	thresholdFilter2->Update();

	//ConnectedComponentImageFilterType::Pointer cCompFilter2 = ConnectedComponentImageFilterType::New ();
	//cCompFilter2->SetInput( thresholdFilter2->GetOutput() );
	//cCompFilter2->SetFullyConnected(true);
	//cCompFilter2->Update();
	//std::cout << "Number of scar label input cc objects found: " << cCompFilter2->GetObjectCount() << std::endl;
	///// CRASHED HERE!!! 

	/// FIX IT, NOW.
	//tempLabel = cCompFilter2->GetOutput();
	tempLabel = thresholdFilter2->GetOutput();

	itk::ImageRegionIterator<UIntImageType> tempCutIterator(tempCut, tempCut->GetLargestPossibleRegion());
	itk::ImageRegionIterator<FloatImageType> cutIterator(cut, cut->GetLargestPossibleRegion());
	//itk::ImageRegionIterator<UIntImageType> labelIterator(tempLabel, tempLabel->GetLargestPossibleRegion());
	itk::ImageRegionIterator<FloatImageType> labelIterator(tempLabel, tempLabel->GetLargestPossibleRegion());

	std::cout << "Number of scar connected component objects found: " << numberCutObjects << std::endl;

	// keep track of scar components with seeds connected
	int *seededCuts = new int[numberCutObjects];
	std::fill_n(seededCuts, numberCutObjects, 0);

	tempCutIterator.GoToBegin();
	labelIterator.GoToBegin();
	while(!labelIterator.IsAtEnd())
	{
		if(labelIterator.Get() != 0 && tempCutIterator.Get() != 0){
			seededCuts[tempCutIterator.Get()-1] = 1;	
		}
		
		++tempCutIterator;
		++labelIterator;
	}
	

	tempCutIterator.GoToBegin();
	cutIterator.GoToBegin();

	while(!cutIterator.IsAtEnd())
	{
		if(tempCutIterator.Get() != 0){
			if (seededCuts[tempCutIterator.Get()-1] == 1){
				cutIterator.Set(4.0f); // set to scar tissue
			}
			else{
				cutIterator.Set(0.0f); // set to zero
			}
		}
		++tempCutIterator;
		++cutIterator;

	}


	typedef itk::ImageToVTKImageFilter< FloatImageType> ExportConnectorType;
	ExportConnectorType::Pointer connector3 = ExportConnectorType::New();
	connector3->SetInput( cut );
	connector3->Update();



	//vtkSmartPointer<vtkImageCast> cast3 = vtkSmartPointer<vtkImageCast>::New();
	//cast3->SetInput( connector3->GetOutput() );
	//cast3->SetOutputScalarTypeToFloat();
	//cast3->Update();

	//vtkSmartPointer<vtkMetaImageWriter> writer = vtkSmartPointer<vtkMetaImageWriter>::New();
	//writer->SetInput( cast3->GetOutput() );
	//writer->SetFileName( "ccCut.mhd" );
	//writer->Write();

}




void runMaxFlow (vtkImageData *img, vtkImageData *label, vtkImageData *cut, float *penaltyParameters1, float *penaltyParameters2){




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

	inputImage = connector1->GetImporter()->GetOutput();
	inputLabel = connector2->GetImporter()->GetOutput();



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

	int nLab = 5;


	float pars[8] = {imageSize[0],
		imageSize[1],
		imageSize[2],
		(float) nLab,
		200,
		0.0001,
		cc,
		steps};


	//normalizeImage < FloatImageType > ( inputImage );


	float *fCt = new float[ imageSize[0] * imageSize[1] * imageSize[2] * nLab ];
	std::fill_n(fCt, imageSize[0] * imageSize[1] * imageSize[2] * nLab , 0.0f);
	//calcDataTerm2 < FloatImageType > ( inputImage, inputLabel, fCs, fCt );
	calcMaxLikelihoodMultiLabelDataTerm < FloatImageType > ( inputImage, inputLabel, fCt, nLab );

	FloatImageType::Pointer penalty1 = FloatImageType::New();
	FloatImageType::Pointer penalty2 = FloatImageType::New();
	calcTVRegTerm <FloatImageType> (inputImage, penalty1, penalty2, penaltyParameters1, penaltyParameters2);



	float *u = new float[ imageSize[0] * imageSize[1] * imageSize[2] * nLab ];
	std::fill_n(u, imageSize[0] * imageSize[1] * imageSize[2] * nLab , 0.0f);

	//float alpha1 = 1.0f;
	//float alpha2 = 0.5f;
	//float *penalty1 = new float[ imageSize[0] * imageSize[1] * imageSize[2] ];
	//float *penalty2 = new float[ imageSize[0] * imageSize[1] * imageSize[2] ];
	//std::fill_n(penalty1, imageSize[0] * imageSize[1] * imageSize[2] , alpha[0]);
	//std::fill_n(penalty2, imageSize[0] * imageSize[1] * imageSize[2] , alpha[1]);



	cmf3DHybridPottsCut(penalty1->GetBufferPointer(),
		penalty2->GetBufferPointer(),
		fCt,
		pars,
		u);



	FloatImageType::Pointer outputImage = multiArrayToImageAndThreshold2(inputImage->GetLargestPossibleRegion().GetSize()[0],
		inputImage->GetLargestPossibleRegion().GetSize()[1],
		inputImage->GetLargestPossibleRegion().GetSize()[2],
		inputImage->GetSpacing()[0],
		inputImage->GetSpacing()[1],
		inputImage->GetSpacing()[2],
		inputImage->GetOrigin()[0],
		inputImage->GetOrigin()[1],
		inputImage->GetOrigin()[2],
		u,
		nLab
		);



	// Connected component filter with seeds

	connectedSeedComponentFilter < FloatImageType >(outputImage, inputLabel);



	typedef itk::ImageToVTKImageFilter< FloatImageType> ExportConnectorType;
	ExportConnectorType::Pointer connector3 = ExportConnectorType::New();
	connector3->SetInput( outputImage );
	connector3->Update();



	vtkSmartPointer<vtkImageCast> cast3 = vtkSmartPointer<vtkImageCast>::New();
	cast3->SetInput( connector3->GetOutput() );
	//  cast3->SetOutputScalarTypeToUnsignedChar();
	cast3->SetOutputScalarTypeToFloat();
	cast3->Update();
	cut->DeepCopy( cast3->GetOutput() );
	cut->Update();


	vtkSmartPointer<vtkMetaImageWriter> writer = vtkSmartPointer<vtkMetaImageWriter>::New();
	writer->SetInput( cast3->GetOutput() );
	writer->SetFileName( "lastCut.mhd" );
	writer->Write();


	//delete[] penalty1;
	//delete[] penalty2;
	delete[] u;


}



void calculateHistogram(float *bin, float *sample, int szSample, int histBinSize){

	std::fill_n(bin, histBinSize , 0.0f);

	for (int i =0; i < szSample; i++){

		float data = sample[i];
		/* if (histBinSize == 256){

		bin[(int)floor(data)]++;

		}*/
		if (histBinSize == 64){

			if (data >=0 && data <4) {bin[0]++; }
			else if (data >=4 && data <8) {bin[1]++; }
			else if (data >=8 && data <12) {bin[2]++; }
			else if (data >=12 && data <16) {bin[3]++; }
			else if (data >=16 && data <20) {bin[4]++; }
			else if (data >=20 && data <24) {bin[5]++; }
			else if (data >=24 && data <28) {bin[6]++; }
			else if (data >=28 && data <32) {bin[7]++; }
			else if (data >=32 && data <36) {bin[8]++; }
			else if (data >=36 && data <40) {bin[9]++; }
			else if (data >=40 && data <44) {bin[10]++; }
			else if (data >=44 && data <48) {bin[11]++; }
			else if (data >=48 && data <52) {bin[12]++; }
			else if (data >=52 && data <56) {bin[13]++; }
			else if (data >=56 && data <60) {bin[14]++; }
			else if (data >=60 && data <64) {bin[15]++; }
			else if (data >=64 && data <68) {bin[16]++; }
			else if (data >=68 && data <72) {bin[17]++; }
			else if (data >=72 && data <76) {bin[18]++; }
			else if (data >=76 && data <80) {bin[19]++; }
			else if (data >=80 && data <84) {bin[20]++; }
			else if (data >=84 && data <88) {bin[21]++; }
			else if (data >=88 && data <92) {bin[22]++; }
			else if (data >=92 && data <96) {bin[23]++; }
			else if (data >=96 && data <100) {bin[24]++; }
			else if (data >=100 && data <104) {bin[25]++; }
			else if (data >=104 && data <108) {bin[26]++; }
			else if (data >=108 && data <112) {bin[27]++; }
			else if (data >=112 && data <116) {bin[28]++; }
			else if (data >=116 && data <120) {bin[29]++; }
			else if (data >=120 && data <124) {bin[30]++; }
			else if (data >=124 && data <128) {bin[31]++; }
			else if (data >=128 && data <132) {bin[32]++; }
			else if (data >=132 && data <136) {bin[33]++; }
			else if (data >=136 && data <140) {bin[34]++; }
			else if (data >=140 && data <144) {bin[35]++; }
			else if (data >=144 && data <148) {bin[36]++; }
			else if (data >=148 && data <152) {bin[37]++; }
			else if (data >=152 && data <156) {bin[38]++; }
			else if (data >=156 && data <160) {bin[39]++; }
			else if (data >=160 && data <164) {bin[40]++; }
			else if (data >=164 && data <168) {bin[41]++; }
			else if (data >=168 && data <172) {bin[42]++; }
			else if (data >=172 && data <176) {bin[43]++; }
			else if (data >=176 && data <180) {bin[44]++; }
			else if (data >=180 && data <184) {bin[45]++; }
			else if (data >=184 && data <188) {bin[46]++; }
			else if (data >=188 && data <192) {bin[47]++; }
			else if (data >=192 && data <196) {bin[48]++; }
			else if (data >=196 && data <200) {bin[49]++; }
			else if (data >=200 && data <204) {bin[50]++; }
			else if (data >=204 && data <208) {bin[51]++; }
			else if (data >=208 && data <212) {bin[52]++; }
			else if (data >=212 && data <216) {bin[53]++; }
			else if (data >=216 && data <220) {bin[54]++; }
			else if (data >=220 && data <224) {bin[55]++; }
			else if (data >=224 && data <228) {bin[56]++; }
			else if (data >=228 && data <232) {bin[57]++; }
			else if (data >=232 && data <236) {bin[58]++; }
			else if (data >=236 && data <240) {bin[59]++; }
			else if (data >=240 && data <244) {bin[60]++; }
			else if (data >=244 && data <248) {bin[61]++; }
			else if (data >=248 && data <252) {bin[62]++; }
			else if (data >=252 && data <256) {bin[63]++; }

		}
		else if (histBinSize == 32){

			if (data >=0 && data <8) {bin[0]++; }
			else if (data >=8 && data <16) {bin[1]++; }
			else if (data >=16 && data <24) {bin[2]++; }
			else if (data >=24 && data <32) {bin[3]++; }
			else if (data >=32 && data <40) {bin[4]++; }
			else if (data >=40 && data <48) {bin[5]++; }
			else if (data >=48 && data <56) {bin[6]++; }
			else if (data >=56 && data <64) {bin[7]++; }
			else if (data >=64 && data <72) {bin[8]++; }
			else if (data >=72 && data <80) {bin[9]++; }
			else if (data >=80 && data <88) {bin[10]++; }
			else if (data >=88 && data <96) {bin[11]++; }
			else if (data >=96 && data <104) {bin[12]++; }
			else if (data >=104 && data <112) {bin[13]++; }
			else if (data >=112 && data <120) {bin[14]++; }
			else if (data >=120 && data <128) {bin[15]++; }
			else if (data >=128 && data <136) {bin[16]++; }
			else if (data >=136 && data <144) {bin[17]++; }
			else if (data >=144 && data <152) {bin[18]++; }
			else if (data >=152 && data <160) {bin[19]++; }
			else if (data >=160 && data <168) {bin[20]++; }
			else if (data >=168 && data <176) {bin[21]++; }
			else if (data >=176 && data <184) {bin[22]++; }
			else if (data >=184 && data <192) {bin[23]++; }
			else if (data >=192 && data <200) {bin[24]++; }
			else if (data >=200 && data <208) {bin[25]++; }
			else if (data >=208 && data <216) {bin[26]++; }
			else if (data >=216 && data <224) {bin[27]++; }
			else if (data >=224 && data <232) {bin[28]++; }
			else if (data >=232 && data <240) {bin[29]++; }
			else if (data >=240 && data <248) {bin[30]++; }
			else if (data >=248 && data <256) {bin[31]++; }

		}


		else if (histBinSize == 16){

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

			if (/*data >=0 && */ data <32) {bin[0]++; }
			else if (data >=32 && data <64) {bin[1]++; }
			else if (data >=64 && data <96) {bin[2]++; }
			else if (data >=96 && data <128) {bin[3]++; }
			else if (data >=128 && data <160) {bin[4]++; }
			else if (data >=160 && data <192) {bin[5]++; }
			else if (data >=192 && data <224) {bin[6]++; }
			else if (data >=224 /*&& data <256*/) {bin[7]++; }

		}

		else {
			std::cout << "Error: Incorrect bin size: " << histBinSize << std::endl;
			return;
		}
	}



	for (int i = 0; i < histBinSize; i++){
		bin[i] =  bin[i]/(float)szSample + 1.0e-8;
		std::cout << "Bin[" << i << "]: " << bin[i] << std::endl; 
	}
	std::cout << std::endl;

}
