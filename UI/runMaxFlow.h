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


#ifndef _RUNMAXFLOW_H_
#define _RUNMAXFLOW_H_


// ITK
#include <itkImage.h>


#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkCastImageFilter.h>
#include <itkStatisticsImageFilter.h>


// VTK
#include "vtkMetaImageReader.h"
#include "vtkMetaImageWriter.h"
#include "vtkSmartPointer.h"
#include "vtkImageCast.h"

#include "itkImageToVTKImageFilter.h"
#include "itkVTKImageToImageFilter.h"



void runMaxFlow(vtkImageData*, vtkImageData*,vtkImageData*,float*,float*); 

#endif // _RUNMAXFLOW_H_
