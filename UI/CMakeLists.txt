
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

################################################

SET(HierarchicalMFGUI_SRCS  
main.cpp
iambusy.cpp
SliceViewer.cpp
MainApplication.cpp
OrthogonalView.cpp
QRangeSlider.cpp
qcolortablewidget.cpp
vtkRegionGrowing.cpp
LURange.cpp
QVolumeSlicer.cpp
vtkSurfacePipeline.cpp
vtkSurface.cpp
#vtkImageStatistics.cxx
vtkImageLuminance2.cpp
)


################################################
SET(HierarchicalMFGUI_HDRS  
iambusy.h
SliceViewer.h
MainApplication.h
OrthogonalView.h
QRangeSlider.h
qcolortablewidget.h
vtkRegionGrowing.h
LURange.h
#TVector2.h
QVolumeSlicer.h
vtkSurfacePipeline.h
vtkSurface.h
#vtkImageStatistics.h
vtkImageLuminance2.h
)


SET(HierarchicalMFGUI_MOCS
MainApplication.h
SliceViewer.h
OrthogonalView.h
QRangeSlider.h
qcolortablewidget.h
)


SET(HierarchicalMFGUI_UIS
 xyzwidget.ui
)


SET(UI_RESOURCES  Roil.qrc)

################################################
QT4_WRAP_UI( HierarchicalMFGUI_UI_HDRS ${HierarchicalMFGUI_UIS} )

QT4_WRAP_CPP( HierarchicalMFGUI_MOC_SRCS ${HierarchicalMFGUI_MOCS})

QT4_ADD_RESOURCES( RCS_SOURCES  ${UI_RESOURCES} )
#######################################################

SET(CPP_SOURCES ${HierarchicalMFGUI_SRCS} ${HierarchicalMFGUI_MOC_SRCS} ${HierarchicalMFGUI_UI_HDRS} ${HierarchicalMFGUI_HDRS} ${RCS_SOURCES})


SET(MAXFLOW_SOURCES 
  runMaxFlow.h
  runMaxFlow.cpp
  cmf3DHybridPottsCut.cu
)

#######################################################



LINK_LIBRARIES (
${QT_LIBRARIES} 
cuda
${ITK_LIBRARIES}
${VTK_LIBRARIES}
QVTK
)

################################################


CUDA_ADD_EXECUTABLE(HierarchicalMFGUI
    ${CPP_SOURCES}
    ${MAXFLOW_SOURCES}
    )

################################################


#TARGET_LINK_LIBRARIES( MaxFlowGUI maxFlow )