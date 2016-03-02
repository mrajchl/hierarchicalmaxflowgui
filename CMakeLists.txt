################################################
PROJECT(HierarchicalMFGUI)
################################################

CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

#-----------------------------------------------------------------------------
# Set a default build type if none was specified
#-----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()


#-----------------------------------------------------------------------------
# ITK
#-----------------------------------------------------------------------------
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

#-----------------------------------------------------------------------------
# VTK
#-----------------------------------------------------------------------------
find_package(VTK REQUIRED)
if(NOT VTK_USE_QVTK)
  message(FATAL_ERROR "error: VTK was not configured to use QT, you probably need "
                      "to recompile it with VTK_USE_GUISUPPORT ON, VTK_USE_QVTK ON, "
                      "DESIRED_QT_VERSION 4 and QT_QMAKE_EXECUTABLE set appropriatly. "
                      "Note that Qt >= ${Slicer_REQUIRED_QT_VERSION} is *required*")
endif()
include(${VTK_USE_FILE})
link_directories(${VTK_LIBRARY_DIRS})

#-----------------------------------------------------------------------------
# Qt4
#-----------------------------------------------------------------------------
find_package( Qt4 REQUIRED )

if(QT_FOUND)
  include(${QT_USE_FILE})
    message(STATUS "Qt4 found.")
else(QT_FOUND)
  message(FATAL_ERROR "Cannot build  without Qt4.")
endif(QT_FOUND)
 message(STATUS ${QT_LIBRARIES})

set(QT_USE_QTSVG true)
set(QT_USE_QTCORE true)
set(QT_USE_QTGUI true)
set(QT_USE_QTXML true)

set(QT_USE_QTOPENGL true)
add_definitions(-DQT_SVG_LIB)

#-----------------------------------------------------------------------------
# CUDA
#-----------------------------------------------------------------------------
include(FindCUDA)


add_subdirectory(UI)

