#check for the mininum version of cmake
cmake_minimum_required (VERSION 3.0 FATAL_ERROR)

#set
set (SHARED_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/header)
set (SHARED_INCLUDE_DIR ${SHARED_INCLUDE_DIR} PARENT_SCOPE)

if (TARGET shared)
    message ("shared library is already defined")
else()
    include_directories (${SHARED_INCLUDE_DIR})
    set (LIB_SOURCES    ./header/Calibration.h
                        ./header/Converter.h
                        ./header/FeatureDetection.h
                        ./header/ManualCorrespondences.h
                        ./header/PCLCloudSearch.h
                        ./header/PnPSolver.h
                        ./header/PointProjection.h
                        ./header/Reprojection.h
                        ./header/VisualOdometry.h
                        ./header/Common.h
                        ./header/BundleAdjust.h
                        ./header/Frame.h

                        ./source/Calibration.cpp
                        ./source/Converter.cpp
                        ./source/FeatureDetection.cpp
                        ./source/ManualCorrespondences.cpp
                        ./source/PCLCloudSearch.cpp
                        ./source/PnPSolver.cpp
                        ./source/PointProjection.cpp
                        ./source/Reprojection.cpp
                        ./source/VisualOdometry.cpp
                        ./source/Common.cpp
                        ./source/BundleAdjust.cpp
                        ./source/Frame.cpp)
                        #./source/Triangulation.cpp
                        #./source/Log.cpp
    add_library (shared ${LIB_SOURCES})
endif()
