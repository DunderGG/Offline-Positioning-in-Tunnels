#check for the mininum version of cmake
cmake_minimum_required (VERSION 3.0 FATAL_ERROR)

#set
set (SHARED_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/header)
set (SHARED_INCLUDE_DIR ${SHARED_INCLUDE_DIR} PARENT_SCOPE)

if (TARGET shared)
    message ("shared library is already defined")
else()
    include_directories (${SHARED_INCLUDE_DIR})
    set (LIB_SOURCES    ./source/Calibration.cpp
                        ./source/Converter.cpp
                        ./source/FeatureDetection.cpp
                        ./source/ManualCorrespondences.cpp
                        ./source/PCLCloudSearch.cpp
                        ./source/PnPSolver.cpp
                        ./source/PointProjection.cpp
                        ./source/Reprojection.cpp
                        ./source/Triangulation.cpp
                        ./source/VisualOdometry.cpp
                        ./source/Common.cpp
                        ./source/Log.cpp
                        ./source/BundleAdjust.cpp)
    add_library (shared ${LIB_SOURCES})
endif()
