#check for the mininum version of cmake
cmake_minimum_required (VERSION 3.0 FATAL_ERROR)

#project name
project (OPiT)

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#pcl
find_package (PCL 1.6 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#opencl
# find_package(OpenCL REQUIRED)

#add include header
# add_subdirectory (src)

#add all of the sources & headers except the main
file (GLOB src_files
    "*.cpp"
    "*.h")

list (REMOVE_ITEM src_files ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp)

#compile flag
# add_definitions(-DKITTI_DATASET)

#set compiler flag to C++11
set (CMAKE_CXX_COMPILER "/usr/bin/clang++")
set (CMAKE_CXX_FLAGS    "-std=c++11 -stdlib=libc++ -g")

#build the library for the source files
add_library (src ${src_files})

#executable
add_executable(OPiT Main.cpp)

#linker
target_link_libraries(OPiT src ${OpenCV_LIBS} ${PCL_LIBRARIES})
