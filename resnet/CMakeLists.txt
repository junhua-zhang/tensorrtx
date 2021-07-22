cmake_minimum_required(VERSION 2.6)

project(resnet)
set(OpenCV_DIR "C:\\Software\\opencv\\build")  #2
set(TRT_DIR "C:\\Software\\TensorRT-7.0.0.11")  #3

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TRT_DIR}\\include)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${TRT_DIR}\\lib)

add_executable(resnet18 ${PROJECT_SOURCE_DIR}/resnet18.cpp)
target_link_libraries(resnet18 nvinfer)
target_link_libraries(resnet18 cudart)

add_executable(resnet50 ${PROJECT_SOURCE_DIR}/resnet50.cpp)
target_link_libraries(resnet50 nvinfer)
target_link_libraries(resnet50 cudart)

add_executable(resnext50 ${PROJECT_SOURCE_DIR}/resnext50_32x4d.cpp)
target_link_libraries(resnext50 nvinfer)
target_link_libraries(resnext50 cudart)

add_definitions(-O2 -pthread)

