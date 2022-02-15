cmake_minimum_required(VERSION 2.6)

project(resnet)
set(OpenCV_DIR "C:\\Software\\opencv\\build")  #2
set(TRT_DIR "C:\\Software\\TensorRT-8.0.3.4")  #3

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

find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
	
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${TRT_DIR}\\lib)

add_executable(resnet18 ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/classificator.h ${PROJECT_SOURCE_DIR}/classificator.cpp ${PROJECT_SOURCE_DIR}/resnet18.h ${PROJECT_SOURCE_DIR}/resnet18.cpp ${PROJECT_SOURCE_DIR}/common.h ${PROJECT_SOURCE_DIR}/common.cpp)
#add_executable(resnet18 ${PROJECT_SOURCE_DIR}/classificator.h ${PROJECT_SOURCE_DIR}/classificator.cpp ${PROJECT_SOURCE_DIR}/resnet18.h ${PROJECT_SOURCE_DIR}/resnet18.cpp ${PROJECT_SOURCE_DIR}/common.h ${PROJECT_SOURCE_DIR}/common.cpp)
target_link_libraries(resnet18 nvinfer)
# target_link_libraries(resnet18 cudart)
target_link_libraries(resnet18 ${OpenCV_LIBS})          #6
target_link_libraries(resnet18 ${CUDA_LIBRARIES})   #7

#add_executable(resnet50 ${PROJECT_SOURCE_DIR}/resnet50.cpp)
#target_link_libraries(resnet50 nvinfer)
##target_link_libraries(resnet50 cudart)
#target_link_libraries(resnet50 ${OpenCV_LIBS})          #6
#target_link_libraries(resnet50 ${CUDA_LIBRARIES})   #7
#
#add_executable(resnext50 ${PROJECT_SOURCE_DIR}/resnext50_32x4d.cpp)
#target_link_libraries(resnext50 nvinfer)
##target_link_libraries(resnext50 cudart)
#target_link_libraries(resnext50 ${OpenCV_LIBS})          #6
#target_link_libraries(resnext50 ${CUDA_LIBRARIES})   #7

add_definitions(-O2 -pthread)

