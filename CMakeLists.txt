cmake_minimum_required(VERSION 3.10)
project(alpr)
enable_language(C)
enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss")

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES include)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_UFF_PARSER nvparsers ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNX_PARSER nvonnxparser ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 lib/x64)

set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_ONNX_PARSER} ${TENSORRT_LIBRARY_UFF_PARSER})
message(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
message(STATUS "Find TensorRT ONNX parser lib at ${TENSORRT_LIBRARY_ONNX_PARSER}")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# App
add_executable(app alpr/app.cu)
set_property(TARGET app PROPERTY CUDA_ARCHITECTURES 35 50 61 72)
target_link_libraries(app ${TENSORRT_LIBRARY})
target_link_libraries(app ${OpenCV_LIBS})

# App
add_executable(run alpr/run.cu)
set_property(TARGET run PROPERTY CUDA_ARCHITECTURES 35 50 61 72)
target_link_libraries(run ${TENSORRT_LIBRARY})
target_link_libraries(run ${OpenCV_LIBS})

# Image
add_executable(image alpr/image.cu)
set_property(TARGET image PROPERTY CUDA_ARCHITECTURES 35 50 61 72)
target_link_libraries(image ${TENSORRT_LIBRARY})
target_link_libraries(image ${OpenCV_LIBS})

# Strea
add_executable(stream alpr/stream.cu)
set_property(TARGET stream PROPERTY CUDA_ARCHITECTURES 35 50 61 72)
target_link_libraries(stream ${TENSORRT_LIBRARY})
target_link_libraries(stream ${OpenCV_LIBS})


# copition
configure_file(${CMAKE_SOURCE_DIR}/examples/example.jpg ${CMAKE_BINARY_DIR}/example.jpg COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/models/detection.uff ${CMAKE_BINARY_DIR}/detection.uff COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/models/recognation.cfg ${CMAKE_BINARY_DIR}/recognation.cfg COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/models/recognation.weights ${CMAKE_BINARY_DIR}/recognation.weights COPYONLY)
