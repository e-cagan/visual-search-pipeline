cmake_minimum_required(VERSION 3.10)
project(VisualSearchEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV Entegrasyonu
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime Entegrasyonu
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Ana Çalıştırılabilir Dosya (Executable)
add_executable(inference_engine 
    src/inference/main.cpp 
    src/inference/knn_engine.cpp
)

# Kütüphaneleri Bağla
target_link_libraries(inference_engine ${OpenCV_LIBS} onnxruntime)