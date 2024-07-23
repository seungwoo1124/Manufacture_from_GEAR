cmake_minimum_required(VERSION 3.10)
project(KiviGemv LANGUAGES CXX CUDA)

# CUDA 설정
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math")

# PyTorch 설정
find_package(Torch REQUIRED)

# CUDA 및 C++ 파일 목록
set(SOURCES
    gemv_cuda.cu
    pybind.cpp
)

# 타겟 정의
cuda_add_library(kivi_gemv SHARED ${SOURCES})
target_include_directories(kivi_gemv PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(kivi_gemv ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})

# PyTorch 확장 모듈 설정
set_target_properties(kivi_gemv PROPERTIES PREFIX "" SUFFIX ".so")

# 파이썬 모듈 설정
set(PYBIND11_CPP_STANDARD -std=c++14)