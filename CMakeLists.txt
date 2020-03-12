cmake_minimum_required(VERSION 3.15)

project(gpgpu LANGUAGES CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fdiagnostics-color=always>
    $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>
)

add_subdirectory(deps)
add_subdirectory(matmul)
add_subdirectory(nbodies)
add_subdirectory(saxpy)
