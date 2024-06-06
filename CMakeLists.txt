cmake_minimum_required(VERSION 3.10)

# Do no set the host compiler to a version higher than the one supported by
# nvcc:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
# This may be unnecessary and has to be updated every term based on the
# the gcc version of the lab / local machine.
# set(CMAKE_CXX_COMPILER "g++-8")

set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-10")

project(practice LANGUAGES CXX CUDA)

add_subdirectory(googletest)

set(CMAKE_CUDA_FLAGS "-arch=sm_61")

add_library(abdulrahman_exploration_options INTERFACE)
target_compile_features(abdulrahman_exploration_options INTERFACE cxx_std_11)

add_library(abdulrahman_exploration_kernels
  best_cpu.cu
  pgm.cu
  kernel1.cu
  kernel2.cu
  kernel3.cu
  kernel4.cu
  kernel5.cu
)

target_link_libraries(abdulrahman_exploration_kernels
  PUBLIC abdulrahman_exploration_options
)

add_executable(main
  main.cu
)

target_link_libraries(main
  PUBLIC abdulrahman_exploration_kernels abdulrahman_exploration_options
)

add_library(pgm
  pgm.cu
)

set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 61)

add_executable(pgm_creator
  pgm_creator.cu
)

target_link_libraries(pgm_creator
  PUBLIC pgm abdulrahman_exploration_options
)

set_property(TARGET pgm_creator PROPERTY CUDA_ARCHITECTURES 61)
set_property(TARGET pgm PROPERTY CUDA_ARCHITECTURES 61)

add_executable(test_solution
  tests.cu
)
target_link_libraries(test_solution
  PUBLIC pgm abdulrahman_exploration_kernels gtest abdulrahman_exploration_options
)

set_property(TARGET test_solution PROPERTY CUDA_ARCHITECTURES 61)

set_property(TARGET abdulrahman_exploration_kernels PROPERTY CUDA_ARCHITECTURES 61)
