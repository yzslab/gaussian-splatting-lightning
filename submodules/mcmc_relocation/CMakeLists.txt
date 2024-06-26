cmake_minimum_required(VERSION 3.28)

project(mcmc_relocation LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)


add_library(mcmc_relocation
        ext.cpp
        mcmc_relocation.cu)

set_target_properties(mcmc_relocation PROPERTIES CUDA_ARCHITECTURES "70;75;86")