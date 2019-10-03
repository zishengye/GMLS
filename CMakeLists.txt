cmake_minimum_required(VERSION 3.9.1)
project (GMLS_SOLVER)

# set(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)

include_directories("${PROJECT_SOURCE_DIR}/include/")
include_directories("/opt/kokkos/include/")
include_directories("/opt/petsc/include/")
include_directories("/opt/compadre/include")

link_directories("/opt/kokkos/lib/")
link_directories("/opt/petsc/lib/")
link_directories("/opt/compadre/lib/")

link_libraries("-lmpi -lmpicxx -lHYPRE -lpetsc -lfblas -lflapack -lstdc++ -lm")
link_libraries("-lcompadre")
link_libraries("-lkokkos")

file(GLOB_RECURSE sources ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_compile_options(-O3 -Wall)

add_executable(nsmpi ${sources} ${PROJECT_SOURCE_DIR}/main/ns.cpp)
