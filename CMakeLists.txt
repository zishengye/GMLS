cmake_minimum_required(VERSION 3.9.0)
project (GMLS_SOLVER)

# set(CMAKE_VERBOSE_MAKEFILE on)

find_package(MPI)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories("${PROJECT_SOURCE_DIR}/include/")
include_directories("/opt/kokkos/include/")
include_directories("/opt/petsc/include/")
include_directories("/opt/compadre/include")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

link_directories("/opt/kokkos/lib/")
link_directories("/opt/petsc/lib/")
link_directories("/opt/compadre/lib/")

link_libraries("-lHYPRE -lpetsc -lfblas -lflapack -lstdc++ -lm -lgfortran -ldl -lgomp")
link_libraries("-lcompadre")
link_libraries("-lkokkos")
link_libraries(${MPI_CXX_LIBRARIES})

file(GLOB_RECURSE sources ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_compile_options(-O3 -w -fopenmp -fPIC -march=broadwell)

add_executable(nsmpi ${sources} ${PROJECT_SOURCE_DIR}/main/ns.cpp)
add_executable(test_geometry ${sources} ${PROJECT_SOURCE_DIR}/test/test_geometry.cpp)