cmake_minimum_required(VERSION 3.9.0)
project(GMLS_SOLVER)

# set(CMAKE_VERBOSE_MAKEFILE on)
find_package(GTest REQUIRED)
find_package(MPI)

set(TRILINOS_PATH /opt/trilinos/)
find_package(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos ${TRILINOS_PATH})

IF(Trilinos_FOUND)
  MESSAGE("\nFound Trilinos!  Here are the details: ")
  MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
  MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
  MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
  MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
  MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
  MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
  MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
  MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
  MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
  MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
  MESSAGE("   Trilinos_C_COMPILER = ${Trilinos_C_COMPILER}")
  MESSAGE("   Trilinos_Fortran_COMPILER = ${Trilinos_Fortran_COMPILER}")
  MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
  MESSAGE("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
  MESSAGE("   Trilinos_Fortran_COMPILER_FLAGS =
     ${Trilinos_Fortran_COMPILER_FLAGS}")
  MESSAGE("   Trilinos_LINKER = ${Trilinos_LINKER}")
  MESSAGE("   Trilinos_EXTRA_LD_FLAGS = ${Trilinos_EXTRA_LD_FLAGS}")
  MESSAGE("   Trilinos_AR = ${Trilinos_AR}")
  MESSAGE("End of Trilinos details\n")
ELSE()
  MESSAGE(FATAL_ERROR "Could not find Trilinos!")
ENDIF()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)

include_directories("${PROJECT_SOURCE_DIR}/include/")
include_directories("/opt/petsc/include/")
include_directories("/opt/compadre/include")
include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${Trilinos_TPL_INCLUDE_DIRS})
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

link_directories("/opt/petsc/lib/")
link_directories("/opt/compadre/lib/")
link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

link_libraries("-lpetsc -lstdc++ -lm -lgfortran -ldl -lgomp -ltbb")
link_libraries("-lcompadre")
link_libraries(${Trilinos_LIBRARIES})
link_libraries(${Trilinos_TPL_LIBRARIES})
link_libraries(${MPI_CXX_LIBRARIES})

file(GLOB_RECURSE sources ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_compile_options(-O3 -Werror -Wall -Wextra -Wwrite-strings -Wno-cast-function-type -Wno-unused-function -Wno-unused-variable -Wno-unused-parameter -Wno-deprecated-copy -Wno-reorder -Wno-sign-compare -Wno-unused-but-set-variable -Wno-deprecated-declarations -Wno-unused-result -Wno-implicit-fallthrough -fopenmp -fPIC)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_library(GmlsModule SHARED ${sources})

include(${PROJECT_SOURCE_DIR}/test/UnitTest.cmake)