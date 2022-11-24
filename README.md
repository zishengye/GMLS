# GMLS Solver

## How to build

In the root directory, use

```[cmake]
cmake ./
```

build up all the dependencies and use

```[make]
make nsmpi -j
```

build the executable.

## Preliminary

[PETSc](https://www.mcs.anl.gov/petsc/)

Suggested installation command:

```[bash]
./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --with-debugging=0 COPTFLAGS='-O3 -march=native -mtune=native' CXXOPTFLAGS='-O3 -march=native -mtune=native' FOPTFLAGS='-O3 -march=native -mtune=native' --download-mpich --download-fblaslapack --download-superlu_dist --download-mumps --download-scalapack --download-hypre --prefix=/opt/petsc/
```

[Kokkos](https://github.com/kokkos/kokkos-kernels)

It is suggested to install a standalone Kokkos kernel. One could directly install the kernel from Compadre package listed below. Up to now, in order to cooperate with PETSc, an openmp version Kokkos is suggested to be installed.

Suggested installation commamd:

```[bash]
../generate_makefile.bash --prefix=/opt/kokkos --cxxflags="-fPIC" --ldflags="-ldl"
```

[Compadre](https://github.com/SNLComputation/compadre)

Suggested configure file setup

```[bash]
#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 


# Serial on CPU via Kokkos
# No Python interface
# Standalone Kokkos

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

# pick your favorite c++ compiler
MY_CXX_COMPILER=`which g++`

# this will install in your build directory in a folder called install by default
INSTALL_PREFIX="/opt/compadre/"
MY_KOKKOSCORE_PREFIX="/opt/kokkos/"

cmake \
    -D CMAKE_CXX_COMPILER="$MY_CXX_COMPILER" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -D Compadre_USE_PYTHON:BOOL=OFF \
    -D KokkosCore_PREFIX="$MY_KOKKOSCORE_PREFIX" \
    -D Compadre_USE_OpenMP:BOOL=ON \
    \
    ..
```

```[cmake]
cmake -D Trilinos_ENABLE_Zoltan2:BOOL=ON -D CMAKE_INSTALL_PREFIX:PATH=$PREFIX_PATH -D Trilinos_ENABLE_Fortran:BOOL=OFF -D BUILD_SHARED_LIBS=ON -D TPL_ENABLE_MPI:BOOL=ON -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF -D Kokkos_ENABLE_OpenMP:BOOL=ON -D Kokkos_ENABLE_SERIAL:BOOL=OFF -D Trilinos_ENABLE_OpenMP:BOOL=ON ../
```

If use GPU as the accelerator

```[cmake]
export OMPI_CXX=/bin/nvcc_wrapper
cmake -D Trilinos_ENABLE_Zoltan2:BOOL=ON -D Trilinos_ENABLE_Compadre:BOOL=ON -D CMAKE_INSTALL_PREFIX:PATH=$PREFIX_PATH -D Trilinos_ENABLE_Fortran:BOOL=OFF -D BUILD_SHARED_LIBS=ON -D TPL_ENABLE_MPI:BOOL=ON -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF -D BUILD_SHARED_LIBS:BOOL=ON -D Trilinos_ENABLE_CXX11:BOOL=ON -D CMAKE_CXX_FLAGS:STRING='-expt-extended-lambda -Wall -Wno-unknown-pragmas -Wno-unused-but-set-variable -Wno-inline -Wshadow' -D Trilinos_ENABLE_Fortran:BOOL=OFF -D TPL_ENABLE_CUDA:BOOL=ON -D TPL_ENABLE_CUSPARSE:BOOL=OFF -D TPL_ENABLE_HWLOC:BOOL=OFF -D TPL_ENABLE_BLAS:BOOL=ON -D TPL_ENABLE_LAPACK:BOOL=ON -D Kokkos_ENABLE_CUDA:BOOL=ON -D Kokkos_ENABLE_CUDA_UVM:BOOL=ON -D Kokkos_ENABLE_CUDA_LAMBDA:BOOL=ON -D Kokkos_ARCH_BDW:BOOL=ON -D Kokkos_ARCH_AMPERE86:BOOL=ON ../
```

## How to execute the program

To invoke the executable, one need to use mpiexec and use "-input" command to transfer execution command for the program. Here is an example

```[bash]
/opt/petsc/bin/mpiexec -np 4 ./nsmpi -input ./example/stokes_2d.in
```
