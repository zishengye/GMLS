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
./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --with-debugging=no COPTFLAGS='-fPIC -O3 -march=native -mtune=native' CXXOPTFLAGS='-fPIC -O3 -march=native -mtune=native' FOPTFLAGS='-fPIC -O3 -march=native -mtune=native' --with-blaslapack-dir=/opt/intel/oneapi/mkl/latest/ --with-openmp --with-threadcomm --download-hwloc --prefix=/opt/petsc/
```

[Trilinos](https://github.com/trilinos/Trilinos)



```[cmake]
cmake -D Trilinos_ENABLE_Zoltan2:BOOL=ON -D Trilinos_ENABLE_Compadre:BOOL=ON -D CMAKE_INSTALL_PREFIX:PATH=$PREFIX_PATH -D Trilinos_ENABLE_Fortran:BOOL=OFF -D BUILD_SHARED_LIBS=ON -D TPL_ENABLE_MPI:BOOL=ON -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF -D Kokkos_ENABLE_OPENMP:BOOL=ON -D Kokkos_ENABLE_SERIAL:BOOL=OFF -D Trilinos_ENABLE_OpenMP:BOOL=ON ../
```

If use GPU as the accelerator

```[cmake]
export OMPI_CXX=/bin/nvcc_wrapper
cmake -D Trilinos_ENABLE_Zoltan2:BOOL=ON -D Trilinos_ENABLE_Compadre:BOOL=ON -D CMAKE_INSTALL_PREFIX:PATH=$PREFIX_PATH -D Trilinos_ENABLE_Fortran:BOOL=OFF -D BUILD_SHARED_LIBS=ON -D TPL_ENABLE_MPI:BOOL=ON -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF -D BUILD_SHARED_LIBS:BOOL=ON -D Trilinos_ENABLE_CXX11:BOOL=ON -D CMAKE_CXX_FLAGS:STRING='-expt-extended-lambda -Wall -Wno-unknown-pragmas -Wno-unused-but-set-variable -Wno-inline -Wshadow' -D Trilinos_ENABLE_Fortran:BOOL=OFF -D Kokkos_ENABLE_OPENMP:BOOL=ON -D Kokkos_ENABLE_SERIAL:BOOL=OFF -D TPL_ENABLE_CUDA:BOOL=ON -D TPL_ENABLE_CUSPARSE:BOOL=OFF -D TPL_ENABLE_HWLOC:BOOL=OFF -D TPL_ENABLE_BLAS:BOOL=ON -D TPL_ENABLE_LAPACK:BOOL=ON -D Kokkos_ENABLE_CUDA:BOOL=ON -D Kokkos_ENABLE_CUDA_UVM:BOOL=ON -D Kokkos_ENABLE_CUDA_LAMBDA:BOOL=ON -D Kokkos_ARCH_BDW:BOOL=ON -D Kokkos_ARCH_AMPERE86:BOOL=ON ../
```

## How to execute the program

To invoke the executable, one need to use mpiexec and use "-input" command to transfer execution command for the program. Here is an example

```[bash]
/opt/petsc/bin/mpiexec -np 4 ./nsmpi -input ./example/stokes_2d.in
```
