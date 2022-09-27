#ifndef _Typedef_Hpp_
#define _Typedef_Hpp_

#include <Compadre_Typedefs.hpp>

typedef scalar_type Scalar;
typedef local_index_type LocalIndex;
typedef global_index_type GlobalIndex;

typedef Kokkos::View<Scalar **, Kokkos::DefaultHostExecutionSpace>
    HostRealMatrix;
typedef Kokkos::View<Scalar **, Kokkos::DefaultExecutionSpace> DeviceRealMatrix;

typedef Kokkos::View<std::size_t **, Kokkos::DefaultHostExecutionSpace>
    HostIndexMatrix;
typedef Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
    DeviceIndexMatrix;

typedef Kokkos::View<int **, Kokkos::DefaultHostExecutionSpace> HostIntMatrix;
typedef Kokkos::View<int **, Kokkos::DefaultExecutionSpace> DeviceIntMatrix;

typedef Kokkos::View<Scalar *, Kokkos::DefaultHostExecutionSpace>
    HostRealVector;
typedef Kokkos::View<Scalar *, Kokkos::DefaultExecutionSpace> DeviceRealVector;

typedef Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> HostIntVector;
typedef Kokkos::View<int *, Kokkos::DefaultExecutionSpace> DeviceIntVector;

typedef Kokkos::View<std::size_t *, Kokkos::DefaultHostExecutionSpace>
    HostIndexVector;
typedef Kokkos::View<std::size_t *, Kokkos::DefaultExecutionSpace>
    DeviceIndexVector;

#include <cstdint>
#include <limits>

#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "WRONG SIZE_T SIZING"
#endif

#include "LinearAlgebra/Impl/Petsc/PetscBackend.hpp"
typedef PetscBackend DefaultLinearAlgebraBackend;

#endif