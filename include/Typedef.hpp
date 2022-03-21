#ifndef _TYPEDEF_HPP_
#define _TYPEDEF_HPP_

#include <Compadre_Typedefs.hpp>

typedef scalar_type Scalar;
typedef local_index_type LocalIndex;
typedef global_index_type GlobalIndex;

typedef Kokkos::View<scalar_type **, Kokkos::DefaultHostExecutionSpace>
    HostRealMatrix;
typedef Kokkos::View<scalar_type **, Kokkos::DefaultExecutionSpace>
    DeviceRealMatrix;

typedef Kokkos::View<scalar_type *, Kokkos::DefaultHostExecutionSpace>
    HostRealVector;

typedef Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> HostIntVector;
typedef Kokkos::View<int *, Kokkos::DefaultExecutionSpace> DeviceIntVector;

typedef Kokkos::View<std::size_t *, Kokkos::DefaultHostExecutionSpace>
    HostIndexVector;
typedef Kokkos::View<std::size_t *, Kokkos::DefaultExecutionSpace>
    DeviceIndexVector;

#endif