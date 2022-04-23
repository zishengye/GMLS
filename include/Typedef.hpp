#ifndef _Typedef_Hpp_
#define _Typedef_Hpp_

#include <Compadre_Typedefs.hpp>

typedef scalar_type Scalar;
typedef local_index_type LocalIndex;
typedef global_index_type GlobalIndex;

typedef Kokkos::View<Scalar **, Kokkos::DefaultHostExecutionSpace>
    HostRealMatrix;
typedef Kokkos::View<Scalar **, Kokkos::DefaultExecutionSpace> DeviceRealMatrix;

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

#endif