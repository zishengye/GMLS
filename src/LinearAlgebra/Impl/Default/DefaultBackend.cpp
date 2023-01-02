#include "LinearAlgebra/Impl/Default/DefaultBackend.hpp"
#include <mpi.h>

LinearAlgebra::Impl::DefaultBackend::DefaultBackend() {}

LinearAlgebra::Impl::DefaultBackend::~DefaultBackend() {}

Void LinearAlgebra::Impl::DefaultBackend::LinearAlgebraInitialize(
    int *argc, char ***args, std::string &fileName) {
  MPI_Init(argc, args);
}

Void LinearAlgebra::Impl::DefaultBackend::LinearAlgebraFinalize() {
  MPI_Finalize();
}