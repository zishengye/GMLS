#include "petsc_index_set.hpp"

void petsc_is::create(std::vector<int> &idx) {
  ISCreateGeneral(MPI_COMM_WORLD, idx.size(), idx.data(), PETSC_COPY_VALUES,
                  &is);
}

void petsc_is::create_local(std::vector<int> &idx) {
  ISCreateGeneral(MPI_COMM_SELF, idx.size(), idx.data(), PETSC_COPY_VALUES,
                  &is);
}