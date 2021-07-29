#include "petsc_block_matrix.hpp"

petsc_block_matrix::~petsc_block_matrix() {
  for (int i = 0; i < mat_list.size(); i++) {
    MatDestroy(&mat_list[i]);
  }
  if (is_assembled)
    MatDestroy(&mat);
}