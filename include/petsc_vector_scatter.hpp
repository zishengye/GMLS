#ifndef _PETSC_VECTOR_SCATTER_HPP_
#define _PETSC_VECTOR_SCATTER_HPP_

#include <petscksp.h>

#include "petsc_index_set.hpp"
#include "petsc_vector.hpp"

class petsc_vecscatter {
private:
  VecScatter vec_scatter;

public:
  petsc_vecscatter() : vec_scatter(PETSC_NULL) {}

  ~petsc_vecscatter() {
    if (vec_scatter != PETSC_NULL)
      VecScatterDestroy(&vec_scatter);
  }

  void create(petsc_is &is, petsc_vector &vec1, petsc_vector &vec2);

  VecScatter &get_reference() { return vec_scatter; }

  VecScatter *get_pointer() { return &vec_scatter; }
};

#endif