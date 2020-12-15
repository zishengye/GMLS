#ifndef _PETSC_VECTOR_SCATTER_HPP_
#define _PETSC_VECTOR_SCATTER_HPP_

#include <petscksp.h>

#include "petsc_index_set.hpp"
#include "petsc_vector.hpp"

class petsc_vecscatter {
private:
  VecScatter vec_scatter;

  bool is_created;

public:
  petsc_vecscatter() : is_created(false) {}

  ~petsc_vecscatter() {
    if (is_created)
      VecScatterDestroy(&vec_scatter);
  }

  void create(petsc_is &is, petsc_vector &vec1, petsc_vector &vec2);

  VecScatter &get_reference() { return vec_scatter; }
};

#endif