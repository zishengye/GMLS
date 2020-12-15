#include "petsc_vector_scatter.hpp"

void petsc_vecscatter::create(petsc_is &is, petsc_vector &vec1,
                              petsc_vector &vec2) {
  VecScatterCreate(vec1.get_reference(), is.get_reference(),
                   vec2.get_reference(), NULL, &vec_scatter);

  is_created = true;
}