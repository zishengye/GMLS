#include "petsc_vector_scatter.hpp"

void petsc_vecscatter::create(petsc_is &is, petsc_vector &vec1,
                              petsc_vector &vec2) {
  VecScatterCreate(vec1.GetReference(), is.GetReference(), vec2.GetReference(),
                   NULL, &vec_scatter);
}