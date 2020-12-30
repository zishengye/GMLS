#ifndef _PETSC_VECTOR_HPP_
#define _PETSC_VECTOR_HPP_

#include <vector>

#include <petscksp.h>

class petsc_vector {
private:
  Vec vec;

public:
  petsc_vector() : vec(PETSC_NULL) {}

  ~petsc_vector() {
    if (vec != PETSC_NULL)
      VecDestroy(&vec);
  }

  void create(std::vector<double> &_vec);
  void create(petsc_vector &_vec);

  void copy(std::vector<double> &_vec);

  Vec &get_reference() { return vec; }

  Vec *get_pointer() { return &vec; }
};

#endif