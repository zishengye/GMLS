#ifndef _PETSC_VECTOR_HPP_
#define _PETSC_VECTOR_HPP_

#include <vector>

#include <petscksp.h>

class petsc_vector {
private:
  Vec vec;

  // control flags
  bool is_created;

public:
  petsc_vector() : is_created(false) {}

  ~petsc_vector() {
    if (is_created)
      VecDestroy(&vec);
  }

  void create(std::vector<double> &_vec);
  void create(petsc_vector &_vec);

  void copy(std::vector<double> &_vec);

  Vec &get_reference() { return vec; }
};

#endif