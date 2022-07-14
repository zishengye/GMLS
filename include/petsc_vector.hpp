#ifndef _PETSC_VECTOR_HPP_
#define _PETSC_VECTOR_HPP_

#include <vector>

#include <petscksp.h>

class petsc_vector {
private:
  Vec vec_;

public:
  petsc_vector() : vec_(PETSC_NULL) {}

  ~petsc_vector() {
    if (vec_ != PETSC_NULL)
      VecDestroy(&vec_);
  }

  void create(const std::vector<double> &vec);
  void create(petsc_vector &vec);

  void copy(std::vector<double> &vec);

  Vec &GetReference() { return vec_; }

  Vec *GetPointer() { return &vec_; }
};

#endif