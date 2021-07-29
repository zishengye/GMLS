#ifndef _PETSC_NEST_VECTOR_HPP_
#define _PETSC_NEST_VECTOR_HPP_

#include <vector>

#include <petscksp.h>

class petsc_nest_vector {
private:
  Vec vec_list[2];

  Vec vec;

public:
  petsc_nest_vector() : vec(PETSC_NULL) {
    vec_list[0] = PETSC_NULL;
    vec_list[1] = PETSC_NULL;
  }

  ~petsc_nest_vector();

  void create(std::vector<double> &_vec1, std::vector<double> &_vec2);

  void copy(std::vector<double> &_vec1, std::vector<double> &_vec2);

  Vec &get_reference() { return vec; }

  Vec *get_pointer() { return &vec; }
};

#endif