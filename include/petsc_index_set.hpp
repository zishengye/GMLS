#ifndef _PETSC_INDEX_SET_HPP_
#define _PETSC_INDEX_SET_HPP_

#include <vector>

#include <petscksp.h>

class petsc_is {
private:
  IS is;

public:
  petsc_is() : is(PETSC_NULL) {}

  ~petsc_is() {
    if (is != PETSC_NULL)
      ISDestroy(&is);
  }

  void create(std::vector<int> &idx);
  void create_local(std::vector<int> &idx);

  IS &GetReference() { return is; }

  IS *GetPointer() { return &is; }
};

#endif