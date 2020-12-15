#ifndef _PETSC_INDEX_SET_HPP_
#define _PETSC_INDEX_SET_HPP_

#include <vector>

#include <petscksp.h>

class petsc_is {
private:
  IS is;

  // control flags
  bool is_created;

public:
  petsc_is() : is_created(false) {}

  ~petsc_is() {
    if (is_created)
      ISDestroy(&is);
  }

  void create(std::vector<int> &idx);
  void create_local(std::vector<int> &idx);

  IS &get_reference() { return is; }
};

#endif