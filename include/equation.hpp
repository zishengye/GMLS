#ifndef _EQUATION_HPP_
#define _EQUATION_HPP_

#include <memory>
#include <vector>

#include "geometry.hpp"
#include "sparse_matrix.hpp"

class equation {
protected:
  std::shared_ptr<geometry> _geo;
  int _dimension;

public:
  equation() : _dimension(0) {}
  ~equation() {}

  void set_dimension(int dimension) { _dimension = dimension; }

  void add_geometry(std::shared_ptr<geometry> geo) { _geo = geo; }
};

#endif