#ifndef _EQUATION_HPP_
#define _EQUATION_HPP_

#include <memory>
#include <vector>

#include "geometry.hpp"
#include "sparse_matrix.hpp"

class equation {
protected:
  int _dimension;

public:
  equation() : _dimension(0) {}
  ~equation() {}

  void set_dimension(int dimension) { _dimension = dimension; }

  void add_geometry(std::shared_ptr<geometry> geo) { _geo = geo; }

  std::shared_ptr<geometry> _geo;
};

#endif