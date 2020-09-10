#ifndef _EQUATION_HPP_
#define _EQUATION_HPP_

#include <memory>

#include "geometry.hpp"
#include "sparse_matrix.hpp"

class equation {
protected:
  std::shared_ptr<geometry> _geo;

public:
  equation();
  ~equation();

  add_geometry(std::shared_ptr<geometry> geo) { _geo = geo; }
};

#endif