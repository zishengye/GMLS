#ifndef _SOLVER_HPP_
#define _SOLVER_HPP_

#include <memory>

#include "network.hpp"
#include "parser.hpp"

// linear algebra
#include "sparse_matrix.h"

#include "geometry_manager.hpp"

// discretization
#include "gmls.hpp"

class solver {
private:
  std::shared_ptr<network> _net;

  // geometry
  geometry_manager _gm;

public:
  solver(std::shared_ptr<network> net) : _net(net), _gm(net) {}

  void attach_parser(std::shared_ptr<parser> par);
};

#endif