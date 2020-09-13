#ifndef _STOKES_EQUATION_HPP_
#define _STOKES_EQUATION_HPP_

#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

#include "equation.hpp"
#include "sparse_matrix.hpp"

class stokes_equation : public equation {
protected:
  std::vector<std::shared_ptr<sparse_matrix>> _ff;

  void
  build_matrix(std::shared_ptr<std::vector<particle>> particle_set,
               std::shared_ptr<std::vector<particle>> background_particle_set,
               std::shared_ptr<sparse_matrix> ff);

public:
  stokes_equation() {}
  ~stokes_equation() {}

  void build_coarse_level_matrix();

  void clear() { _ff.clear(); }
};

#endif