#ifndef _STOKES_EQUATION_HPP_
#define _STOKES_EQUATION_HPP_

#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

#include "equation.hpp"
#include "sparse_matrix.hpp"

class stokes_equation : public equation {
protected:
  std::vector<std::shared_ptr<sparse_matrix>> _ff;
  std::vector<std::shared_ptr<sparse_matrix>> _interpolation;
  std::vector<std::shared_ptr<sparse_matrix>> _restriction;

  std::vector<std::shared_ptr<Vec>> _x;
  std::vector<std::shared_ptr<Vec>> _y;

  void
  build_matrix(std::shared_ptr<std::vector<particle>> particle_set,
               std::shared_ptr<std::vector<particle>> background_particle_set,
               std::shared_ptr<sparse_matrix> ff);

  void build_interpolation(
      std::shared_ptr<std::vector<particle>> coarse_grid_particle_set,
      std::shared_ptr<std::vector<particle>> fine_grid_particle_set,
      std::shared_ptr<sparse_matrix> interpolation);
  void build_restriction(
      std::shared_ptr<std::vector<particle>> coarse_grid_particle_set,
      std::shared_ptr<std::vector<particle>> fine_grid_particle_set,
      std::shared_ptr<sparse_matrix> restriction,
      std::shared_ptr<std::vector<std::vector<std::size_t>>> hierarchy);

public:
  stokes_equation() {}
  ~stokes_equation() {}

  void build_coarse_level_matrix();
  void build_interpolation_restriction_operators();

  void clear() {
    _ff.clear();
    _interpolation.clear();
    _restriction.clear();

    for (int i = 0; i < _x.size(); i++) {
      VecDestroy(_x[i].get());
      VecDestroy(_y[i].get());
    }
  }
};

#endif