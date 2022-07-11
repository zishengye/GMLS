#ifndef _StokesEquation_Hpp_
#define _StokesEquation_Hpp_

#include <memory>

#include <Compadre_Config.h>
#include <Compadre_Evaluator.hpp>
#include <Compadre_GMLS.hpp>
#include <Compadre_PointCloudSearch.hpp>

#include "ParticleGeometry.hpp"
#include "PetscWrapper.hpp"
#include "StokesMultilevelPreconditioning.hpp"
#include "rigid_body_manager.hpp"

#define VELOCITY_ERROR_EST 1
#define PRESSURE_ERROR_EST 2

class StokesEquation {
private:
  std::shared_ptr<ParticleGeometry> geoMgr_;
  std::shared_ptr<rigid_body_manager> rbMgr_;

  std::shared_ptr<StokesMultilevelPreconditioning> multiMgr_;

  void BuildCoefficientMatrix();
  void ConstructRhs();
  void SolveEquation();
  void CheckSolution();
  void CalculateError();
  void CollectForce();

  Kokkos::View<int **, Kokkos::HostSpace> neighbor_lists_;
  Kokkos::View<double *, Kokkos::HostSpace> epsilon_;
  Kokkos::View<double *, Kokkos::HostSpace> bi_;

  std::vector<double> rhs;
  std::vector<double> res;
  std::vector<double> error;
  std::vector<int> idx_colloid;

  std::vector<int> neumann_map;

  std::vector<Vec3> velocity;
  std::vector<double> pressure;

  std::vector<int> invert_row_index;

  std::vector<std::vector<double>> gradient;
  int gradient_component_num;

  int polyOrder_, dim_, errorEstimationMethod_;
  double eta_;
  int number_of_batches;

  int min_neighbor, max_neighbor;

  double global_error;

  int mpiRank_, mpiSize_;

  int currentRefinementLevel_;

  bool useViewer_;

public:
  StokesEquation() { useViewer_ = false; }

  void Init(std::shared_ptr<ParticleGeometry> geoMgr,
            std::shared_ptr<rigid_body_manager> rbMgr, const int polyOrder,
            const int dim, const int errorEstimationMethod = VELOCITY_ERROR_EST,
            const double epsilonMultiplier = 0.0, const double eta = 1.0);
  void Reset();
  void Update();

  void SetViewer() { useViewer_ = true; }

  std::vector<Vec3> &get_velocity() { return velocity; }

  std::vector<double> &get_pressure() { return pressure; }

  std::vector<std::vector<double>> &get_gradient() { return gradient; }

  std::vector<double> &get_error() { return error; }

  Kokkos::View<double *, Kokkos::HostSpace> &getEpsilon() { return epsilon_; }

  double get_estimated_error() { return global_error; }
};

#endif