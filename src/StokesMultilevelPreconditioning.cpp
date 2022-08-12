#include "StokesMultilevelPreconditioning.hpp"
#include "PetscNestedMatrix.hpp"
#include "PetscVector.hpp"
#include "gmls_solver.hpp"
#include "petscksp.h"
#include "petscpc.h"
#include "petscsys.h"
#include "petscvec.h"

#include <algorithm>
#include <iostream>
#include <mpi.h>

using namespace std;
using namespace Compadre;

PetscErrorCode StokesMultilevelIterationWrapper(PC pc, Vec x, Vec y) {
  StokesMultilevelPreconditioning *ctx;
  PCShellGetContext(pc, (void **)&ctx);

  return ctx->MultilevelIteration(x, y);
}

void StokesMultilevelPreconditioning::BuildInterpolationRestrictionOperators(
    const int numRigidBody, const int dimension_, const int polyOrder) {
  PetscNestedMatrix &interpolation =
      *(interpolationList_[currentRefinementLevel_ - 1]);
  PetscNestedMatrix &restriction =
      *(restrictionList_[currentRefinementLevel_ - 1]);

  int fieldDof = dimension_ + 1;
  int velocityDof = dimension_;

  int rigidBodyDof = (dimension_ == 3) ? 6 : 3;

  double timer1, timer2;
  timer1 = MPI_Wtime();

  unsigned int numLocalRigidBody, rigidBodyStartIndex, rigidBodyEndIndex;
  numLocalRigidBody =
      numRigidBody / (unsigned int)mpiSize_ +
      ((numRigidBody % (unsigned int)mpiSize_ > mpiRank_) ? 1 : 0);

  rigidBodyStartIndex = 0;
  for (int i = 0; i < mpiRank_; i++) {
    rigidBodyStartIndex +=
        numRigidBody / (unsigned int)mpiSize_ +
        ((numRigidBody % (unsigned int)mpiSize_ > i) ? 1 : 0);
  }
  rigidBodyEndIndex = rigidBodyStartIndex + numLocalRigidBody;

  {
    auto &coord = *(geoMgr_->get_current_work_particle_coord());
    auto &new_added = *(geoMgr_->get_current_work_particle_new_added());
    auto &spacing = *(geoMgr_->get_current_work_particle_spacing());
    auto &local_idx = *(geoMgr_->get_current_work_particle_local_index());

    auto &old_coord = *(geoMgr_->get_last_work_particle_coord());

    auto &old_source_coord = *(geoMgr_->get_clll_particle_coord());
    auto &old_source_index = *(geoMgr_->get_clll_particle_index());

    unsigned int numNewLocalParticle = coord.size();

    unsigned int numNewGlobalParticleNum;

    MPI_Allreduce(&numNewLocalParticle, &numNewGlobalParticleNum, 1,
                  MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

    unsigned int numOldLocalParticle = old_coord.size();
    unsigned int numOldGlobalParticle;

    MPI_Allreduce(&numOldLocalParticle, &numOldGlobalParticle, 1, MPI_UNSIGNED,
                  MPI_SUM, MPI_COMM_WORLD);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        old_source_coords_device("old source coordinates",
                                 old_source_coord.size(), 3);
    Kokkos::View<double **>::HostMirror old_source_coords_host =
        Kokkos::create_mirror_view(old_source_coords_device);

    int actual_new_target = 0;
    vector<int> new_actual_index(coord.size());
    for (int i = 0; i < coord.size(); i++) {
      new_actual_index[i] = actual_new_target;
      if (new_added[i] < 0)
        actual_new_target++;
    }

    int global_actual_new_target;
    int min_new_target, max_new_target;
    MPI_Allreduce(&actual_new_target, &global_actual_new_target, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&actual_new_target, &min_new_target, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&actual_new_target, &max_new_target, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Total new particle: %d\n",
                global_actual_new_target);
    PetscPrintf(PETSC_COMM_WORLD, "new particle imbalance: %f\n",
                (double)(max_new_target) / (double)(min_new_target));

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        new_target_coords_device("new target coordinates", actual_new_target,
                                 3);
    Kokkos::View<double **>::HostMirror new_target_coords_host =
        Kokkos::create_mirror_view(new_target_coords_device);

    // copy old source coords
    for (int i = 0; i < old_source_coord.size(); i++) {
      for (int j = 0; j < dimension_; j++)
        old_source_coords_host(i, j) = old_source_coord[i][j];
    }

    // copy new target coords
    int counter = 0;
    for (int i = 0; i < coord.size(); i++) {
      if (new_added[i] < 0) {
        for (int j = 0; j < dimension_; j++) {
          new_target_coords_host(counter, j) = coord[i][j];
        }

        counter++;
      }
    }

    Kokkos::deep_copy(old_source_coords_device, old_source_coords_host);
    Kokkos::deep_copy(new_target_coords_device, new_target_coords_host);

    double max_epsilon = geoMgr_->get_old_cutoff_distance();
    int ite_counter = 1;
    int min_neighbor = 1000, max_neighbor = 0;

    auto neighbor_needed =
        2.0 * Compadre::GMLS::getNP(polyOrder, dimension_,
                                    DivergenceFreeVectorTaylorPolynomial);

    int estimated_num_neighbor_max = neighbor_needed + 1;

    Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
        old_to_new_neighbor_lists_device("old to new neighbor lists",
                                         actual_new_target,
                                         estimated_num_neighbor_max);
    Kokkos::View<int **>::HostMirror old_to_new_neighbor_lists_host =
        Kokkos::create_mirror_view(old_to_new_neighbor_lists_device);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> old_epsilon_device(
        "h supports", actual_new_target);
    Kokkos::View<double *>::HostMirror old_epsilon_host =
        Kokkos::create_mirror_view(old_epsilon_device);

    {
      auto old_to_new_point_search(
          CreatePointCloudSearch(old_source_coords_host, dimension_));

      for (int i = 0; i < coord.size(); i++) {
        if (new_added[i] < 0) {
          old_epsilon_host[new_actual_index[i]] = spacing[i];
        }
      }

      size_t actual_neighbor_max;

      double sub_timer1, sub_timer2;
      sub_timer1 = MPI_Wtime();

      old_to_new_point_search.generate2DNeighborListsFromKNNSearch(
          true, new_target_coords_host, old_to_new_neighbor_lists_host,
          old_epsilon_host, neighbor_needed, 1.0);

      counter = 0;
      for (int i = 0; i < coord.size(); i++) {
        if (new_added[i] < 0) {
          double minEpsilon = 1.50 * spacing[i];
          double minSpacing = 0.25 * spacing[i];
          old_epsilon_host(new_actual_index[i]) =
              std::max(minEpsilon, old_epsilon_host(new_actual_index[i]));

          int scaling = std::max(
              0.0,
              std::ceil((old_epsilon_host(new_actual_index[i]) - minEpsilon) /
                        minSpacing));
          old_epsilon_host(new_actual_index[i]) =
              minEpsilon + scaling * minSpacing;

          counter++;
        }
      }

      auto actual_neighbor =
          old_to_new_point_search.generate2DNeighborListsFromRadiusSearch(
              true, new_target_coords_host, old_to_new_neighbor_lists_host,
              old_epsilon_host, 0.0, 0.0) +
          1;
      if (actual_neighbor > old_to_new_neighbor_lists_host.extent(1)) {
        Kokkos::resize(old_to_new_neighbor_lists_device, actual_new_target,
                       actual_neighbor);
        old_to_new_neighbor_lists_host =
            Kokkos::create_mirror_view(old_to_new_neighbor_lists_device);
      }
      old_to_new_point_search.generate2DNeighborListsFromRadiusSearch(
          false, new_target_coords_host, old_to_new_neighbor_lists_host,
          old_epsilon_host, 0.0, 0.0);

      min_neighbor = 1000;
      max_neighbor = 0;
      int enough_neighbor = 0;
      for (int i = 0; i < coord.size(); i++) {
        if (new_added[i] < 0) {
          int num_neighbor =
              old_to_new_neighbor_lists_host(new_actual_index[i], 0);
          if (num_neighbor < neighbor_needed) {
            if ((old_epsilon_host[new_actual_index[i]] + 0.25 * spacing[i]) <
                max_epsilon) {
              old_epsilon_host[new_actual_index[i]] += 0.25 * spacing[i];
              enough_neighbor = 1;
            }
          }
          if (min_neighbor > num_neighbor)
            min_neighbor = num_neighbor;
          if (max_neighbor < num_neighbor)
            max_neighbor = num_neighbor;
        }
      }

      MPI_Allreduce(MPI_IN_PLACE, &min_neighbor, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &max_neighbor, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);

      MPI_Allreduce(MPI_IN_PLACE, &enough_neighbor, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);

      sub_timer2 = MPI_Wtime();

      PetscPrintf(
          PETSC_COMM_WORLD,
          "iteration count: %d, min neighbor: %d, max neighbor: %d, time "
          "duration: %fs\n",
          ite_counter, min_neighbor, max_neighbor, sub_timer2 - sub_timer1);
    }

    PetscLogDouble minMem, maxMem, mem;
    PetscMemoryGetCurrentUsage(&mem);
    MPI_Allreduce(&mem, &maxMem, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&mem, &minMem, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD,
                "Current memory usage %.2f GB, maximum memory usage: %.2f GB, "
                "minimum: %.2f GB\n",
                mem / 1e9, maxMem / 1e9, minMem / 1e9);

    auto interpolation00 = interpolation.GetMatrix(0, 0);
    auto interpolation01 = interpolation.GetMatrix(0, 1);
    auto interpolation10 = interpolation.GetMatrix(1, 0);
    auto interpolation11 = interpolation.GetMatrix(1, 1);

    interpolation00->Resize(fieldDof * numNewLocalParticle,
                            fieldDof * numOldLocalParticle);
    interpolation01->Resize(fieldDof * numNewLocalParticle,
                            numLocalRigidBody * rigidBodyDof);
    interpolation10->Resize(numLocalRigidBody * rigidBodyDof,
                            fieldDof * numOldLocalParticle);
    interpolation11->Resize(numLocalRigidBody * rigidBodyDof,
                            numLocalRigidBody * rigidBodyDof);

    // compute matrix graph
    vector<PetscInt> index;
    for (int i = 0; i < numNewLocalParticle; i++) {
      int particleLocalIndex = local_idx[i];
      if (new_added[i] < 0) {
        // velocity interpolation
        index.resize(old_to_new_neighbor_lists_host(new_actual_index[i], 0) *
                     velocityDof);
        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          for (int k = 0; k < velocityDof; k++) {
            index[j * velocityDof + k] =
                fieldDof * old_source_index[old_to_new_neighbor_lists_host(
                               new_actual_index[i], j + 1)] +
                k;
          }
        }

        for (int k = 0; k < velocityDof; k++)
          interpolation00->SetColIndex(fieldDof * particleLocalIndex + k,
                                       index);

        // pressure interpolation
        index.resize(old_to_new_neighbor_lists_host(new_actual_index[i], 0));
        for (int j = 0;
             j < old_to_new_neighbor_lists_host(new_actual_index[i], 0); j++) {
          index[j] = fieldDof * old_source_index[old_to_new_neighbor_lists_host(
                                    new_actual_index[i], j + 1)] +
                     velocityDof;
        }
        interpolation00->SetColIndex(
            fieldDof * particleLocalIndex + velocityDof, index);
      } else {
        index.resize(1);
        for (int j = 0; j < fieldDof; j++) {
          index[0] = fieldDof * new_added[i] + j;
          interpolation00->SetColIndex(fieldDof * particleLocalIndex + j,
                                       index);
        }
      }
    }

    // rigid body
    index.resize(1);
    for (int i = rigidBodyStartIndex; i < rigidBodyEndIndex; i++) {
      for (int j = 0; j < rigidBodyDof; j++) {
        index[0] = i * rigidBodyDof + j;
        interpolation11->SetColIndex(
            (i - rigidBodyStartIndex) * rigidBodyDof + j, index);
      }
    }

    interpolation.GraphAssemble();

    PetscMemoryGetCurrentUsage(&mem);
    MPI_Allreduce(&mem, &maxMem, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&mem, &minMem, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD,
                "Current memory usage %.2f GB, maximum memory usage: %.2f GB, "
                "minimum: %.2f GB\n",
                mem / 1e9, maxMem / 1e9, minMem / 1e9);

    const unsigned int batchSize = (dimension_ == 2) ? 1000 : 100;
    const unsigned int numBatch =
        numNewLocalParticle / batchSize +
        ((numNewLocalParticle % batchSize > 0) ? 1 : 0);

    for (unsigned int batch = 0; batch < numBatch; batch++) {
      const unsigned int startParticle = batch * batchSize;
      const unsigned int endParticle =
          std::min((batch + 1) * batchSize, (unsigned int)numNewLocalParticle);

      unsigned int numBatchTargetParticle = 0;
      for (unsigned int i = startParticle; i < endParticle; i++) {
        if (new_added[i] < 0)
          numBatchTargetParticle++;
      }

      Kokkos::View<int **, Kokkos::DefaultExecutionSpace>
          batchNeighborListsDevice("batch neighbor lists",
                                   numBatchTargetParticle,
                                   old_to_new_neighbor_lists_device.extent(1));
      Kokkos::View<int **>::HostMirror batchNeighborListsHost =
          Kokkos::create_mirror_view(batchNeighborListsDevice);

      Kokkos::View<double *, Kokkos::DefaultExecutionSpace> batchEpsilonDevice(
          "batch epsilon", numBatchTargetParticle);
      Kokkos::View<double *>::HostMirror batchEpsilonHost =
          Kokkos::create_mirror_view(batchEpsilonDevice);

      Kokkos::View<double **, Kokkos::DefaultExecutionSpace> batchCoordsDevice(
          "batch coords", numBatchTargetParticle, 3);
      Kokkos::View<double **>::HostMirror batchCoordsHost =
          Kokkos::create_mirror_view(batchCoordsDevice);

      counter = 0;
      for (unsigned int i = startParticle; i < endParticle; i++) {
        if (new_added[i] < 0) {
          for (int j = 0; j < dimension_; j++) {
            batchCoordsHost(counter, j) = coord[i][j];
          }
          batchNeighborListsHost(counter, 0) =
              old_to_new_neighbor_lists_host(new_actual_index[i], 0);
          for (unsigned int j = 0; j < batchNeighborListsHost(counter, 0);
               j++) {
            batchNeighborListsHost(counter, j + 1) =
                old_to_new_neighbor_lists_host(new_actual_index[i], j + 1);
          }

          batchEpsilonHost(counter) = old_epsilon_host(new_actual_index[i]);

          counter++;
        }
      }

      Kokkos::deep_copy(batchEpsilonDevice, batchEpsilonHost);
      Kokkos::deep_copy(batchNeighborListsDevice, batchNeighborListsHost);
      Kokkos::deep_copy(batchCoordsDevice, batchCoordsHost);

      // velocity block
      if (numBatchTargetParticle != 0) {
        GMLS velocityBasis(DivergenceFreeVectorTaylorPolynomial,
                           VectorPointSample, polyOrder, dimension_, "LU",
                           "STANDARD");

        // old to new velocity field transition
        velocityBasis.setProblemData(batchNeighborListsDevice,
                                     old_source_coords_device,
                                     batchCoordsDevice, batchEpsilonDevice);

        velocityBasis.addTargets(VectorPointEvaluation);

        velocityBasis.setWeightingType(WeightingFunctionType::Power);
        velocityBasis.setWeightingParameter(2 * polyOrder);

        velocityBasis.generateAlphas(1, false);

        auto velocitySolutionSet = velocityBasis.getSolutionSetHost();
        auto velocityAlphas = velocitySolutionSet->getAlphas();

        // compute interpolation matrix entity
        vector<int> velocityAlphasIndex(pow(dimension_, 2));
        for (int axes1 = 0; axes1 < dimension_; axes1++)
          for (int axes2 = 0; axes2 < dimension_; axes2++)
            velocityAlphasIndex[axes1 * dimension_ + axes2] =
                velocitySolutionSet->getAlphaColumnOffset(VectorPointEvaluation,
                                                          axes1, 0, axes2, 0);

        counter = 0;
        for (unsigned int i = startParticle; i < endParticle; i++) {
          int particleLocalIndex = local_idx[i];
          if (new_added[i] < 0) {
            for (int j = 0; j < batchNeighborListsHost(counter, 0); j++) {
              for (int axes1 = 0; axes1 < dimension_; axes1++)
                for (int axes2 = 0; axes2 < dimension_; axes2++) {
                  auto alpha_index = velocitySolutionSet->getAlphaIndex(
                      counter, velocityAlphasIndex[axes1 * dimension_ + axes2]);
                  int neighbor_index =
                      old_source_index[batchNeighborListsHost(counter, j + 1)];
                  interpolation00->Increment(fieldDof * particleLocalIndex +
                                                 axes1,
                                             fieldDof * neighbor_index + axes2,
                                             velocityAlphas(alpha_index + j));
                }
            }

            counter++;
          }
        }
      }

      // pressure block
      if (numBatchTargetParticle != 0) {
        GMLS pressureBasis(ScalarTaylorPolynomial, PointSample, polyOrder,
                           dimension_, "LU", "STANDARD");

        // old to new pressure field transition
        pressureBasis.setProblemData(batchNeighborListsDevice,
                                     old_source_coords_device,
                                     batchCoordsDevice, batchEpsilonDevice);

        pressureBasis.addTargets(ScalarPointEvaluation);

        pressureBasis.setWeightingType(WeightingFunctionType::Power);
        pressureBasis.setWeightingParameter(2 * polyOrder);

        pressureBasis.generateAlphas(1, false);

        auto pressureSolution = pressureBasis.getSolutionSetHost();
        auto pressureAlphas = pressureSolution->getAlphas();

        const auto pressureAlphasIndex = pressureSolution->getAlphaColumnOffset(
            ScalarPointEvaluation, 0, 0, 0, 0);

        counter = 0;
        for (unsigned int i = startParticle; i < endParticle; i++) {
          int particleLocalIndex = local_idx[i];
          if (new_added[i] < 0) {
            for (int j = 0; j < batchNeighborListsHost(counter, 0); j++) {
              auto alpha_index =
                  pressureSolution->getAlphaIndex(counter, pressureAlphasIndex);
              int neighbor_index =
                  old_source_index[batchNeighborListsHost(counter, j + 1)];
              interpolation00->Increment(
                  fieldDof * particleLocalIndex + velocityDof,
                  fieldDof * neighbor_index + velocityDof,
                  pressureAlphas(alpha_index + j));
            }

            counter++;
          }
        }
      }

      {
        for (unsigned int i = startParticle; i < endParticle; i++) {
          int particleLocalIndex = local_idx[i];
          if (new_added[i] >= 0) {
            for (int j = 0; j < fieldDof; j++) {
              interpolation00->Increment(fieldDof * particleLocalIndex + j,
                                         fieldDof * new_added[i] + j, 1.0);
            }
          }
        }
      }
    }

    // rigid body
    for (int i = rigidBodyStartIndex; i < rigidBodyEndIndex; i++)
      for (int j = 0; j < rigidBodyDof; j++)
        interpolation11->Increment((i - rigidBodyStartIndex) * rigidBodyDof + j,
                                   i * rigidBodyDof + j, 1.0);

    interpolation.Assemble();
  }

  timer2 = MPI_Wtime();
  PetscPrintf(PETSC_COMM_WORLD, "Interpolation matrix building duration: %fs\n",
              timer2 - timer1);

  {
    auto &coord = *(geoMgr_->get_current_work_particle_coord());

    auto &source_coord = *(geoMgr_->get_llcl_particle_coord());
    auto &source_index = *(geoMgr_->get_llcl_particle_index());
    auto &source_particle_type = *(geoMgr_->get_llcl_particle_type());

    auto &old_coord = *(geoMgr_->get_last_work_particle_coord());
    auto &old_index = *(geoMgr_->get_last_work_particle_index());
    auto &old_local_index = *(geoMgr_->get_last_work_particle_local_index());
    auto &old_particle_type = *(geoMgr_->get_last_work_particle_type());
    auto &old_spacing = *(geoMgr_->get_last_work_particle_spacing());

    unsigned int numOldLocalParticle = old_coord.size();
    unsigned int numOldGlobalParticle;

    MPI_Allreduce(&numOldLocalParticle, &numOldGlobalParticle, 1, MPI_UNSIGNED,
                  MPI_SUM, MPI_COMM_WORLD);

    unsigned int numNewLocalParticle = coord.size();
    unsigned int numNewGlobalParticleNum;

    MPI_Allreduce(&numNewLocalParticle, &numNewGlobalParticleNum, 1,
                  MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

    auto restriction00 = restriction.GetMatrix(0, 0);
    auto restriction01 = restriction.GetMatrix(0, 1);
    auto restriction10 = restriction.GetMatrix(1, 0);
    auto restriction11 = restriction.GetMatrix(1, 1);

    restriction00->Resize(fieldDof * numOldLocalParticle,
                          fieldDof * numNewLocalParticle);
    restriction01->Resize(fieldDof * numOldLocalParticle,
                          numLocalRigidBody * rigidBodyDof);
    restriction10->Resize(numLocalRigidBody * rigidBodyDof,
                          fieldDof * numNewLocalParticle);
    restriction11->Resize(numLocalRigidBody * rigidBodyDof,
                          numLocalRigidBody * rigidBodyDof);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> source_coords_device(
        "old source coordinates", source_coord.size(), 3);
    Kokkos::View<double **>::HostMirror source_coords_host =
        Kokkos::create_mirror_view(source_coords_device);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace> target_coords_device(
        "new target coordinates", old_coord.size(), 3);
    Kokkos::View<double **>::HostMirror target_coords_host =
        Kokkos::create_mirror_view(target_coords_device);

    // copy old source coords
    for (int i = 0; i < source_coord.size(); i++) {
      for (int j = 0; j < dimension_; j++)
        source_coords_host(i, j) = source_coord[i][j];
    }

    // copy new target coords
    for (int i = 0; i < old_coord.size(); i++) {
      for (int j = 0; j < dimension_; j++) {
        target_coords_host(i, j) = old_coord[i][j];
      }
    }

    Kokkos::deep_copy(target_coords_device, target_coords_host);
    Kokkos::deep_copy(source_coords_device, source_coords_host);

    auto point_search(CreatePointCloudSearch(source_coords_host, dimension_));

    int estimated_num_neighbor_max = pow(2, dimension_) + 1;

    Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighbor_lists_device(
        "old to new neighbor lists", old_coord.size(),
        estimated_num_neighbor_max);
    Kokkos::View<int **>::HostMirror neighbor_lists_host =
        Kokkos::create_mirror_view(neighbor_lists_device);

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilon_device(
        "h supports", old_coord.size());
    Kokkos::View<double *>::HostMirror epsilon_host =
        Kokkos::create_mirror_view(epsilon_device);

    for (int i = 0; i < old_coord.size(); i++) {
      epsilon_host(i) = 0.25 * sqrt(dimension_) * old_spacing[i] + 1e-15;
    }

    int actual_neighbor = point_search.generate2DNeighborListsFromRadiusSearch(
                              true, target_coords_host, neighbor_lists_host,
                              epsilon_host, 0.0, 0.0) +
                          1;

    if (actual_neighbor > neighbor_lists_host.extent(1)) {
      Kokkos::resize(neighbor_lists_device, old_coord.size(), actual_neighbor);
      neighbor_lists_host = Kokkos::create_mirror_view(neighbor_lists_device);
    }
    point_search.generate2DNeighborListsFromRadiusSearch(
        false, target_coords_host, neighbor_lists_host, epsilon_host, 0.0, 0.0);

    for (int i = 0; i < old_coord.size(); i++) {
      bool is_boundary = (old_particle_type[i] == 0) ? false : true;
      vector<int> index;
      int corresponding_index = -1;
      for (int j = 0; j < neighbor_lists_host(i, 0); j++) {
        int neighbor_index = neighbor_lists_host(i, j + 1);
        bool is_neighbor_boundary =
            (source_particle_type[neighbor_index] == 0) ? false : true;
        if (is_boundary == is_neighbor_boundary)
          index.push_back(neighbor_index);
        Vec3 dX = source_coord[neighbor_index] - old_coord[i];
        if (dX.mag() < 1e-15) {
          corresponding_index = neighbor_index;
        }
      }
      if (corresponding_index == -1) {
        neighbor_lists_host(i, 0) = index.size();
        for (int j = 0; j < index.size(); j++) {
          neighbor_lists_host(i, j + 1) = index[j];
        }
      } else {
        neighbor_lists_host(i, 0) = 1;
        neighbor_lists_host(i, 1) = corresponding_index;
      }
    }

    Kokkos::deep_copy(neighbor_lists_device, neighbor_lists_host);
    Kokkos::deep_copy(epsilon_device, epsilon_host);

    // compute restriction matrix graph
    vector<PetscInt> index;
    int min_neighbor = 1000, max_neighbor = 0;
    for (int i = 0; i < numOldLocalParticle; i++) {
      int particleLocalIndex = old_local_index[i];
      index.resize(neighbor_lists_host(i, 0));
      if (min_neighbor > neighbor_lists_host(i, 0))
        min_neighbor = neighbor_lists_host(i, 0);
      if (max_neighbor < neighbor_lists_host(i, 0))
        max_neighbor = neighbor_lists_host(i, 0);
      for (int j = 0; j < fieldDof; j++) {
        for (int k = 0; k < neighbor_lists_host(i, 0); k++) {
          int neighbor_index = neighbor_lists_host(i, k + 1);
          index[k] = source_index[neighbor_index] * fieldDof + j;
        }

        restriction00->SetColIndex(fieldDof * particleLocalIndex + j, index);
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &min_neighbor, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_neighbor, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "min neighbor: %d, max neighbor: %d\n",
                min_neighbor, max_neighbor);

    // rigid body
    index.resize(1);
    for (int i = rigidBodyStartIndex; i < rigidBodyEndIndex; i++) {
      for (int j = 0; j < rigidBodyDof; j++) {
        index[0] = i * rigidBodyDof + j;
        restriction11->SetColIndex((i - rigidBodyStartIndex) * rigidBodyDof + j,
                                   index);
      }
    }

    restriction.GraphAssemble();

    for (int i = 0; i < numOldLocalParticle; i++) {
      int particleLocalIndex = old_local_index[i];
      for (int j = 0; j < fieldDof; j++) {
        for (int k = 0; k < neighbor_lists_host(i, 0); k++) {
          int neighbor_index = neighbor_lists_host(i, k + 1);
          restriction00->Increment(fieldDof * particleLocalIndex + j,
                                   source_index[neighbor_index] * fieldDof + j,
                                   1.0 / neighbor_lists_host(i, 0));
        }
      }
    }

    // rigid body
    for (int i = rigidBodyStartIndex; i < rigidBodyEndIndex; i++)
      for (int j = 0; j < rigidBodyDof; j++)
        restriction11->Increment((i - rigidBodyStartIndex) * rigidBodyDof + j,
                                 i * rigidBodyDof + j, 1.0);

    restriction.Assemble();
  }
}

void StokesMultilevelPreconditioning::InitialGuess(
    std::vector<double> &initial_guess, std::vector<Vec3> &velocity,
    std::vector<double> &pressure) {
  auto &local_idx = *(geoMgr_->get_last_work_particle_local_index());

  PetscNestedMatrix &interpolation =
      *(interpolationList_[currentRefinementLevel_ - 1]);
  Vec x1, x2;
  MatCreateVecs(interpolation.GetMatrix(0, 0)->GetReference(), &x2, &x1);

  const int numOldLocalParticle = pressure.size();

  const int fieldDof = dimension_ + 1;
  const int velocityDof = dimension_;
  const int rigidBodyDof = (dimension_ == 3) ? 6 : 3;

  PetscReal *a;
  VecGetArray(x2, &a);

  for (int i = 0; i < numOldLocalParticle; i++) {
    int particleLocalIndex = local_idx[i];
    for (int j = 0; j < velocityDof; j++) {
      a[particleLocalIndex * fieldDof + j] = velocity[i][j];
    }
    a[particleLocalIndex * fieldDof + velocityDof] = pressure[i];
  }

  VecRestoreArray(x2, &a);

  MatMult(interpolation.GetMatrix(0, 0)->GetReference(), x2, x1);

  VecGetArray(x1, &a);
  for (int i = 0; i < initial_guess.size(); i++) {
    initial_guess[i] = a[i];
  }
  VecRestoreArray(x1, &a);

  auto &coord = *(geoMgr_->get_current_work_particle_coord());
  int numLocalParticle = coord.size();
  int numGlobalParticle;
  MPI_Allreduce(&numLocalParticle, &numGlobalParticle, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  double pressure_sum = 0.0;
  for (int i = 0; i < numLocalParticle; i++) {
    pressure_sum += initial_guess[i * fieldDof + velocityDof];
  }

  MPI_Allreduce(MPI_IN_PLACE, &pressure_sum, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  double average_pressure = pressure_sum / numGlobalParticle;
  for (int i = 0; i < numLocalParticle; i++) {
    initial_guess[i * fieldDof + velocityDof] -= average_pressure;
  }

  VecDestroy(&x1);
  VecDestroy(&x2);
}

int StokesMultilevelPreconditioning::Solve(std::vector<double> &rhs0,
                                           std::vector<double> &x0,
                                           std::vector<double> &rhs1,
                                           std::vector<double> &x1) {
  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "\nstart of linear system solving setup\n");

  int hasRigidBody = rhs1.size();
  MPI_Allreduce(MPI_IN_PLACE, &hasRigidBody, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);

  int refinementStep = linearSystemList_.size() - 1;

  int fieldDof = dimension_ + 1;
  int velocityDof = dimension_;
  int rigidBodyDof = (dimension_ == 3) ? 6 : 3;

  Mat &matA = linearSystemList_[refinementStep]->GetReference();

  fieldRelaxationList_.push_back(make_shared<PetscKsp>());
  neighborRelaxationList_.push_back(make_shared<PetscKsp>());
  wholeRelaxationList_.push_back(make_shared<PetscKsp>());
  wholePcList_.push_back(make_shared<StokesSchurComplementPreconditioning>());

  // setup preconditioner for base level
  if (refinementStep == 0) {
    PC pcFieldBase;

    Mat &ff =
        linearSystemList_[refinementStep]->GetMatrix(0, 0)->GetReference();
    Mat &ffShell =
        linearSystemList_[refinementStep]->GetFieldFieldShellMatrix();

    KSPCreate(PETSC_COMM_WORLD, &fieldRelaxationList_[0]->GetReference());
    KSPSetOperators(fieldRelaxationList_[0]->GetReference(), ff, ff);
    KSPSetType(fieldRelaxationList_[0]->GetReference(), KSPRICHARDSON);
    KSPSetTolerances(fieldRelaxationList_[0]->GetReference(), 1e-2, 1e-50, 1e50,
                     10);
    KSPSetResidualHistory(fieldRelaxationList_[0]->GetReference(), NULL, 500,
                          PETSC_TRUE);

    KSPGetPC(fieldRelaxationList_[0]->GetReference(), &pcFieldBase);
    PCSetType(pcFieldBase, PCSOR);
    PCSetFromOptions(pcFieldBase);
    KSPSetUp(fieldRelaxationList_[0]->GetReference());

    if (hasRigidBody != 0) {
      Mat &nn = linearSystemList_[refinementStep]->GetNeighborNeighborMatrix();

      KSPCreate(MPI_COMM_WORLD, &neighborRelaxationList_[0]->GetReference());

      KSPSetType(neighborRelaxationList_[0]->GetReference(), KSPGMRES);
      KSPGMRESSetRestart(neighborRelaxationList_[0]->GetReference(), 100);
      KSPSetTolerances(neighborRelaxationList_[0]->GetReference(), 1e-3, 1e-50,
                       1e50, 500);
      KSPSetOperators(neighborRelaxationList_[0]->GetReference(), nn, nn);

      PC pcNeighborBase;
      KSPGetPC(neighborRelaxationList_[0]->GetReference(), &pcNeighborBase);
      PCSetType(pcNeighborBase, PCFIELDSPLIT);

      IS isg[2];
      MatNestGetISs(nn, isg, PETSC_NULL);

      PCFieldSplitSetIS(pcNeighborBase, "0", isg[0]);
      PCFieldSplitSetIS(pcNeighborBase, "1", isg[1]);

      Mat &S = linearSystemList_[refinementStep]->GetNeighborSchurMatrix();
      Mat &sub00 =
          linearSystemList_[refinementStep]->GetNeighborNeighborSubMatrix(0, 0);

      PCFieldSplitSetSchurPre(pcNeighborBase, PC_FIELDSPLIT_SCHUR_PRE_USER, S);
      PCSetFromOptions(pcNeighborBase);
      PCSetUp(pcNeighborBase);

      KSP *fieldsplitSubKsp;
      PetscInt n;
      PCFieldSplitGetSubKSP(pcNeighborBase, &n, &fieldsplitSubKsp);
      KSPSetOperators(fieldsplitSubKsp[1], S, S);
      KSPSetOperators(fieldsplitSubKsp[0], sub00, sub00);
      KSPSetUp(fieldsplitSubKsp[0]);
      KSPSetUp(fieldsplitSubKsp[1]);
      PetscFree(fieldsplitSubKsp);

      // KSPCreate(MPI_COMM_WORLD, &wholeRelaxationList_[0]->GetReference());

      // KSPSetType(wholeRelaxationList_[0]->GetReference(), KSPGMRES);
      // KSPGMRESSetRestart(wholeRelaxationList_[0]->GetReference(), 100);
      // KSPSetTolerances(wholeRelaxationList_[0]->GetReference(), 1e-2, 1e-50,
      //                  1e50, 500);
      // KSPSetOperators(wholeRelaxationList_[0]->GetReference(), matA, matA);

      // PC pcWholeBase;
      // KSPGetPC(wholeRelaxationList_[0]->GetReference(), &pcWholeBase);
      // PCSetType(pcWholeBase, PCSHELL);

      // wholePcList_[0]->AddLinearSystem(linearSystemList_[0]);

      // PCShellSetApply(pcWholeBase, StokesSchurComplementIterationWrapper);
      // PCShellSetContext(pcWholeBase, wholePcList_[0].get());

      // KSPSetUp(wholeRelaxationList_[0]->GetReference());
    }
  } else {
    Mat &ff =
        linearSystemList_[refinementStep]->GetMatrix(0, 0)->GetReference();
    Mat &ffShell =
        linearSystemList_[refinementStep]->GetFieldFieldShellMatrix();

    // setup relaxation on field for current level
    KSPCreate(MPI_COMM_WORLD,
              fieldRelaxationList_[refinementStep]->GetPointer());

    KSPSetType(fieldRelaxationList_[refinementStep]->GetReference(),
               KSPRICHARDSON);
    KSPSetOperators(fieldRelaxationList_[refinementStep]->GetReference(), ff,
                    ffShell);
    KSPSetTolerances(fieldRelaxationList_[refinementStep]->GetReference(), 5e-1,
                     1e-50, 1e10, 1);

    PC field_relaxation_pc;
    KSPGetPC(fieldRelaxationList_[refinementStep]->GetReference(),
             &field_relaxation_pc);
    PCSetType(field_relaxation_pc, PCSOR);
    PCSetFromOptions(field_relaxation_pc);
    PCSetUp(field_relaxation_pc);

    KSPSetUp(fieldRelaxationList_[refinementStep]->GetReference());

    if (hasRigidBody != 0) {
      Mat &nn = linearSystemList_[refinementStep]->GetNeighborNeighborMatrix();
      // setup relaxation on neighbor for current level
      KSPCreate(MPI_COMM_WORLD,
                neighborRelaxationList_[refinementStep]->GetPointer());

      KSPSetType(neighborRelaxationList_[refinementStep]->GetReference(),
                 KSPPREONLY);
      // KSPGMRESSetRestart(
      //     neighborRelaxationList_[refinementStep]->GetReference(), 100);
      // KSPSetTolerances(neighborRelaxationList_[refinementStep]->GetReference(),
      //                  1e-1, 1e-50, 1e10, 500);
      KSPSetOperators(neighborRelaxationList_[refinementStep]->GetReference(),
                      nn, nn);

      PC neighbor_relaxation_pc;
      KSPGetPC(neighborRelaxationList_[refinementStep]->GetReference(),
               &neighbor_relaxation_pc);
      PCSetType(neighbor_relaxation_pc, PCFIELDSPLIT);

      IS isg[2];
      MatNestGetISs(nn, isg, PETSC_NULL);

      PCFieldSplitSetIS(neighbor_relaxation_pc, "0", isg[0]);
      PCFieldSplitSetIS(neighbor_relaxation_pc, "1", isg[1]);

      Mat &S = linearSystemList_[refinementStep]->GetNeighborSchurMatrix();
      Mat &sub00 =
          linearSystemList_[refinementStep]->GetNeighborNeighborSubMatrix(0, 0);

      PCFieldSplitSetSchurPre(neighbor_relaxation_pc,
                              PC_FIELDSPLIT_SCHUR_PRE_USER, S);
      PCSetFromOptions(neighbor_relaxation_pc);
      PCSetUp(neighbor_relaxation_pc);

      KSP *fieldsplitSubKsp;
      PetscInt n;
      PCFieldSplitGetSubKSP(neighbor_relaxation_pc, &n, &fieldsplitSubKsp);
      KSPSetOperators(fieldsplitSubKsp[1], S, S);
      KSPSetOperators(fieldsplitSubKsp[0], sub00, sub00);
      KSPSetFromOptions(fieldsplitSubKsp[0]);
      KSPSetUp(fieldsplitSubKsp[0]);
      KSPSetUp(fieldsplitSubKsp[1]);
      PetscFree(fieldsplitSubKsp);

      // KSPCreate(MPI_COMM_WORLD,
      //           &wholeRelaxationList_[currentRefinementLevel_]->GetReference());

      // KSPSetType(wholeRelaxationList_[currentRefinementLevel_]->GetReference(),
      //            KSPGMRES);
      // KSPGMRESSetRestart(
      //     wholeRelaxationList_[currentRefinementLevel_]->GetReference(),
      //     100);
      // KSPSetTolerances(
      //     wholeRelaxationList_[currentRefinementLevel_]->GetReference(),
      //     1e-1, 1e-50, 1e50, 500);
      // KSPSetOperators(
      //     wholeRelaxationList_[currentRefinementLevel_]->GetReference(),
      //     matA, matA);

      // PC pcWholeBase;
      // KSPGetPC(wholeRelaxationList_[currentRefinementLevel_]->GetReference(),
      //          &pcWholeBase);
      // PCSetType(pcWholeBase, PCSHELL);

      // wholePcList_[currentRefinementLevel_]->AddLinearSystem(
      //     linearSystemList_[currentRefinementLevel_]);

      // PCShellSetApply(pcWholeBase, StokesSchurComplementIterationWrapper);
      // PCShellSetContext(pcWholeBase,
      //                   wholePcList_[currentRefinementLevel_].get());

      // KSPSetUp(wholeRelaxationList_[currentRefinementLevel_]->GetReference());
    }
  }

  PetscVector rhs, x;
  rhs.Create(rhs0.size() + rhs1.size());
  x.Create(rhs0.size() + rhs1.size());

  PetscReal *a;
  VecGetArray(rhs.GetReference(), &a);
  for (unsigned int i = 0; i < rhs0.size(); i++)
    a[i] = rhs0[i];
  for (unsigned int i = 0; i < rhs1.size(); i++)
    a[rhs0.size() + i] = rhs1[i];
  VecRestoreArray(rhs.GetReference(), &a);

  VecGetArray(x.GetReference(), &a);
  for (unsigned int i = 0; i < rhs0.size(); i++)
    a[i] = x0[i];
  for (unsigned int i = 0; i < rhs1.size(); i++)
    a[rhs0.size() + i] = x1[i];
  VecRestoreArray(x.GetReference(), &a);

  xList_.push_back(std::make_shared<PetscVector>());
  yList_.push_back(std::make_shared<PetscVector>());
  bList_.push_back(std::make_shared<PetscVector>());
  rList_.push_back(std::make_shared<PetscVector>());

  xFieldList_.push_back(std::make_shared<PetscVector>());
  yFieldList_.push_back(std::make_shared<PetscVector>());
  bFieldList_.push_back(std::make_shared<PetscVector>());
  rFieldList_.push_back(std::make_shared<PetscVector>());

  xList_[refinementStep]->Create(rhs0.size() + rhs1.size());
  yList_[refinementStep]->Create(rhs0.size() + rhs1.size());
  bList_[refinementStep]->Create(rhs0.size() + rhs1.size());
  rList_[refinementStep]->Create(rhs0.size() + rhs1.size());

  xFieldList_.push_back(std::make_shared<PetscVector>());
  yFieldList_.push_back(std::make_shared<PetscVector>());
  bFieldList_.push_back(std::make_shared<PetscVector>());
  rFieldList_.push_back(std::make_shared<PetscVector>());

  xFieldList_[refinementStep]->Create(rhs0.size());
  yFieldList_[refinementStep]->Create(rhs0.size());
  bFieldList_[refinementStep]->Create(rhs0.size());
  rFieldList_[refinementStep]->Create(rhs0.size());

  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);

  PC pc;

  KSPGetPC(ksp, &pc);
  KSPSetOperators(ksp, matA, matA);
  KSPSetFromOptions(ksp);

  PCSetType(pc, PCSHELL);

  PCShellSetApply(pc, StokesMultilevelIterationWrapper);
  PCShellSetContext(pc, this);

  KSPSetUp(ksp);

  fieldRelaxationDuration_.resize(linearSystemList_.size());
  neighborRelaxationDuration_.resize(linearSystemList_.size());

  for (unsigned int i = 0; i < linearSystemList_.size(); i++) {
    fieldRelaxationDuration_[i] = 0;
    neighborRelaxationDuration_[i] = 0;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "start of stokes_multilevel preconditioner setup\n");

  PetscPrintf(PETSC_COMM_WORLD, "final solving of linear system\n");
  PetscReal residualNorm, rhsNorm;

  VecNorm(rhs.GetReference(), NORM_2, &rhsNorm);

  Vec residual;
  VecDuplicate(rhs.GetReference(), &residual);
  MatMult(matA, x.GetReference(), residual);
  VecAXPY(residual, -1.0, rhs.GetReference());
  VecNorm(residual, NORM_2, &residualNorm);
  PetscPrintf(PETSC_COMM_WORLD, "relative residual norm: %f\n",
              residualNorm / rhsNorm);

  KSPSolve(ksp, rhs.GetReference(), x.GetReference());

  MatMult(matA, x.GetReference(), residual);
  VecAXPY(residual, -1.0, rhs.GetReference());
  VecNorm(residual, NORM_2, &residualNorm);
  PetscPrintf(PETSC_COMM_WORLD, "relative residual norm: %f\n",
              residualNorm / rhsNorm);

  VecDestroy(&residual);

  KSPDestroy(&ksp);

  VecGetArray(x.GetReference(), &a);
  for (unsigned int i = 0; i < x0.size(); i++)
    x0[i] = a[i];
  for (unsigned int i = 0; i < x1.size(); i++)
    x1[i] = a[rhs0.size() + i];
  VecRestoreArray(x.GetReference(), &a);

  PetscLogDouble maxMem, mem;
  PetscMemoryGetCurrentUsage(&mem);
  MPI_Allreduce(&mem, &maxMem, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "Current memory usage %.2f GB, maximum memory usage: %.2f GB\n",
              mem / 1e9, maxMem / 1e9);

  for (unsigned int i = 0; i < fieldRelaxationDuration_.size(); i++) {
    PetscPrintf(MPI_COMM_WORLD,
                "Level: %d, field relaxation duration: %.4fs, neighbor "
                "relaxation duration: %.4fs\n",
                i, fieldRelaxationDuration_[i], neighborRelaxationDuration_[i]);
  }

  return 0;
}

void StokesMultilevelPreconditioning::Clear() {
  MPI_Barrier(MPI_COMM_WORLD);

  linearSystemList_.clear();
  interpolationList_.clear();
  restrictionList_.clear();

  xList_.clear();
  yList_.clear();
  bList_.clear();
  rList_.clear();

  xFieldList_.clear();
  yFieldList_.clear();
  bFieldList_.clear();
  rFieldList_.clear();

  fieldRelaxationList_.clear();
  neighborRelaxationList_.clear();
  wholeRelaxationList_.clear();
  wholePcList_.clear();

  numLocalParticleList_.clear();
  numGlobalParticleList_.clear();

  currentRefinementLevel_ = -1;
}

PetscErrorCode StokesMultilevelPreconditioning::MultilevelIteration(Vec x,
                                                                    Vec y) {
  double timer1, timer2;
  VecCopy(x, bList_[currentRefinementLevel_]->GetReference());

  for (unsigned int i = currentRefinementLevel_; i > 0; i--) {
    linearSystemList_[i]->ConstantVec(bList_[i]->GetReference());

    VecSet(xList_[i]->GetReference(), 0.0);

    linearSystemList_[i]->ForwardField(bList_[i]->GetReference(),
                                       bFieldList_[i]->GetReference());

    timer1 = MPI_Wtime();

    KSPSolve(fieldRelaxationList_[i]->GetReference(),
             bFieldList_[i]->GetReference(), xFieldList_[i]->GetReference());

    timer2 = MPI_Wtime();
    fieldRelaxationDuration_[i] += (timer2 - timer1);

    linearSystemList_[i]->ConstantVec(xFieldList_[i]->GetReference());

    linearSystemList_[i]->BackwardField(xList_[i]->GetReference(),
                                        xFieldList_[i]->GetReference());

    if (numRigidBody_ != 0) {
      MatMult(linearSystemList_[i]->GetReference(), xList_[i]->GetReference(),
              yList_[i]->GetReference());

      VecAYPX(yList_[i]->GetReference(), -1.0, bList_[i]->GetReference());

      linearSystemList_[i]->ForwardNeighbor(
          yList_[i]->GetReference(),
          linearSystemList_[i]->GetNeighborB()->GetReference());

      timer1 = MPI_Wtime();

      KSPSolve(neighborRelaxationList_[i]->GetReference(),
               linearSystemList_[i]->GetNeighborB()->GetReference(),
               linearSystemList_[i]->GetNeighborX()->GetReference());

      VecSet(yList_[i]->GetReference(), 0.0);
      linearSystemList_[i]->BackwardNeighbor(
          yList_[i]->GetReference(),
          linearSystemList_[i]->GetNeighborX()->GetReference());
      VecAXPY(xList_[i]->GetReference(), 1.0, yList_[i]->GetReference());
      linearSystemList_[i]->ConstantVec(xList_[i]->GetReference());

      timer2 = MPI_Wtime();
      neighborRelaxationDuration_[i] += (timer2 - timer1);

      // MatMult(linearSystemList_[i]->GetReference(),
      // xList_[i]->GetReference(),
      //         rList_[i]->GetReference());

      // linearSystemList_[i]->ConstantVec(rList_[i]->GetReference());

      // VecAYPX(rList_[i]->GetReference(), -1.0, bList_[i]->GetReference());

      // KSPSolve(wholeRelaxationList_[i]->GetReference(),
      //          rList_[i]->GetReference(), yList_[i]->GetReference());

      // VecAXPY(xList_[i]->GetReference(), 1.0, yList_[i]->GetReference());

      // linearSystemList_[i]->ConstantVec(xList_[i]->GetReference());
    }

    MatMult(linearSystemList_[i]->GetReference(), xList_[i]->GetReference(),
            rList_[i]->GetReference());
    VecAYPX(rList_[i]->GetReference(), -1.0, bList_[i]->GetReference());

    MatMult(restrictionList_[i - 1]->GetReference(), rList_[i]->GetReference(),
            bList_[i - 1]->GetReference());
  }

  // solve on base level
  timer1 = MPI_Wtime();

  linearSystemList_[0]->ConstantVec(bList_[0]->GetReference());

  VecSet(xList_[0]->GetReference(), 0.0);

  linearSystemList_[0]->ForwardField(bList_[0]->GetReference(),
                                     bFieldList_[0]->GetReference());

  KSPSolve(fieldRelaxationList_[0]->GetReference(),
           bFieldList_[0]->GetReference(), xFieldList_[0]->GetReference());

  linearSystemList_[0]->BackwardField(xList_[0]->GetReference(),
                                      xFieldList_[0]->GetReference());

  linearSystemList_[0]->ConstantVec(xList_[0]->GetReference());

  timer2 = MPI_Wtime();
  fieldRelaxationDuration_[0] += (timer2 - timer1);

  if (numRigidBody_ != 0) {
    MatMult(linearSystemList_[0]->GetReference(), xList_[0]->GetReference(),
            yList_[0]->GetReference());

    VecAYPX(yList_[0]->GetReference(), -1.0, bList_[0]->GetReference());

    linearSystemList_[0]->ForwardNeighbor(
        yList_[0]->GetReference(),
        linearSystemList_[0]->GetNeighborB()->GetReference());

    timer1 = MPI_Wtime();

    KSPSolve(neighborRelaxationList_[0]->GetReference(),
             linearSystemList_[0]->GetNeighborB()->GetReference(),
             linearSystemList_[0]->GetNeighborX()->GetReference());

    VecSet(yList_[0]->GetReference(), 0.0);
    linearSystemList_[0]->BackwardNeighbor(
        yList_[0]->GetReference(),
        linearSystemList_[0]->GetNeighborX()->GetReference());
    VecAXPY(xList_[0]->GetReference(), 1.0, yList_[0]->GetReference());

    linearSystemList_[0]->ConstantVec(xList_[0]->GetReference());

    timer2 = MPI_Wtime();
    neighborRelaxationDuration_[0] += (timer2 - timer1);

    // MatMult(linearSystemList_[0]->GetReference(), xList_[0]->GetReference(),
    //         rList_[0]->GetReference());

    // linearSystemList_[0]->ConstantVec(rList_[0]->GetReference());

    // VecAYPX(rList_[0]->GetReference(), -1.0, bList_[0]->GetReference());

    // KSPSolve(wholeRelaxationList_[0]->GetReference(),
    // rList_[0]->GetReference(),
    //          yList_[0]->GetReference());

    // VecAXPY(xList_[0]->GetReference(), 1.0, yList_[0]->GetReference());

    // linearSystemList_[0]->ConstantVec(xList_[0]->GetReference());
  }

  for (unsigned int i = 1; i <= currentRefinementLevel_; i++) {
    MatMult(interpolationList_[i - 1]->GetReference(),
            xList_[i - 1]->GetReference(), yList_[i]->GetReference());
    VecAXPY(xList_[i]->GetReference(), 1.0, yList_[i]->GetReference());
    linearSystemList_[i]->ConstantVec(xList_[i]->GetReference());

    timer1 = MPI_Wtime();

    MatMult(linearSystemList_[i]->GetReference(), xList_[i]->GetReference(),
            rList_[i]->GetReference());
    VecAYPX(rList_[i]->GetReference(), -1.0, bList_[i]->GetReference());

    linearSystemList_[i]->ForwardField(rList_[i]->GetReference(),
                                       rFieldList_[i]->GetReference());

    KSPSolve(fieldRelaxationList_[i]->GetReference(),
             rFieldList_[i]->GetReference(), yFieldList_[i]->GetReference());

    VecSet(yList_[i]->GetReference(), 0.0);
    linearSystemList_[i]->BackwardField(yList_[i]->GetReference(),
                                        yFieldList_[i]->GetReference());

    VecAXPY(xList_[i]->GetReference(), 1.0, yList_[i]->GetReference());
    linearSystemList_[i]->ConstantVec(xList_[i]->GetReference());

    timer2 = MPI_Wtime();
    fieldRelaxationDuration_[i] += (timer2 - timer1);

    if (numRigidBody_ != 0) {
      MatMult(linearSystemList_[i]->GetReference(), xList_[i]->GetReference(),
              yList_[i]->GetReference());

      VecAYPX(yList_[i]->GetReference(), -1.0, bList_[i]->GetReference());

      linearSystemList_[i]->ForwardNeighbor(
          yList_[i]->GetReference(),
          linearSystemList_[i]->GetNeighborB()->GetReference());

      timer1 = MPI_Wtime();

      KSPSolve(neighborRelaxationList_[i]->GetReference(),
               linearSystemList_[i]->GetNeighborB()->GetReference(),
               linearSystemList_[i]->GetNeighborX()->GetReference());

      VecSet(yList_[i]->GetReference(), 0.0);
      linearSystemList_[i]->BackwardNeighbor(
          yList_[i]->GetReference(),
          linearSystemList_[i]->GetNeighborX()->GetReference());
      VecAXPY(xList_[i]->GetReference(), 1.0, yList_[i]->GetReference());
      linearSystemList_[i]->ConstantVec(xList_[i]->GetReference());

      timer2 = MPI_Wtime();
      neighborRelaxationDuration_[i] += (timer2 - timer1);

      // MatMult(linearSystemList_[i]->GetReference(),
      // xList_[i]->GetReference(),
      //         rList_[i]->GetReference());

      // linearSystemList_[i]->ConstantVec(rList_[i]->GetReference());

      // VecAYPX(rList_[i]->GetReference(), -1.0, bList_[i]->GetReference());

      // KSPSolve(wholeRelaxationList_[i]->GetReference(),
      //          rList_[i]->GetReference(), yList_[i]->GetReference());

      // VecAXPY(xList_[i]->GetReference(), 1.0, yList_[i]->GetReference());

      // linearSystemList_[i]->ConstantVec(xList_[i]->GetReference());
    }
  }

  VecCopy(xList_[currentRefinementLevel_]->GetReference(), y);

  return 0;
}