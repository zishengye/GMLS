#include <Kokkos_CopyViews.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <execution>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>

#include "Equation/Equation.hpp"
#include "Equation/MultilevelPreconditioner.hpp"
#include "Geometry/DomainGeometry.hpp"
#include "Geometry/ParticleGeometry.hpp"
#include "LinearAlgebra/LinearAlgebra.hpp"
#include "LinearAlgebra/LinearSolverDescriptor.hpp"

Void Equation::Equation::AddLinearSystem(std::shared_ptr<DefaultMatrix> mat) {
  linearSystemsPtr_.push_back(mat);
  preconditionerPtr_->AddLinearSystem(mat);
}

Void Equation::Equation::InitLinearSystem() {
  BuildGhost();

  GlobalIndex globalParticleNum = particleMgr_.GetGlobalParticleNum();
  if (mpiRank_ == 0)
    printf("Start of preparing linear system with %lu particles\n",
           globalParticleNum);

  MPI_Barrier(MPI_COMM_WORLD);
}

Void Equation::Equation::ConstructLinearSystem() {}

Void Equation::Equation::ConstructRhs() {}

Void Equation::Equation::Clear() {
  linearSystemsPtr_.clear();
  preconditionerPtr_.reset();
  particleMgr_.Clear();
}

Void Equation::Equation::DiscretizeEquation() {
  this->InitLinearSystem();
  this->ConstructLinearSystem();
  this->ConstructRhs();
}

Void Equation::Equation::InitPreconditioner() {
  preconditionerPtr_->ConstructSmoother();
  preconditionerPtr_->ConstructInterpolation(particleMgr_);
  preconditionerPtr_->ConstructRestriction(particleMgr_);

  DefaultLinearAlgebraBackend::DefaultInteger localCol =
      linearSystemsPtr_[refinementIteration_]->GetLocalColSize();
  preconditionerPtr_->PrepareVectors(localCol);

  descriptor_.outerIteration = 1;
  descriptor_.spd = -1;
  descriptor_.maxIter = 500;
  descriptor_.relativeTol = 1e-6;
  descriptor_.setFromDatabase = true;

  descriptor_.preconditioningIteration =
      std::function<Void(DefaultVector &, DefaultVector &)>(
          [=](DefaultVector &x, DefaultVector &y) {
            preconditionerPtr_->ApplyPreconditioningIteration(x, y);
          });

  solver_.AddLinearSystem(linearSystemsPtr_[refinementIteration_], descriptor_);
}

Void Equation::Equation::SolveEquation() {
  preconditionerPtr_->ClearTimer();
  solver_.Solve(b_, x_);
}

Void Equation::Equation::CalculateError() {
  const LocalIndex localParticleNum = particleMgr_.GetLocalParticleNum();
  Kokkos::resize(error_, localParticleNum);
}

Void Equation::Equation::Mark() {
  if (mpiRank_ == 0)
    printf("Global error: %.6f, with tolerance: %.4f\n", globalNormalizedError_,
           errorTolerance_);

  Kokkos::resize(splitTag_, error_.extent(0));
  if (globalNormalizedError_ < errorTolerance_)
    return;

  std::vector<double> sortedError(error_.extent(0));
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             error_.extent(0)),
      [&](const int i) { sortedError[i] = pow(error_(i), 2); });
  Kokkos::fence();
  std::sort(std::execution::par_unseq, sortedError.begin(), sortedError.end());

  double maxError = sortedError[error_.extent(0) - 1];
  double minError = sortedError[0];

  MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &minError, 1, MPI_DOUBLE, MPI_MIN,
                MPI_COMM_WORLD);

  double currentErrorSplit = (maxError + minError) / 2.0;
  int iteCounter = 0;
  bool isSplit = false;
  if (markRatio_ != 1.0)
    while (!isSplit && iteCounter < 20) {
      iteCounter++;
      auto result = std::lower_bound(sortedError.begin(), sortedError.end(),
                                     currentErrorSplit);
      double localError = 0.0;
      double globalError;
      for (auto iter = result; iter != sortedError.end(); iter++) {
        localError += *iter;
      }
      MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      double nextError;
      if (result != sortedError.begin()) {
        nextError = *(result - 1);
      } else
        nextError = 0.0;
      MPI_Allreduce(MPI_IN_PLACE, &nextError, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);

      if (sqrt(globalError) <= markRatio_ * globalError_ &&
          sqrt(globalError + nextError) > markRatio_ * globalError_)
        isSplit = true;
      else if (sqrt(globalError) <= markRatio_ * globalError_) {
        maxError = currentErrorSplit;
        currentErrorSplit = (maxError + minError) / 2.0;
      } else {
        minError = currentErrorSplit;
        currentErrorSplit = (maxError + minError) / 2.0;
      }
    }
  else
    currentErrorSplit = 0.0;

  currentErrorSplit = sqrt(currentErrorSplit);

  // mark particles
  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                           0, error_.extent(0)),
                       [&](const int i) {
                         if (error_(i) >= currentErrorSplit)
                           splitTag_(i) = 1;
                         else
                           splitTag_(i) = 0;
                       });
  Kokkos::fence();

  std::size_t markedParticleNum = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             error_.extent(0)),
      [&](const int i, std::size_t &tMarkedParticleNum) {
        if (splitTag_(i) != 0)
          tMarkedParticleNum++;
      },
      Kokkos::Sum<std::size_t>(markedParticleNum));
  Kokkos::fence();
  MPI_Allreduce(MPI_IN_PLACE, &markedParticleNum, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  if (mpiRank_ == 0)
    printf("Marked %ld particles with mark ratio: %.2f\n", markedParticleNum,
           markRatio_);

  // ensure quasi-uniform
  auto &coords = particleMgr_.GetParticleCoords();
  auto &sourceCoords = hostGhostParticleCoords_;
  auto &particleType = particleMgr_.GetParticleType();
  auto &sourceParticleType = hostGhostParticleType_;

  auto &particleRefinementLevel = particleMgr_.GetParticleRefinementLevel();
  HostIndexVector ghostParticleRefinementLevel;
  ghost_.ApplyGhost(particleRefinementLevel, ghostParticleRefinementLevel);

  HostIndexVector ghostSplitTag;
  HostIntVector crossRefinementLevel, ghostCrossRefinementLevel;
  Kokkos::resize(crossRefinementLevel, error_.extent(0));
  int iterationFinished = 1;
  iteCounter = 0;
  while (iterationFinished != 0 && iteCounter < 20) {
    iteCounter++;

    ghost_.ApplyGhost(splitTag_, ghostSplitTag);
    unsigned int localChange = 0;

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                             0, crossRefinementLevel.extent(0)),
                         [&](const int i) { crossRefinementLevel(i) = -1; });
    Kokkos::fence();

    // ensure boundary particles at least have the same level of refinement
    // to their nearest interior particles after refinement
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, coords.extent(0)),
        [&](const int i, unsigned int &tLocalChange) {
          if (particleType(i) != 0 && splitTag_(i) == 0) {
            double minDistance = 1e9;
            int nearestIndex = 0;
            for (std::size_t j = 1; j < neighborLists_(i, 0); j++) {
              std::size_t neighborParticleIndex = neighborLists_(i, j + 1);
              if (sourceParticleType(neighborParticleIndex) == 0) {
                double x =
                    coords(i, 0) - sourceCoords(neighborParticleIndex, 0);
                double y =
                    coords(i, 1) - sourceCoords(neighborParticleIndex, 1);
                double z =
                    coords(i, 2) - sourceCoords(neighborParticleIndex, 2);
                double distance = sqrt(x * x + y * y + z * z);
                if (distance < minDistance) {
                  minDistance = distance;
                  nearestIndex = neighborParticleIndex;
                }
              }
            }

            if ((ghostSplitTag(nearestIndex) != 0) +
                    ghostParticleRefinementLevel(nearestIndex) >
                particleRefinementLevel(i)) {
              splitTag_(i) = 2;
              tLocalChange++;
            }
          }
        },
        Kokkos::Sum<unsigned int>(localChange));

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, crossRefinementLevel.extent(0)),
        [&](const int i) {
          std::size_t minRefinementLevel = 100;
          std::size_t maxRefinementLevel = 0;
          for (std::size_t j = 1; j < neighborLists_(i, 0); j++) {
            std::size_t neighborParticleIndex = neighborLists_(i, j + 1);

            std::size_t ghostRefinementLevel =
                ghostParticleRefinementLevel[neighborParticleIndex] +
                (ghostSplitTag[neighborParticleIndex] != 0);
            if (ghostRefinementLevel < minRefinementLevel)
              minRefinementLevel = ghostRefinementLevel;
            if (ghostRefinementLevel > maxRefinementLevel)
              maxRefinementLevel = ghostRefinementLevel;
          }

          if (maxRefinementLevel - minRefinementLevel > 1)
            crossRefinementLevel(i) =
                (splitTag_(i) != 0) + particleRefinementLevel(i);
          else
            crossRefinementLevel(i) = 0;
        });
    Kokkos::fence();

    ghost_.ApplyGhost(crossRefinementLevel, ghostCrossRefinementLevel);

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, crossRefinementLevel.extent(0)),
        [&](const int i, unsigned int &tLocalChange) {
          for (std::size_t j = 1; j < neighborLists_(i, 0); j++) {
            std::size_t neighborParticleIndex = neighborLists_(i, j + 1);

            if ((particleRefinementLevel(i) <
                 ghostCrossRefinementLevel(neighborParticleIndex)) &&
                splitTag_(i) == 0) {
              splitTag_(i) = 3;
              tLocalChange++;
            }
          }
        },
        Kokkos::Sum<unsigned int>(localChange));

    MPI_Allreduce(MPI_IN_PLACE, &localChange, 1, MPI_UNSIGNED, MPI_SUM,
                  MPI_COMM_WORLD);
    if (localChange == 0) {
      iterationFinished = 0;
    }
  }

  markedParticleNum = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             error_.extent(0)),
      [&](const int i, std::size_t &tMarkedParticleNum) {
        if (splitTag_(i) != 0)
          tMarkedParticleNum++;
      },
      Kokkos::Sum<std::size_t>(markedParticleNum));
  Kokkos::fence();
  MPI_Allreduce(MPI_IN_PLACE, &markedParticleNum, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  if (mpiRank_ == 0)
    printf("Marked %ld particles after smoothing particles\n",
           markedParticleNum);
}

Void Equation::Equation::BuildGhost() {
  auto particleCoords = particleMgr_.GetParticleCoords();
  auto particleSize = particleMgr_.GetParticleSize();
  auto particleIndex = particleMgr_.GetParticleIndex();
  auto particleType = particleMgr_.GetParticleType();
  ghost_.Init(particleCoords, particleSize, particleCoords, ghostMultiplier_,
              particleMgr_.GetDimension());
  ghost_.ApplyGhost(particleCoords, hostGhostParticleCoords_);
  ghost_.ApplyGhost(particleIndex, hostGhostParticleIndex_);
  ghost_.ApplyGhost(particleType, hostGhostParticleType_);
}

Void Equation::Equation::Output() {
  if (outputLevel_ == 0)
    return;

  std::string outputFileName =
      "vtk/AdaptiveStep" + std::to_string(refinementIteration_) + ".vtk";

  Output(outputFileName);
}

Void Equation::Equation::Output(String &outputFileName) {
  particleMgr_.Output(outputFileName, true);

  if (mpiRank_ == 0)
    printf("Start of writing adaptive step output file\n");

  std::ofstream vtkStream;
  // output number of neighbor
  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS nn int 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < neighborLists_.extent(0); i++) {
        int x = neighborLists_(i, 0);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(int));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // output split tag
  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS split int 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < splitTag_.extent(0); i++) {
        int x = splitTag_(i);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(int));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

Void Equation::Equation::ConstructNeighborLists(
    const Size satisfiedNumNeighbor) {
  auto &sourceCoords = hostGhostParticleCoords_;
  auto &coords = particleMgr_.GetParticleCoords();
  auto &spacing = particleMgr_.GetParticleSize();

  const Size localParticleNum = coords.extent(0);
  const Size globalParticleNum = particleMgr_.GetGlobalParticleNum();

  // construct neighbor lists
  Kokkos::resize(neighborLists_, localParticleNum, satisfiedNumNeighbor + 1);
  Kokkos::resize(epsilon_, localParticleNum);

  auto pointCloudSearch(Compadre::CreatePointCloudSearch(
      sourceCoords, particleMgr_.GetDimension()));

  // initialize epsilon
  pointCloudSearch.generate2DNeighborListsFromKNNSearch(
      true, coords, neighborLists_, epsilon_, satisfiedNumNeighbor, 1.0);

  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                           0, localParticleNum),
                       [&](const int i) {
                         int scaling =
                             floor(epsilon_(i) / spacing(i) * 1000 + 0.5) + 1;
                         epsilon_(i) = scaling * 1e-3 * spacing(i);
                         // double minEpsilon = 1.50 * spacing(i);
                         // double minSpacing = 0.50 * spacing(i);
                         // epsilon_(i) = std::max(minEpsilon, epsilon_(i));
                         // unsigned int scaling =
                         //     std::ceil((epsilon_(i) - minEpsilon) /
                         //     minSpacing);
                         // epsilon_(i) = minEpsilon + scaling * minSpacing;
                       });
  Kokkos::fence();

  // perform neighbor search by epsilon size
  double maxRatio, meanNeighbor;
  unsigned int minNeighbor, maxNeighbor;
  unsigned int minNeighborLists =
      1 + pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
              true, coords, neighborLists_, epsilon_, 0.0, 0.0);
  if (minNeighborLists > neighborLists_.extent(1))
    Kokkos::resize(neighborLists_, localParticleNum, minNeighborLists);
  pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
      false, coords, neighborLists_, epsilon_, 0.0, 0.0);

  maxRatio = 0.0;
  minNeighbor = 1000;
  maxNeighbor = 0;
  meanNeighbor = 0;
  for (std::size_t i = 0; i < localParticleNum; i++) {
    if (neighborLists_(i, 0) < minNeighbor)
      minNeighbor = neighborLists_(i, 0);
    if (neighborLists_(i, 0) > maxNeighbor)
      maxNeighbor = neighborLists_(i, 0);
    meanNeighbor += neighborLists_(i, 0);

    if (maxRatio < epsilon_(i) / spacing(i)) {
      maxRatio = epsilon_(i) / spacing(i);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &minNeighbor, 1, MPI_UNSIGNED, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maxNeighbor, 1, MPI_UNSIGNED, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &meanNeighbor, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maxRatio, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpiRank_ == 0)
    printf("\nAfter satisfying least number of neighbors\nmin neighbor: %d, "
           "max neighbor: %d , mean "
           "neighbor: %.2f, max ratio: %.2f\n",
           minNeighbor, maxNeighbor, meanNeighbor / (double)globalParticleNum,
           maxRatio);
}

Equation::Equation::Equation()
    : globalError_(0.0), globalNormalizedError_(0.0), errorTolerance_(1e-3),
      maxRefinementIteration_(6), refinementIteration_(0),
      refinementMethod_(AdaptiveRefinement), mpiRank_(0), mpiSize_(0),
      outputLevel_(0), polyOrder_(2), ghostMultiplier_(8.0) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

Void Equation::Equation::SetPolyOrder(const Size polyOrder) {
  polyOrder_ = polyOrder;
}

Void Equation::Equation::SetDimension(const Size dimension) {
  particleMgr_.SetDimension(dimension);
}

Void Equation::Equation::SetDomainType(
    const Geometry::SupportedDomainShape shape) {
  particleMgr_.SetDomainType(shape);
}

Void Equation::Equation::SetDomainSize(const std::vector<Scalar> &size) {
  particleMgr_.SetSize(size);
}

Void Equation::Equation::SetInitialDiscretizationResolution(
    const Scalar spacing) {
  particleMgr_.SetSpacing(spacing);
}

Void Equation::Equation::SetErrorTolerance(const Scalar errorTolerance) {
  errorTolerance_ = errorTolerance;
}

Void Equation::Equation::SetRefinementMethod(
    const RefinementMethod refinementMethod) {
  refinementMethod_ = refinementMethod;
}

Void Equation::Equation::SetMaxRefinementIteration(
    const Size maxRefinementIteration) {
  maxRefinementIteration_ = maxRefinementIteration;
}

Void Equation::Equation::SetOutputLevel(const Size outputLevel) {
  outputLevel_ = outputLevel;
}

Void Equation::Equation::SetGhostMultiplier(const Scalar multiplier) {
  ghostMultiplier_ = multiplier;
}

Void Equation::Equation::SetRefinementMarkRatio(const Scalar ratio) {
  markRatio_ = ratio;
}

Scalar Equation::Equation::GetObjFunc() { return 0.0; }

Void Equation::Equation::Init() {
  Clear();

  SetGhostMultiplier(8.0);
  particleMgr_.Init();
}

Void Equation::Equation::Update() {
  double tStart, tEnd;
  tStart = MPI_Wtime();
  refinementIteration_ = 0;
  double error = 1e9;

  while (error > errorTolerance_ &&
         refinementIteration_ < maxRefinementIteration_) {
    if (mpiRank_ == 0) {
      printf("\nRefinement iteration: %ld\n", refinementIteration_);
    }
    this->DiscretizeEquation();
    this->InitPreconditioner();
    {
      double tLinearSystemStart, tLinearSystemEnd;
      tLinearSystemStart = MPI_Wtime();
      this->SolveEquation();
      tLinearSystemEnd = MPI_Wtime();
      if (mpiRank_ == 0) {
        for (int i = linearSystemsPtr_.size() - 1; i >= 0; i--) {
          for (unsigned int j = 0; j < linearSystemsPtr_.size() - i; j++)
            printf("  ");
          printf("Level: %4d, field relaxation duration: %.4fs\n", i,
                 preconditionerPtr_->GetFieldRelaxationTimer(i));
        }
        printf("Duration of solving linear system: %.4fs\n",
               tLinearSystemEnd - tLinearSystemStart);
      }
    }
    this->CalculateError();
    this->Mark();
    this->Output();

    refinementIteration_++;
    error = globalNormalizedError_;

    if (refinementIteration_ < maxRefinementIteration_ &&
        error > errorTolerance_)
      particleMgr_.Refine(splitTag_);

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpiRank_ == 0)
      printf("End of adaptive refinement iteration %ld\n",
             refinementIteration_);
  }
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("End of updating physics\nDuration of updating is %.4fs\n",
           tEnd - tStart);
  }
}

Void Equation::Equation::CalculateSensitivity(
    DefaultParticleManager &particleMgr, HostRealVector &sensitivity) {}

void Equation::Equation::SetKappa(
    const std::function<double(const double, const double, const double)>
        &func) {
  singleKappaFunc_ = func;

  kappaFuncType_ = 1;
}

void Equation::Equation::SetKappa(
    const std::function<Void(const HostRealMatrix &coords,
                             const HostRealVector &spacing,
                             HostRealVector &kappa)> &func) {
  multipleKappaFunc_ = func;

  kappaFuncType_ = 2;
}