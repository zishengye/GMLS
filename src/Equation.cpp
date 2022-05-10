#include <algorithm>
#include <execution>
#include <iostream>
#include <vector>

#include "Equation.hpp"

void Equation::InitLinearSystem() {
  BuildGhost();

  GlobalIndex globalParticleNum = particleMgr_.GetGlobalParticleNum();
  if (mpiRank_ == 0)
    printf("Start of preparing linear system with %lu particles\n",
           globalParticleNum);

  auto newMat = std::make_shared<PetscMatrix>();
  linearSystemsPtr_.push_back(newMat);
  preconditionerPtr_->AddLinearSystem(newMat);

  MPI_Barrier(MPI_COMM_WORLD);
}

void Equation::ConstructLinearSystem() {}

void Equation::ConstructRhs() {}

void Equation::DiscretizeEquation() {
  this->InitLinearSystem();
  this->ConstructLinearSystem();
  this->ConstructRhs();
}

void Equation::InitPreconditioner() {
  preconditionerPtr_->ConstructSmoother(
      std::make_shared<HierarchicalParticleManager>(particleMgr_));
  preconditionerPtr_->ConstructInterpolation(
      std::make_shared<HierarchicalParticleManager>(particleMgr_));
  preconditionerPtr_->ConstructRestriction(
      std::make_shared<HierarchicalParticleManager>(particleMgr_));

  PetscInt localRow;
  MatGetLocalSize(linearSystemsPtr_[refinementIteration_]->GetReference(),
                  &localRow, PETSC_NULL);
  preconditionerPtr_->PrepareVectors(localRow);

  KSP &ksp = ksp_.GetReference();
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPGMRES);
  KSPSetOperators(ksp, linearSystemsPtr_[refinementIteration_]->GetReference(),
                  linearSystemsPtr_[refinementIteration_]->GetReference());
  KSPSetFromOptions(ksp);

  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCSHELL);

  PCShellSetApply(pc, PreconditioningIterationWrapper);
  PCShellSetContext(pc, preconditionerPtr_.get());

  KSPSetUp(ksp);
}

void Equation::SolveEquation() {
  ksp_.Solve(b_.GetReference(), x_.GetReference());
}

void Equation::CalculateError() {
  const LocalIndex localParticleNum = particleMgr_.GetLocalParticleNum();
  Kokkos::resize(error_, localParticleNum);
}

void Equation::Refine() {
  if (mpiRank_ == 0)
    printf("Global error: %.4f, with tolerance: %.4f\n", globalError_,
           errorTolerance_);

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
  while (!isSplit) {
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

  currentErrorSplit = sqrt(currentErrorSplit);

  // mark particles
  Kokkos::resize(splitTag_, error_.extent(0));
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
        if (splitTag_(i) == 1)
          tMarkedParticleNum++;
      },
      Kokkos::Sum<std::size_t>(markedParticleNum));
  Kokkos::fence();
  MPI_Allreduce(MPI_IN_PLACE, &markedParticleNum, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  if (mpiRank_ == 0)
    printf("Marked %ld particles with mark ratio: %.2f\n", markedParticleNum,
           markRatio_);

  // ensure quasiuniform
  auto &coords = particleMgr_.GetParticleCoords();
  auto &sourceCoords = hostGhostParticleCoords_;
  auto &particleType = particleMgr_.GetParticleType();
  auto &sourceParticleType = hostGhostParticleType_;

  auto &particleRefinementLevel = particleMgr_.GetParticleRefinementLevel();
  HostIndexVector ghostParticleRefinementLevel;
  ghost_.ApplyGhost(particleRefinementLevel, ghostParticleRefinementLevel);

  HostIndexVector ghostSplitTag;
  HostIndexVector crossRefinementLevel, ghostCrossRefinementLevel;
  Kokkos::resize(crossRefinementLevel, error_.extent(0));
  int iterationFinished = 1;
  iteCounter = 0;
  while (iterationFinished != 0) {
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

            if (ghostSplitTag(nearestIndex) +
                    ghostParticleRefinementLevel(nearestIndex) >
                particleRefinementLevel(i)) {
              splitTag_(i) = 1;
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
                ghostSplitTag[neighborParticleIndex];
            if (ghostRefinementLevel < minRefinementLevel)
              minRefinementLevel = ghostRefinementLevel;
            if (ghostRefinementLevel > maxRefinementLevel)
              maxRefinementLevel = ghostRefinementLevel;
          }

          if (maxRefinementLevel - minRefinementLevel > 1)
            crossRefinementLevel(i) = splitTag_(i) + particleRefinementLevel(i);
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
              splitTag_(i) = 1;
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
        if (splitTag_(i) == 1)
          tMarkedParticleNum++;
      },
      Kokkos::Sum<std::size_t>(markedParticleNum));
  Kokkos::fence();
  MPI_Allreduce(MPI_IN_PLACE, &markedParticleNum, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  if (mpiRank_ == 0)
    printf("Marked %ld particles after smoothing particles\n",
           markedParticleNum);

  particleMgr_.Refine(splitTag_);
}

void Equation::BuildGhost() {
  auto &particleCoords = particleMgr_.GetParticleCoords();
  auto &particleSize = particleMgr_.GetParticleSize();
  auto &particleIndex = particleMgr_.GetParticleIndex();
  auto &particleType = particleMgr_.GetParticleType();
  ghost_.Init(particleCoords, particleSize, particleCoords, ghostMultiplier_,
              particleMgr_.GetDimension());
  ghost_.ApplyGhost(particleCoords, hostGhostParticleCoords_);
  ghost_.ApplyGhost(particleIndex, hostGhostParticleIndex_);
  ghost_.ApplyGhost(particleType, hostGhostParticleType_);
}

void Equation::Output() {
  if (outputLevel_ > 0) {
    if (mpiRank_ == 0)
      printf("Start of writing adaptive step output file\n");
    // write particles
    particleMgr_.Output(
        "AdaptiveStep" + std::to_string(refinementIteration_) + ".vtk", true);
  }

  std::ofstream vtkStream;
  // output epsilon
  if (mpiRank_ == 0) {
    vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                       ".vtk",
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS epsilon float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                         ".vtk",
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < epsilon_.extent(0); i++) {
        float x = epsilon_(i);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // output number of neighbor
  if (mpiRank_ == 0) {
    vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                       ".vtk",
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS nn int 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open("vtk/AdaptiveStep" + std::to_string(refinementIteration_) +
                         ".vtk",
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
}

void Equation::ConstructNeighborLists(const unsigned int satisfiedNumNeighbor) {
  auto &sourceCoords = hostGhostParticleCoords_;
  auto &coords = particleMgr_.GetParticleCoords();
  auto &spacing = particleMgr_.GetParticleSize();

  const unsigned int localParticleNum = coords.extent(0);
  const unsigned long globalParticleNum = particleMgr_.GetGlobalParticleNum();

  // construct neighbor lists
  Kokkos::resize(neighborLists_, localParticleNum, 1);
  Kokkos::resize(epsilon_, localParticleNum);

  for (std::size_t i = 0; i < localParticleNum; i++) {
    epsilon_(i) = 1.50005 * spacing(i);
  }

  auto pointCloudSearch(Compadre::CreatePointCloudSearch(
      sourceCoords, particleMgr_.GetDimension()));
  bool isNeighborSearchPassed = false;

  double maxRatio, meanNeighbor;
  unsigned int minNeighbor, maxNeighbor, iteCounter;
  iteCounter = 0;
  while (!isNeighborSearchPassed) {
    iteCounter++;
    unsigned int minNeighborLists =
        1 + pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
                true, coords, neighborLists_, epsilon_, 0.0, 0.0);
    if (minNeighborLists > neighborLists_.extent(1))
      Kokkos::resize(neighborLists_, localParticleNum, minNeighborLists);
    pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
        false, coords, neighborLists_, epsilon_, 0.0, 0.0);

    bool passNeighborNumCheck = true;

    maxRatio = 0.0;
    minNeighbor = 1000;
    maxNeighbor = 0;
    meanNeighbor = 0;
    for (std::size_t i = 0; i < localParticleNum; i++) {
      if (neighborLists_(i, 0) <= satisfiedNumNeighbor) {
        epsilon_(i) += 0.25 * spacing(i);
        passNeighborNumCheck = false;
      }
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

    unsigned int corePassCheck = 0;
    if (!passNeighborNumCheck)
      corePassCheck = 1;

    MPI_Allreduce(MPI_IN_PLACE, &corePassCheck, 1, MPI_UNSIGNED, MPI_SUM,
                  MPI_COMM_WORLD);
    if (corePassCheck == 0)
      isNeighborSearchPassed = true;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpiRank_ == 0)
    printf("\nAfter satisfying least number of neighbors\niteration count: %d "
           "min neighbor: %d, max neighbor: %d , mean "
           "neighbor: %.2f, max ratio: %.2f\n",
           iteCounter, minNeighbor, maxNeighbor,
           meanNeighbor / (double)globalParticleNum, maxRatio);
}

Equation::Equation()
    : errorTolerance_(1e-3), maxRefinementIteration_(6),
      refinementMethod_(AdaptiveRefinement), outputLevel_(0), polyOrder_(2) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

void Equation::SetPolyOrder(const unsigned int polyOrder) {
  polyOrder_ = polyOrder;
}

void Equation::SetDimension(const unsigned int dimension) {
  particleMgr_.SetDimension(dimension);
}

void Equation::SetDomainType(const SimpleDomainShape shape) {
  particleMgr_.SetDomainType(shape);
}

void Equation::SetDomainSize(const std::vector<Scalar> &size) {
  particleMgr_.SetSize(size);
}

void Equation::SetInitialDiscretizationResolution(const double spacing) {
  particleMgr_.SetSpacing(spacing);
}

void Equation::SetErrorTolerance(const double errorTolerance) {
  errorTolerance_ = errorTolerance;
}

void Equation::SetRefinementMethod(const RefinementMethod refinementMethod) {
  refinementMethod_ = refinementMethod;
}

void Equation::SetMaxRefinementIteration(
    const unsigned int maxRefinementIteration) {
  maxRefinementIteration_ = maxRefinementIteration;
}

void Equation::SetOutputLevel(const unsigned int outputLevel) {
  outputLevel_ = outputLevel;
}

void Equation::SetGhostMultiplier(const double multiplier) {
  ghostMultiplier_ = multiplier;
}

void Equation::SetRefinementMarkRatio(const double ratio) {
  markRatio_ = ratio;
}

void Equation::Init() {
  SetGhostMultiplier(6.0);
  particleMgr_.Init();
}

void Equation::Update() {
  double tStart, tEnd;
  tStart = MPI_Wtime();
  refinementIteration_ = 0;
  double error = 1e9;

  while (error > errorTolerance_ &&
         refinementIteration_ < maxRefinementIteration_) {
    if (mpiRank_ == 0) {
      printf("\nRefinement iteration: %d\n", refinementIteration_);
    }
    this->DiscretizeEquation();
    this->InitPreconditioner();
    {
      double tStart, tEnd;
      tStart = MPI_Wtime();
      this->SolveEquation();
      tEnd = MPI_Wtime();
      if (mpiRank_ == 0)
        printf("Duration of solving linear system: %.4fs\n", tEnd - tStart);
    }
    this->CalculateError();
    this->Output();
    this->Refine();

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpiRank_ == 0)
      printf("End of adaptive refinement iteration %d\n", refinementIteration_);

    refinementIteration_++;
    error = globalError_;
  }
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("End of updating physics\nDuration of updating is %.4fs\n",
           tEnd - tStart);
  }
}