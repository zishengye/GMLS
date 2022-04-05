#include <iostream>

#include "Equation.hpp"

void Equation::InitLinearSystem() {
  BuildGhost();

  GlobalIndex globalParticleNum = particleMgr_.GetGlobalParticleNum();
  if (mpiRank_ == 0)
    printf("Start of preparing linear system with %lu particles\n",
           globalParticleNum);

  linearSystemsPtr_.push_back(std::make_shared<PetscMatrix>());

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
  ksp_.SetUp(*(linearSystemsPtr_[refinementIteration_]), KSPGMRES);
}

void Equation::SolveEquation() {
  ksp_.Solve(b_.GetReference(), x_.GetReference());
}

void Equation::CalculateError() {
  const LocalIndex localParticleNum = particleMgr_.GetLocalParticleNum();
  Kokkos::resize(error_, localParticleNum);
}

void Equation::Refine() {}

void Equation::BuildGhost() {
  auto &particleCoords = particleMgr_.GetParticleCoords();
  auto &particleSize = particleMgr_.GetParticleSize();
  auto &particleIndex = particleMgr_.GetParticleIndex();
  ghost_.Init(particleCoords, particleSize, particleCoords, ghostMultiplier_,
              particleMgr_.GetDimension());
  ghost_.ApplyGhost(particleCoords, hostGhostParticleCoords_);
  ghost_.ApplyGhost(particleIndex, hostGhostParticleIndex_);
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
      for (int i = 0; i < epsilon_.extent(0); i++) {
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
      for (int i = 0; i < neighborLists_.extent(0); i++) {
        int x = neighborLists_(i, 0);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(float));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void Equation::ConstructNeighborLists(const int satisfiedNumNeighbor) {
  auto &sourceCoords = hostGhostParticleCoords_;
  auto &coords = particleMgr_.GetParticleCoords();
  auto &spacing = particleMgr_.GetParticleSize();

  const unsigned int localParticleNum = coords.extent(0);
  const unsigned long globalParticleNum = particleMgr_.GetGlobalParticleNum();

  // construct neighbor lists
  Kokkos::resize(neighborLists_, localParticleNum, 1);
  Kokkos::resize(epsilon_, localParticleNum);

  for (int i = 0; i < localParticleNum; i++) {
    epsilon_(i) = 1.50005 * spacing(i);
  }

  auto pointCloudSearch(Compadre::CreatePointCloudSearch(
      sourceCoords, particleMgr_.GetDimension()));
  bool isNeighborSearchPassed = false;

  double maxRatio, meanNeighbor;
  int minNeighbor, maxNeighbor, iteCounter;
  iteCounter = 0;
  while (!isNeighborSearchPassed) {
    iteCounter++;
    int minNeighborLists =
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
    for (int i = 0; i < localParticleNum; i++) {
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
    MPI_Allreduce(MPI_IN_PLACE, &minNeighbor, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxNeighbor, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanNeighbor, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxRatio, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    int corePassCheck = 0;
    if (!passNeighborNumCheck)
      corePassCheck = 1;

    MPI_Allreduce(MPI_IN_PLACE, &corePassCheck, 1, MPI_INT, MPI_SUM,
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

void Equation::SetPolyOrder(const int polyOrder) { polyOrder_ = polyOrder; }

void Equation::SetDimension(const int dimension) {
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

void Equation::SetMaxRefinementIteration(const int maxRefinementIteration) {
  maxRefinementIteration_ = maxRefinementIteration;
}

void Equation::SetOutputLevel(const int outputLevel) {
  outputLevel_ = outputLevel;
}

void Equation::SetGhostMultiplier(const double multiplier) {
  ghostMultiplier_ = multiplier;
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
      printf("Refinement iteration: %d\n", refinementIteration_);
    }
    this->DiscretizeEquation();
    this->InitPreconditioner();
    {
      double tStart, tEnd;
      tStart = MPI_Wtime();
      this->SolveEquation();
      tEnd = MPI_Wtime();
      if (mpiRank_ == 0)
        printf("Duration of solving linear system: %fs\n", tEnd - tStart);
    }
    this->CalculateError();
    this->Refine();
    this->Output();

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpiRank_ == 0)
      printf("End of adaptive refinement iteration %d\n", refinementIteration_);

    refinementIteration_++;
  }
  tEnd = MPI_Wtime();
  if (mpiRank_ == 0) {
    printf("End of updating physics\nDuration of updating is %fs\n",
           tEnd - tStart);
  }
}