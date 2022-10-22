#include "TopologyOptimization/TopologyOptimization.hpp"
#include "Equation/Equation.hpp"
#include <memory>

Void TopologyOptimization::TopologyOptimization::CalculateSensitivity() {
  equationPtr_->Init();
  equationPtr_->Update();
  equationPtr_->CalculateSensitivity(particleMgr_, sensitivity_);
}

Void TopologyOptimization::TopologyOptimization::Output() {
  std::string outputFileName =
      "vtk/TopologyOptimization" + std::to_string(iteration_) + ".vtk";

  if (mpiRank_ == 0)
    printf("Start of writing adaptive step output file\n");
  particleMgr_.Output(outputFileName, true);

  std::ofstream vtkStream;
  // output density
  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS density float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < density_.extent(0); i++) {
        float x = density_(i);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(int));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // output sensitivity
  if (mpiRank_ == 0) {
    vtkStream.open(outputFileName,
                   std::ios::out | std::ios::app | std::ios::binary);

    vtkStream << "SCALARS sensitivity float 1" << std::endl
              << "LOOKUP_TABLE default" << std::endl;

    vtkStream.close();
  }
  for (int rank = 0; rank < mpiSize_; rank++) {
    if (rank == mpiRank_) {
      vtkStream.open(outputFileName,
                     std::ios::out | std::ios::app | std::ios::binary);
      for (std::size_t i = 0; i < sensitivity_.extent(0); i++) {
        float x = sensitivity_(i);
        SwapEnd(x);
        vtkStream.write(reinterpret_cast<char *>(&x), sizeof(int));
      }
      vtkStream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

TopologyOptimization::TopologyOptimization::TopologyOptimization() {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

TopologyOptimization::TopologyOptimization::~TopologyOptimization() {}

Void TopologyOptimization::TopologyOptimization::AddEquation(
    Equation::Equation &equation) {
  equationPtr_ = &equation;
}

Void TopologyOptimization::TopologyOptimization::SetDimension(
    const Size dimension) {
  equationPtr_->SetDimension(dimension);

  particleMgr_.SetDimension(dimension);
}

Void TopologyOptimization::TopologyOptimization::SetDomainType(
    const Geometry::SupportedDomainShape shape) {
  equationPtr_->SetDomainType(shape);

  particleMgr_.SetDomainType(shape);
}

Void TopologyOptimization::TopologyOptimization::SetDomainSize(
    const std::vector<Scalar> &size) {
  equationPtr_->SetDomainSize(size);

  particleMgr_.SetSize(size);

  domainVolume_ = size[0] * size[1];
}

Void TopologyOptimization::TopologyOptimization::
    SetInitialDiscretizationResolution(const Scalar spacing) {
  equationPtr_->SetInitialDiscretizationResolution(spacing);

  particleMgr_.SetSpacing(spacing);
}

Void TopologyOptimization::TopologyOptimization::SetVolumeFraction(
    const Scalar volumeFraction) {
  volumeFraction_ = volumeFraction;
}

Void TopologyOptimization::TopologyOptimization::SetMaxIteration(
    const Size maxIteration) {
  maxIteration_ = maxIteration;
}

Void TopologyOptimization::TopologyOptimization::Init() {
  equationPtr_->SetPolyOrder(1);
  equationPtr_->SetMaxRefinementIteration(1);
  equationPtr_->SetOutputLevel(0);
  equationPtr_->SetRefinementMarkRatio(1.0);
}

Void TopologyOptimization::TopologyOptimization::Optimize() {}