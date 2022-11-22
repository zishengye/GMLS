#include "Equation/Stokes/StokesMatrix.hpp"
#include "LinearAlgebra/BlockMatrix.hpp"

Void Equation::StokesMatrix::ClearTimer() {
  matrixMultiplicationDuration_ = 0.0;
  matrixSmoothingDuration_ = 0.0;

  blockMat_->ClearTimer();
}

double Equation::StokesMatrix::GetMatrixMultiplicationDuration() {
  return matrixMultiplicationDuration_;
}

double Equation::StokesMatrix::GetMatrixSmoothingDuration() {
  return matrixSmoothingDuration_;
}

double Equation::StokesMatrix::GetVelocitySmoothingDuration() {
  return blockMat_->GetA00Timer();
}

double Equation::StokesMatrix::GetPressureSmoothingDuration() {
  return blockMat_->GetA11Timer();
}

Void Equation::StokesMatrix::MatrixVectorMultiplication(DefaultVector &x,
                                                        DefaultVector &y) {
  double timer1, timer2;
  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();

  x.OrthogonalizeToConstant(localRhsVectorOffset_[1], localRhsVectorOffset_[2]);
  BlockMatrix::MatrixVectorMultiplication(x, y);
  y.OrthogonalizeToConstant(localLhsVectorOffset_[1], localLhsVectorOffset_[2]);

  timer2 = MPI_Wtime();
  matrixMultiplicationDuration_ += timer2 - timer1;
}

Void Equation::StokesMatrix::ApplyPreconditioningIteration(DefaultVector &x,
                                                           DefaultVector &y) {
  double timer1, timer2;
  MPI_Barrier(MPI_COMM_WORLD);
  timer1 = MPI_Wtime();

  x.OrthogonalizeToConstant(localRhsVectorOffset_[1], localRhsVectorOffset_[2]);
  this->ApplySchurComplementPreconditioningIteration(x, y);
  y.OrthogonalizeToConstant(localLhsVectorOffset_[1], localLhsVectorOffset_[2]);

  timer2 = MPI_Wtime();
  matrixSmoothingDuration_ += timer2 - timer1;
}