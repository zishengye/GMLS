#include "LinearAlgebra/Impl/Default/DefaultLinearSolver.hpp"
#include "Core/Typedef.hpp"
#include "LinearAlgebra/Impl/Default/DefaultVector.hpp"
#include "LinearAlgebra/Impl/Default/SquareMatrix.hpp"
#include "LinearAlgebra/LinearAlgebra.hpp"

#include <memory>
#include <mpi.h>

Void LinearAlgebra::Impl::DefaultLinearSolver::GmresIteration(
    DefaultVector &b, DefaultVector &x) {
  LocalIndex iterationStep = 0;
  LocalIndex innerStep = 0;

  Scalar bNorm = b.Norm2();
  Scalar rNorm;

  x.Zeros();

  DefaultVector r, w, z;
  std::vector<DefaultVector> v;
  r.Resize(x.GetLocalSize());
  w.Resize(x.GetLocalSize());
  z.Resize(x.GetLocalSize());
  v.resize(restartIteration_ + 1);
  for (int i = 0; i <= restartIteration_; i++)
    v[i].Resize(x.GetLocalSize());
  linearSystemPtr_->MatrixVectorMultiplication(x, z);
  z -= b;
  z.Scale(-1.0);
  (*preconditioningFunctionPtr_)(z, r);
  rNorm = r.Norm2();

  Scalar beta = rNorm;

  HostRealVector y;

  v[0] = r / beta;

  UpperHessenbergMatrix hessenberg(restartIteration_);

  while (true) {
    if (mpiRank_ == 0 && monitor_)
      printf("  %4d GMRes Residual norm %10.6e\n", iterationStep,
             rNorm / bNorm);

    iterationStep++;
    innerStep++;

    int j = innerStep - 1;

    // stop criterion
    if (iterationStep == maxIteration_ || rNorm / bNorm < relativeTolerance_) {
      UpdateSolution(y, v, x);
      break;
    }

    // preconditioning
    linearSystemPtr_->MatrixVectorMultiplication(v[j], z);
    (*preconditioningFunctionPtr_)(z, w);

    // build hessenberg matrix
    for (int i = 0; i < innerStep; i++) {
      hessenberg(i, j) = w.Dot(v[i]);

      w -= hessenberg(i, j) * v[i];
    }
    hessenberg(j + 1, j) = w.Norm2();
    v[j + 1] = w / hessenberg(j + 1, j);

    // solve hessenberg matrix
    rNorm = std::abs(UpdateHessenbergMatrix(hessenberg, innerStep, beta, y));

    // restart
    if (innerStep == restartIteration_) {
      UpdateSolution(y, v, x);

      innerStep = 0;

      linearSystemPtr_->MatrixVectorMultiplication(x, z);
      z -= b;
      z.Scale(-1.0);
      (*preconditioningFunctionPtr_)(z, r);
      rNorm = r.Norm2();

      if (rNorm / bNorm < relativeTolerance_) {
        break;
      }

      beta = rNorm;

      v[0] = r / beta;
    }
  }
}

Void LinearAlgebra::Impl::DefaultLinearSolver::FlexGmresIteration(
    DefaultVector &b, DefaultVector &x) {
  LocalIndex iterationStep = 0;
  LocalIndex innerStep = 0;

  Scalar bNorm = b.Norm2();
  Scalar rNorm;

  DefaultVector r, w;
  std::vector<DefaultVector> v, z;
  r.Resize(x.GetLocalSize());
  w.Resize(x.GetLocalSize());
  v.resize(restartIteration_ + 1);
  z.resize(restartIteration_);
  for (int i = 0; i <= restartIteration_; i++)
    v[i].Resize(x.GetLocalSize());
  for (int i = 0; i < restartIteration_; i++)
    z[i].Resize(x.GetLocalSize());
  linearSystemPtr_->MatrixVectorMultiplication(x, r);
  r -= b;
  r.Scale(-1.0);
  rNorm = r.Norm2();

  Scalar beta = rNorm;

  HostRealVector y;

  v[0] = r / beta;

  UpperHessenbergMatrix hessenberg(restartIteration_);

  while (true) {
    if (mpiRank_ == 0 && monitor_)
      printf("  %4d Flex-GMRes Residual norm %10.6e\n", iterationStep,
             rNorm / bNorm);

    iterationStep++;
    innerStep++;

    int j = innerStep - 1;

    // stop criterion
    if (iterationStep == maxIteration_ || rNorm / bNorm < relativeTolerance_) {
      UpdateSolution(y, z, x);
      break;
    }

    // preconditioning
    (*preconditioningFunctionPtr_)(v[j], z[j]);

    linearSystemPtr_->MatrixVectorMultiplication(z[j], w);

    // build hessenberg matrix
    for (int i = 0; i < innerStep; i++) {
      hessenberg(i, j) = w.Dot(v[i]);

      w -= hessenberg(i, j) * v[i];
    }
    hessenberg(j + 1, j) = w.Norm2();
    v[j + 1] = w / hessenberg(j + 1, j);

    // solve hessenberg matrix
    rNorm = std::abs(UpdateHessenbergMatrix(hessenberg, innerStep, beta, y));

    // restart
    if (innerStep == restartIteration_) {
      UpdateSolution(y, z, x);

      innerStep = 0;

      linearSystemPtr_->MatrixVectorMultiplication(x, r);
      r -= b;
      r.Scale(-1.0);
      rNorm = r.Norm2();

      if (rNorm / bNorm < relativeTolerance_) {
        break;
      }

      beta = rNorm;

      v[0] = r / beta;
    }
  }
}

Void LinearAlgebra::Impl::DefaultLinearSolver::ConjugateGradientIteration(
    DefaultVector &b, DefaultVector &x) {}

Void LinearAlgebra::Impl::DefaultLinearSolver::RichardsonIteration(
    DefaultVector &b, DefaultVector &x) {
  DefaultVector r, z;
  r.Resize(x.GetLocalSize());
  z.Resize(x.GetLocalSize());

  x.Zeros();

  for (int i = 0; i < 3; i++) {
    linearSystemPtr_->MatrixVectorMultiplication(x, r);
    r -= b;
    (*preconditioningFunctionPtr_)(r, z);
    x -= z;
  }
}

Void LinearAlgebra::Impl::DefaultLinearSolver::PreconditioningOnlyIteration(
    DefaultVector &b, DefaultVector &x) {
  (*preconditioningFunctionPtr_)(b, x);
}

Void LinearAlgebra::Impl::DefaultLinearSolver::CustomPreconditioningIteration(
    DefaultVector &b, DefaultVector &x) {
  LinearAlgebra::Vector<DefaultLinearAlgebraBackend> bVector, xVector;
  bVector.Create(b);
  xVector.Create(x);
  (*customPreconditioningFunctionPtr_)(bVector, xVector);
  xVector.Copy(x);
}

Scalar LinearAlgebra::Impl::DefaultLinearSolver::UpdateHessenbergMatrix(
    LinearAlgebra::Impl::UpperHessenbergMatrix &hessenberg,
    const LocalIndex size, const Scalar beta, HostRealVector &y) {
  LinearAlgebra::Impl::UpperTriangleMatrix triMat(size);

  for (int i = 0; i < size; i++) {
    for (int j = i; j < size; j++) {
      triMat(i, j) = hessenberg(i, j);
    }
  }

  HostRealVector g;
  Kokkos::resize(g, size + 1);
  Kokkos::resize(y, size);

  g(0) = beta;

  // rotate
  for (int i = 0; i < size; i++) {
    Scalar h1 = triMat(i, i);
    Scalar h2 = hessenberg(i + 1, i);

    Scalar r = sqrt(h1 * h1 + h2 * h2);

    Scalar s = h2 / r;
    Scalar c = h1 / r;

    triMat(i, i) = c * h1 + s * h2;

    for (int j = i + 1; j < size; j++) {
      Scalar h1 = triMat(i, j);
      Scalar h2 = triMat(i + 1, j);

      triMat(i, j) = c * h1 + s * h2;
      triMat(i + 1, j) = -s * h1 + c * h2;
    }

    g(i + 1) = -s * g(i);
    g(i) = c * g(i);
  }

  triMat.Solve(g, y);

  return g(size);
}

Void LinearAlgebra::Impl::DefaultLinearSolver::UpdateSolution(
    HostRealVector &y, std::vector<DefaultVector> &z, DefaultVector &x) {
  for (int i = 0; i < y.extent(0); i++)
    x += y(i) * z[i];
}

LinearAlgebra::Impl::DefaultLinearSolver::DefaultLinearSolver() {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize_);
}

LinearAlgebra::Impl::DefaultLinearSolver::~DefaultLinearSolver() {}

Void LinearAlgebra::Impl::DefaultLinearSolver::AddLinearSystem(
    std::shared_ptr<DefaultMatrix> matPtr,
    const LinearSolverDescriptor<LinearAlgebra::Impl::DefaultBackend>
        &descriptor) {
  linearSystemPtr_ = matPtr;
  solveFunctionPtr_ =
      std::make_shared<std::function<Void(DefaultVector &, DefaultVector &)>>();

  if (descriptor.outerIteration == 1)
    *solveFunctionPtr_ = std::function<Void(DefaultVector &, DefaultVector &)>(
        [=](DefaultVector &x, DefaultVector &y) {
          this->FlexGmresIteration(x, y);
        });
  else if (descriptor.outerIteration == 0)
    *solveFunctionPtr_ = std::function<Void(DefaultVector &, DefaultVector &)>(
        [=](DefaultVector &x, DefaultVector &y) {
          this->GmresIteration(x, y);
        });
  else
    *solveFunctionPtr_ = std::function<Void(DefaultVector &, DefaultVector &)>(
        [=](DefaultVector &x, DefaultVector &y) {
          this->RichardsonIteration(x, y);
        });

  if (descriptor.outerIteration > 0) {
    preconditioningFunctionPtr_ = std::make_shared<
        std::function<Void(DefaultVector &, DefaultVector &)>>();
    *preconditioningFunctionPtr_ =
        std::function<Void(DefaultVector &, DefaultVector &)>(
            [=](DefaultVector &b, DefaultVector &x) {
              this->CustomPreconditioningIteration(b, x);
            });

    customPreconditioningFunctionPtr_ = std::make_shared<std::function<Void(
        LinearAlgebra::Vector<LinearAlgebra::Impl::DefaultBackend> &,
        LinearAlgebra::Vector<LinearAlgebra::Impl::DefaultBackend> &)>>();
    *customPreconditioningFunctionPtr_ = descriptor.preconditioningIteration;
  } else {
    if (descriptor.customPreconditioner == false) {
      preconditioningFunctionPtr_ = std::make_shared<
          std::function<Void(DefaultVector &, DefaultVector &)>>();
      *preconditioningFunctionPtr_ =
          std::function<Void(DefaultVector &, DefaultVector &)>(
              [=](DefaultVector &b, DefaultVector &x) {
                matPtr->SorPreconditioning(b, x);
              });
    } else {
      preconditioningFunctionPtr_ = std::make_shared<
          std::function<Void(DefaultVector &, DefaultVector &)>>();
      *preconditioningFunctionPtr_ =
          std::function<Void(DefaultVector &, DefaultVector &)>(
              [=](DefaultVector &b, DefaultVector &x) {
                this->CustomPreconditioningIteration(b, x);
              });

      customPreconditioningFunctionPtr_ = std::make_shared<std::function<Void(
          LinearAlgebra::Vector<LinearAlgebra::Impl::DefaultBackend> &,
          LinearAlgebra::Vector<LinearAlgebra::Impl::DefaultBackend> &)>>();
      *customPreconditioningFunctionPtr_ = descriptor.preconditioningIteration;
    }
  }

  maxIteration_ = descriptor.maxIter;
  relativeTolerance_ = descriptor.relativeTol;
  if (descriptor.outerIteration >= 0)
    restartIteration_ = 30;

  if (descriptor.outerIteration > 0) {
    postCheck_ = true;
    monitor_ = true;
  } else {
    postCheck_ = false;
    monitor_ = false;
  }
}

Void LinearAlgebra::Impl::DefaultLinearSolver::Solve(DefaultVector &b,
                                                     DefaultVector &x) {
  (*solveFunctionPtr_)(b, x);

  if (postCheck_) {
    DefaultVector residual;

    residual.Resize(x.GetLocalSize());

    linearSystemPtr_->MatrixVectorMultiplication(x, residual);

    residual -= b;

    int mpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    Scalar residualNorm = residual.Norm2();
    Scalar bNorm = b.Norm2();
    if (mpiRank == 0)
      printf("The residual norm in the post check: %10.6e\n",
             residualNorm / bNorm);
  }
}