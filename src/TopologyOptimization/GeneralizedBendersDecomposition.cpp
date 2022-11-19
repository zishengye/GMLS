#include "TopologyOptimization/GeneralizedBendersDecomposition.hpp"
#include "Core/Typedef.hpp"

#include <gurobi_c++.h>
#include <gurobi_c.h>

#include <petsclog.h>

#include <mpi.h>

Void TopologyOptimization::GeneralizedBendersDecomposition::
    CalculateSensitivity() {
  TopologyOptimization::CalculateSensitivity();

  sensitivityMatrix_.push_back(HostRealVector());
  auto &currentSensitivity = sensitivityMatrix_.back();
  Kokkos::resize(currentSensitivity, sensitivity_.extent(0));

  double adjustedObjFunc = 0.0;
  for (auto i = 0; i < sensitivity_.extent(0); i++) {
    currentSensitivity(i) = density_(i) * sensitivity_(i);
    sensitivity_(i) = currentSensitivity(i);
    adjustedObjFunc -= currentSensitivity(i) * density_(i);
  }

  objFunc_.push_back(equationPtr_->GetObjFunc());

  MPI_Allreduce(MPI_IN_PLACE, &adjustedObjFunc, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  adjustedObjFunc += objFunc_.back();
  adjustedObjFunc_.push_back(adjustedObjFunc);

  if (mpiRank_ == 0)
    printf("Current adjusted Obj Func: %f, objective function: %f\n",
           adjustedObjFunc, objFunc_.back());
}

Void TopologyOptimization::GeneralizedBendersDecomposition::Output() {
  std::string outputFileName =
      "vtk/TopologyOptimization" + std::to_string(iteration_) + ".vtk";

  Output(outputFileName);
}

Void TopologyOptimization::GeneralizedBendersDecomposition::Output(
    String outputFileName) {
  TopologyOptimization::TopologyOptimization::Output(outputFileName);
}

Void TopologyOptimization::GeneralizedBendersDecomposition::MasterCut() {
  if (mpiRank_ == 0)
    printf("Start of Master cut of GBD\n");

  unsigned int sizeOfCut = effectiveIndex_.size();
  int localParticleNum = particleMgr_.GetLocalParticleNum();
  unsigned int globalParticleNum = particleMgr_.GetGlobalParticleNum();

  // gather sensitivity
  std::vector<double> gatheredSensitivityMatrix;
  std::vector<double> resultingDensity;
  if (mpiRank_ == 0) {
    gatheredSensitivityMatrix.resize(sizeOfCut * globalParticleNum);
    resultingDensity.resize(globalParticleNum);
  }

  for (int i = 0; i < sizeOfCut; i++)
    MPI_Gatherv(
        sensitivityMatrix_[effectiveIndex_[i]].data(), localParticleNum,
        MPI_DOUBLE, gatheredSensitivityMatrix.data() + globalParticleNum * i,
        recvCount_.data(), recvOffset_.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (mpiRank_ == 0) {
    if (sizeOfCut > 1) {
      GRBEnv grbEnv(true);
      grbEnv.set("LogFile", "gbd.log");
      grbEnv.start();
      GRBModel model = GRBModel(grbEnv);
      auto eta = model.addVars(1, GRB_CONTINUOUS);
      auto binVarList = model.addVars((int)globalParticleNum, GRB_BINARY);

      // setup objective function
      GRBLinExpr objective;
      std::vector<double> objectiveWeight;
      objectiveWeight.resize(globalParticleNum + 1);
      objectiveWeight[0] = 1;
      for (int i = 0; i < globalParticleNum; i++)
        objectiveWeight[i + 1] = 0;
      objective.addTerms(objectiveWeight.data(), eta, 1);
      objective.addTerms(objectiveWeight.data() + 1, binVarList,
                         globalParticleNum);

      model.setObjective(objective, GRB_MINIMIZE);

      // setup sensitivity constraints
      std::vector<GRBLinExpr> sensitivityConstraints;
      sensitivityConstraints.resize(sizeOfCut);

      std::vector<double> etaWeight;
      etaWeight.resize(sizeOfCut);

      std::vector<double> sensitivityValue;

      for (int i = 0; i < sizeOfCut; i++) {
        etaWeight[i] = -1;
        sensitivityConstraints[i].addTerms(etaWeight.data() + i, eta, 1);
        sensitivityConstraints[i].addTerms(gatheredSensitivityMatrix.data() +
                                               i * globalParticleNum,
                                           binVarList, globalParticleNum);

        sensitivityValue.emplace_back(-adjustedObjFunc_[effectiveIndex_[i]]);

        model.addConstr(sensitivityConstraints[i], GRB_LESS_EQUAL,
                        sensitivityValue[i]);
      }

      // setup volume constraint
      std::vector<double> volumeConstraintWeight;
      volumeConstraintWeight.resize(globalParticleNum);
      for (int i = 0; i < globalParticleNum; i++) {
        volumeConstraintWeight[i] = 1.0;
      }
      GRBLinExpr volumeConstraint;
      volumeConstraint.addTerms(volumeConstraintWeight.data(), binVarList,
                                globalParticleNum);

      GRBLinExpr volumeFraction(volumeConstraintParticleNum_);

      model.addConstr(volumeConstraint, GRB_EQUAL, volumeFraction);

      // optimize
      printf("Summary of GBD cut: size of the cuts: %d\n", sizeOfCut);
      model.optimize();

      for (int i = 0; i < globalParticleNum; i++) {
        resultingDensity[i] = binVarList[i].get(GRB_DoubleAttr_X);
      }

      for (int i = 0; i < localParticleNum; i++) {
        density_(i) = resultingDensity[i];
      }

      cutCost_ = eta->get(GRB_DoubleAttr_X);

    } else {
      GRBEnv grbEnv(true);
      grbEnv.set("LogFile", "gbd.log");
      grbEnv.start();
      GRBModel model = GRBModel(grbEnv);
      auto varList = model.addVars((int)globalParticleNum, GRB_BINARY);

      // setup objective function
      GRBLinExpr objective;
      objective.addTerms(gatheredSensitivityMatrix.data(), varList,
                         globalParticleNum);

      model.setObjective(objective, GRB_MINIMIZE);

      // setup volume constraint
      std::vector<double> volumeConstraintWeight;
      volumeConstraintWeight.resize(globalParticleNum);
      for (int i = 0; i < globalParticleNum; i++) {
        volumeConstraintWeight[i] = 1.0;
      }
      GRBLinExpr volumeConstraint;
      volumeConstraint.addTerms(volumeConstraintWeight.data(), varList,
                                globalParticleNum);

      GRBLinExpr volumeFraction(volumeConstraintParticleNum_);

      model.addConstr(volumeConstraint, GRB_EQUAL, volumeFraction);

      // optimize
      model.optimize();

      for (int i = 0; i < globalParticleNum; i++) {
        resultingDensity[i] = varList[i].get(GRB_DoubleAttr_X);
      }

      for (int i = 0; i < localParticleNum; i++) {
        density_(i) = resultingDensity[i];
      }

      cutCost_ = model.get(GRB_DoubleAttr_ObjVal) +
                 adjustedObjFunc_[effectiveIndex_[0]];
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // distribute results
  for (int i = 1; i < mpiSize_; i++) {
    if (mpiRank_ == 0)
      MPI_Send(resultingDensity.data() + recvOffset_[i], recvCount_[i],
               MPI_DOUBLE, i, 0, MPI_COMM_WORLD);

    MPI_Status status;
    if (mpiRank_ == i)
      MPI_Recv(density_.data(), localParticleNum, MPI_DOUBLE, 0, 0,
               MPI_COMM_WORLD, &status);

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // broadcast resulting cost of the cut
  MPI_Bcast(&cutCost_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

TopologyOptimization::GeneralizedBendersDecomposition::
    GeneralizedBendersDecomposition() {
  // try {
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   if (mpiRank_ == 0)
  //     grbEnv_.set("LogFile", "gbd.log");

  //   grbEnv_.start();
  // } catch (GRBException e) {
  //   printf("Error code = %d\n", e.getErrorCode());
  //   printf("%s\n", e.getMessage().c_str());
  // } catch (...) {
  //   printf("Exception during optimization\n");
  // }
}

TopologyOptimization::GeneralizedBendersDecomposition::
    ~GeneralizedBendersDecomposition() {}

Void TopologyOptimization::GeneralizedBendersDecomposition::Init() {
  TopologyOptimization::Init();

  particleMgr_.Init();

  auto localSourceParticleNum = particleMgr_.GetLocalParticleNum();
  auto &sourceCoords = particleMgr_.GetParticleCoords();
  Kokkos::resize(volume_, localSourceParticleNum);
  Kokkos::resize(density_, localSourceParticleNum);
  Kokkos::resize(oldDensity_, localSourceParticleNum);
  Kokkos::resize(sensitivity_, localSourceParticleNum);

  int dimension = particleMgr_.GetDimension();

  auto &particleSize = particleMgr_.GetParticleSize();

  for (auto i = 0; i < localSourceParticleNum; i++) {
    density_(i) = 1.0;

    volume_(i) = pow(particleSize(i), dimension);
  }

  equationPtr_->SetKappa([=](const HostRealMatrix &coords,
                             const HostRealVector &spacing,
                             HostRealVector &kappa) {
    auto localTargetParticleNum = coords.extent(0);

    Geometry::Ghost ghost;
    ghost.Init(coords, spacing, sourceCoords, 8.0, particleMgr_.GetDimension());

    HostRealVector ghostDensity;
    HostRealMatrix ghostCoords;
    ghost.ApplyGhost(density_, ghostDensity);
    ghost.ApplyGhost(sourceCoords, ghostCoords);

    HostRealVector epsilon;
    HostIndexMatrix neighborLists;
    Kokkos::resize(epsilon, localTargetParticleNum);
    Kokkos::resize(neighborLists, localTargetParticleNum, 3);
    auto pointCloudSearch(Compadre::CreatePointCloudSearch(
        ghostCoords, particleMgr_.GetDimension()));

    pointCloudSearch.generate2DNeighborListsFromKNNSearch(
        false, coords, neighborLists, epsilon, 2, 1.0);

    for (int i = 0; i < localTargetParticleNum; i++) {
      int scaling = floor(epsilon(i) / spacing(i) * 1000 + 0.5) + 1;
      epsilon(i) = scaling * 1e-3 * spacing(i);
    }

    unsigned int minNeighborLists =
        1 + pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
                true, coords, neighborLists, epsilon, 0.0, 0.0);
    if (minNeighborLists > neighborLists.extent(1))
      Kokkos::resize(neighborLists, localTargetParticleNum, minNeighborLists);
    pointCloudSearch.generate2DNeighborListsFromRadiusSearch(
        false, coords, neighborLists, epsilon, 0.0, 0.0);

    for (auto i = 0; i < kappa.extent(0); i++)
      kappa(i) = ghostDensity(neighborLists(i, 1));
  });

  // Since sensitivity needs to be moved to the root, the offset of gather
  // operation should be calculated in advance.
  if (mpiRank_ == 0)
    recvCount_.resize(mpiSize_);
  int localParticleNum = particleMgr_.GetLocalParticleNum();
  MPI_Gather(&localParticleNum, 1, MPI_INT, recvCount_.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  if (mpiRank_ == 0) {
    recvOffset_.resize(mpiSize_);
    recvOffset_[0] = 0;
    for (int i = 1; i < mpiSize_; i++)
      recvOffset_[i] = recvOffset_[i - 1] + recvCount_[i - 1];
  }

  Kokkos::resize(optimalDensity_, localParticleNum);

  MPI_Barrier(MPI_COMM_WORLD);
}

Void TopologyOptimization::GeneralizedBendersDecomposition::Optimize() {
  iteration_ = 0;
  outerLoop_ = 0;
  unsigned int stage = 1;

  const unsigned int globalParticleNum = particleMgr_.GetGlobalParticleNum();
  const unsigned int localParticleNum = particleMgr_.GetLocalParticleNum();
  deltaParticleNum_ = std::floor(0.05 * globalParticleNum);
  volumeFractionParticleNum_ = std::floor(volumeFraction_ * globalParticleNum);
  volumeConstraintParticleNum_ = globalParticleNum;

  Scalar epsilon = 1e-3;

  auto &particleType = particleMgr_.GetParticleType();

  if (mpiRank_ == 0)
    printf("Start of GBD optimization\n");

  Scalar newVolume;
  Scalar localVolume;

  // outer loop of GBD
  while (stage < 3) {
    outerLoop_++;
    iteration_++;

    sensitivityMatrix_.clear();
    objFunc_.clear();
    adjustedObjFunc_.clear();

    Scalar Upper = 1e9;

    innerLoop_ = 0;

    if (stage == 1) {
      volumeConstraintParticleNum_ -= deltaParticleNum_;
      volumeConstraintParticleNum_ =
          std::max(volumeFractionParticleNum_, volumeConstraintParticleNum_);
      // volumeConstraintParticleNum_ = volumeFractionParticleNum_;

      sensitivityMatrix_.clear();
      objFunc_.clear();
      adjustedObjFunc_.clear();

      CalculateSensitivity();

      Output();

      effectiveIndex_.resize(1);
      effectiveIndex_[0] = objFunc_.size() - 1;

      MasterCut();

      for (unsigned int i = 0; i < localParticleNum; i++) {
        density_(i) = std::max(density_(i), 1e-3);
      }
    }

    if (stage == 2) {
      volumeConstraintParticleNum_ = volumeFractionParticleNum_;
      epsilon = 1e-5;
    }

    sensitivityMatrix_.clear();
    objFunc_.clear();
    adjustedObjFunc_.clear();

    // inner loop of GBD
    while (true) {
      iteration_++;
      innerLoop_++;
      if (mpiRank_ == 0)
        printf("GBD outer iteration: %d, inner iteration: %d, current volume "
               "fraction: %f, target volume fraction: %f\n",
               outerLoop_, innerLoop_,
               (double)(volumeConstraintParticleNum_) /
                   (double)(globalParticleNum),
               volumeFraction_);

      CalculateSensitivity();
      Output();

      if (Upper > objFunc_.back()) {
        Upper = objFunc_.back();

        for (auto i = 0; i < density_.extent(0); i++)
          optimalDensity_(i) = density_(i);
      }

      effectiveIndex_.clear();
      double currentObjFunc = objFunc_.back();
      for (int i = 0; i < objFunc_.size(); i++)
        if (objFunc_[i] <= currentObjFunc)
          effectiveIndex_.push_back(i);

      MasterCut();

      for (unsigned int i = 0; i < localParticleNum; i++) {
        density_(i) = std::max(density_(i), 1e-3);
      }

      if (mpiRank_ == 0)
        printf("Cost of the cut: %f, volume: %f, upper bound: %f, gap: %f%%\n",
               cutCost_,
               (double)(volumeConstraintParticleNum_) /
                   (double)(globalParticleNum),
               Upper, std::abs(cutCost_ - Upper) / Upper * 100);

      if (cutCost_ > Upper || std::abs(cutCost_ - Upper) / Upper < epsilon)
        break;
    }

    for (auto i = 0; i < density_.extent(0); i++)
      density_(i) = optimalDensity_(i);

    if (stage == 1 &&
        volumeConstraintParticleNum_ <= volumeFractionParticleNum_)
      stage = 2;
    else if (stage == 2) {
      stage = 3;

      sensitivityMatrix_.clear();
      objFunc_.clear();
      adjustedObjFunc_.clear();

      CalculateSensitivity();
      Output();

      if (mpiRank_ == 0) {
        printf("Resulting optimal objective function: %f\n", objFunc_.back());
      }
    }
  }
}