#include "gmls_solver.h"

using namespace std;
using namespace Compadre;

bool GMLS_Solver::NeedRefinement() {
  // prepare stage;
  static vector<int> &particleNum = __field.index.GetHandle("particle number");
  int &localParticleNum = particleNum[0];
  int &globalParticleNum = particleNum[1];

  GMLS *all_velocity =
      new GMLS(DivergenceFreeVectorTaylorPolynomial, VectorPointSample,
               __polynomialOrder, __dim, "SVD", "STANDARD");

  GMLS &velocityBasis = *all_velocity;

  static vector<vec3> &backgroundSourceCoord =
      __background.vector.GetHandle("source coord");
  static vector<int> &backgroundSourceIndex =
      __background.index.GetHandle("source index");
  static vector<vec3> &coord = __field.vector.GetHandle("coord");

  int numSourceCoords = backgroundSourceCoord.size();
  int numTargetCoords = coord.size();
  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
      "source coordinates", numSourceCoords, 3);
  Kokkos::View<double **>::HostMirror sourceCoords =
      Kokkos::create_mirror_view(sourceCoordsDevice);

  for (size_t i = 0; i < backgroundSourceCoord.size(); i++) {
    for (int j = 0; j < 3; j++) {
      sourceCoords(i, j) = backgroundSourceCoord[i][j];
    }
  }

  Kokkos::View<double **, Kokkos::DefaultExecutionSpace> targetCoordsDevice(
      "target coordinates", numTargetCoords, 3);
  Kokkos::View<double **>::HostMirror targetCoords =
      Kokkos::create_mirror_view(targetCoordsDevice);

  // create target coords
  vector<int> fluid2NeumannBoundary;
  int iNeumanBoundary = 0;
  for (int i = 0; i < localParticleNum; i++) {
    for (int j = 0; j < 3; j++) {
      targetCoords(i, j) = coord[i][j];
    }
  }

  Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);
  Kokkos::deep_copy(targetCoordsDevice, targetCoords);

  auto pointCloudSearch(CreatePointCloudSearch(sourceCoords, __dim));

  const int minNeighbors = Compadre::GMLS::getNP(__polynomialOrder, __dim);

  double epsilonMultiplier = 2.2;
  int estimatedUpperBoundNumberNeighbors =
      8 * pointCloudSearch.getEstimatedNumberNeighborsUpperBound(
              minNeighbors, __dim, epsilonMultiplier);

  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighborListsDevice(
      "neighbor lists", numTargetCoords, estimatedUpperBoundNumberNeighbors);
  Kokkos::View<int **>::HostMirror neighborLists =
      Kokkos::create_mirror_view(neighborListsDevice);

  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilonDevice(
      "h supports", numTargetCoords);
  Kokkos::View<double *>::HostMirror epsilon =
      Kokkos::create_mirror_view(epsilonDevice);

  pointCloudSearch.generateNeighborListsFromKNNSearch(
      false, targetCoords, neighborLists, epsilon, minNeighbors,
      epsilonMultiplier);

  Kokkos::deep_copy(neighborListsDevice, neighborLists);
  Kokkos::deep_copy(epsilonDevice, epsilon);

  velocityBasis.setProblemData(neighborListsDevice, sourceCoordsDevice,
                               targetCoordsDevice, epsilonDevice);

  vector<TargetOperation> velocityOperation(1);
  velocityOperation[0] = GradientOfVectorPointEvaluation;

  velocityBasis.addTargets(velocityOperation);

  velocityBasis.setWeightingType(WeightingFunctionType::Power);
  velocityBasis.setWeightingPower(2);

  velocityBasis.generateAlphas(1);

  vector<int> velocityGradientIndex(pow(__dim, 3));
  for (int i = 0; i < __dim; i++) {
    for (int j = 0; j < __dim; j++) {
      for (int k = 0; k < __dim; k++) {
        velocityGradientIndex[(i * __dim + j) * __dim + k] =
            velocityBasis.getAlphaColumnOffset(GradientOfVectorPointEvaluation,
                                               i, j, k, 0);
      }
    }
  }
  auto velocityNeighborListsLengths = velocityBasis.getNeighborListsLengths();
  auto velocityAlphas = velocityBasis.getAlphas();

  // communicate velocity field
  static vector<vec3> &velocity = __field.vector.GetHandle("fluid velocity");
  vector<double> recvVelocity;
  DataSwapAmongNeighbor(velocity, recvVelocity);
  vector<vec3> backgroundVelocity;

  for (int i = 0; i < localParticleNum; i++) {
    backgroundVelocity.push_back(velocity[i]);
  }

  static vector<int> &offset = __neighbor.index.GetHandle("offset");
  int neighborNum = pow(3, __dim);
  int totalNeighborParticleNum = offset[neighborNum];

  for (int i = 0; i < totalNeighborParticleNum; i++) {
    backgroundVelocity.push_back(vec3(
        recvVelocity[3 * i], recvVelocity[3 * i + 1], recvVelocity[3 * i + 2]));
  }

  // estimate stage
  vector<vector<double>> directVelocityGradient;
  vector<vector<double>> recoveredVelocityGradient;

  directVelocityGradient.resize(localParticleNum);
  recoveredVelocityGradient.resize(localParticleNum);

  const int gradientComponentNum = pow(__dim, 2);
  for (int i = 0; i < localParticleNum; i++) {
    directVelocityGradient[i].resize(gradientComponentNum);
    recoveredVelocityGradient[i].resize(gradientComponentNum);
    for (int axes1 = 0; axes1 < __dim; axes1++) {
      for (int axes2 = 0; axes2 < __dim; axes2++) {
        directVelocityGradient[i][axes1 * __dim + axes2] = 0.0;
        recoveredVelocityGradient[i][axes1 * __dim + axes2] = 0.0;
      }
    }

    for (int j = 0; j < velocityNeighborListsLengths(i); j++) {
      const int neighborParticleIndex = neighborLists(i, j + 1);
      for (int axes1 = 0; axes1 < __dim; axes1++) {
        for (int axes2 = 0; axes2 < __dim; axes2++) {
          for (int axes3 = 0; axes3 < __dim; axes3++) {
            const int velocityGradientAlphaIndex =
                velocityGradientIndex[(axes1 * __dim + axes2) * __dim + axes3];
            directVelocityGradient[i][axes1 * __dim + axes2] +=
                velocityAlphas(i, velocityGradientAlphaIndex, j) *
                backgroundVelocity[neighborParticleIndex][axes3];
          }
        }
      }
    }
  }

  vector<double> recvGradient;
  DataSwapAmongNeighbor(directVelocityGradient, recvGradient,
                        gradientComponentNum);
  vector<vector<double>> backgroundVelocityGradient;
  backgroundVelocityGradient.resize(localParticleNum +
                                    totalNeighborParticleNum);

  for (int i = 0; i < localParticleNum; i++) {
    for (int j = 0; j < gradientComponentNum; j++) {
      backgroundVelocityGradient[i].push_back(directVelocityGradient[i][j]);
    }
  }
  for (int i = 0; i < totalNeighborParticleNum; i++) {
    for (int j = 0; j < gradientComponentNum; j++) {
      backgroundVelocityGradient[localParticleNum + i].push_back(
          recvGradient[gradientComponentNum * i + j]);
    }
  }

  for (int i = 0; i < localParticleNum; i++) {
    for (int j = 0; j < velocityNeighborListsLengths(i); j++) {
      const int neighborParticleIndex = neighborLists(i, j + 1);
      for (int axes = 0; axes < gradientComponentNum; axes++) {
        recoveredVelocityGradient[i][axes] +=
            backgroundVelocityGradient[neighborParticleIndex][axes];
      }
    }

    for (int axes = 0; axes < gradientComponentNum; axes++) {
      recoveredVelocityGradient[i][axes] /= (neighborLists(i, 0) - 2);
    }
  }

  static std::vector<vec3> &particleSize = __field.vector.GetHandle("size");

  double localError = 0.0;
  double localVol = 0.0;
  double globalError;
  double globalVol;
  vector<double> error(localParticleNum);
  for (int i = 0; i < localParticleNum; i++) {
    error[i] = 0.0;
    for (int axes = 0; axes < gradientComponentNum; axes++) {
      error[i] += pow(
          recoveredVelocityGradient[i][axes] - directVelocityGradient[i][axes],
          2);
    }
    double vol =
        (__dim == 3)
            ? (particleSize[i][0] * particleSize[i][1] * particleSize[i][2])
            : (particleSize[i][0] * particleSize[i][1]);
    localVol += vol;
    localError += error[i] * vol;
  }

  MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localVol, &globalVol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  globalError /= globalVol;
  PetscPrintf(PETSC_COMM_WORLD, "Total error for gradient of velocity: %f\n",
              globalError);

  // mark stage
  if (__adaptive_step < 1) {
    double alpha = pow(0.9, __adaptive_step);
    vector<int> splitTag;
    for (int i = 0; i < localParticleNum; i++) {
      if (error[i] > alpha * globalError) {
        splitTag.push_back(i);
      }
    }
    int localSplitParticleNum = splitTag.size();
    int globalSplitParticleNum;
    MPI_Allreduce(&localSplitParticleNum, &globalSplitParticleNum, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Total number of split particle: %d\n",
                globalSplitParticleNum);

    // refine stage
    SplitParticle(splitTag);

    BuildNeighborList();
  }
  __adaptive_step++;

  if (__adaptive_step > 1) return false;

  return true;
}