#include "gmls_solver.h"

using namespace std;
using namespace Compadre;

bool GMLS_Solver::NeedRefinement() {
  // // prepare stage;
  // static vector<int> &particleNum = __field.index.GetHandle("particle
  // number"); int &localParticleNum = particleNum[0]; int &globalParticleNum =
  // particleNum[1];

  // static GMLS &velocityBasis = __gmls.GetHandle("velocity basis");
  // vector<int> velocityGradientIndex(pow(__dim, 3));
  // for (int i = 0; i < __dim; i++) {
  //   for (int j = 0; j < __dim; j++) {
  //     for (int k = 0; k < __dim; k++) {
  //       velocityGradientIndex[(i * __dim + j) * __dim + k] =
  //           velocityBasis.getAlphaColumnOffset(GradientOfVectorPointEvaluation,
  //                                              i, j, k, 0);
  //     }
  //   }
  // }
  // auto velocityNeighborListsLengths =
  // velocityBasis.getNeighborListsLengths(); auto neighborLists =
  // velocityBasis.getNeighborLists(); auto velocityAlphas =
  // velocityBasis.getAlphas();

  // // communicate velocity field
  // static vector<vec3> &velocity = __field.vector.GetHandle("fluid velocity");
  // vector<double> recvVelocity;
  // DataSwapAmongNeighbor(velocity, recvVelocity);
  // vector<vec3> backgroundVelocity;

  // for (int i = 0; i < localParticleNum; i++) {
  //   backgroundVelocity.push_back(velocity[i]);
  // }

  // static vector<int> &offset = __neighbor.index.GetHandle("offset");
  // int neighborNum = pow(3, __dim);
  // int totalNeighborParticleNum = offset[neighborNum];

  // for (int i = 0; i < totalNeighborParticleNum; i++) {
  //   backgroundVelocity.push_back(vec3(
  //       recvVelocity[3 * i], recvVelocity[3 * i + 1], recvVelocity[3 * i +
  //       2]));
  // }

  // // estimate stage
  // vector<vector<double>> directVelocityGradient;
  // vector<vector<double>> recoveredVelocityGradient;

  // directVelocityGradient.resize(localParticleNum);
  // recoveredVelocityGradient.resize(localParticleNum);

  // static vector<int> &backgroundSourceIndex =
  //     __background.index.GetHandle("source index");

  // const int gradientComponentNum = pow(__dim, 2);
  // for (int i = 0; i < localParticleNum; i++) {
  //   directVelocityGradient[i].resize(gradientComponentNum);
  //   recoveredVelocityGradient[i].resize(gradientComponentNum);
  //   for (int axes1 = 0; axes1 < __dim; axes1++) {
  //     for (int axes2 = 0; axes2 < __dim; axes2++) {
  //       directVelocityGradient[i][axes1 * __dim + axes2] = 0.0;
  //       recoveredVelocityGradient[i][axes1 * __dim + axes2] = 0.0;
  //     }
  //   }

  //   for (int j = 0; j < velocityNeighborListsLengths(i); j++) {
  //     const int neighborParticleIndex = neighborLists(i, j + 1);
  //     for (int axes1 = 0; axes1 < __dim; axes1++) {
  //       for (int axes2 = 0; axes2 < __dim; axes2++) {
  //         for (int axes3 = 0; axes3 < __dim; axes3++) {
  //           const int velocityGradientAlphaIndex =
  //               velocityGradientIndex[(axes1 * __dim + axes2) * __dim +
  //               axes3];
  //           directVelocityGradient[i][axes1 * __dim + axes2] +=
  //               velocityAlphas(i, velocityGradientAlphaIndex, j) *
  //               backgroundVelocity[neighborParticleIndex][axes3];
  //         }
  //       }
  //     }
  //   }
  // }

  // vector<double> recvGradient;
  // DataSwapAmongNeighbor(directVelocityGradient, recvGradient,
  //                       gradientComponentNum);
  // vector<vector<double>> backgroundVelocityGradient;
  // backgroundVelocityGradient.resize(localParticleNum +
  //                                   totalNeighborParticleNum);

  // for (int i = 0; i < localParticleNum; i++) {
  //   for (int j = 0; j < gradientComponentNum; j++) {
  //     backgroundVelocityGradient[i].push_back(directVelocityGradient[i][j]);
  //   }
  // }
  // for (int i = 0; i < totalNeighborParticleNum; i++) {
  //   for (int j = 0; j < gradientComponentNum; j++) {
  //     backgroundVelocityGradient[localParticleNum + i].push_back(
  //         recvGradient[gradientComponentNum * i + j]);
  //   }
  // }

  // for (int i = 0; i < localParticleNum; i++) {
  //   for (int j = 0; j < velocityNeighborListsLengths(i); j++) {
  //     const int neighborParticleIndex = neighborLists(i, j + 1);
  //     for (int axes = 0; axes < gradientComponentNum; axes++) {
  //       recoveredVelocityGradient[i][axes] +=
  //           backgroundVelocityGradient[neighborParticleIndex][axes];
  //     }
  //   }

  //   for (int axes = 0; axes < gradientComponentNum; axes++) {
  //     recoveredVelocityGradient[i][axes] /= (neighborLists(i, 0) - 2);
  //   }
  // }

  // static std::vector<vec3> &particleSize = __field.vector.GetHandle("size");

  // double localError = 0.0;
  // double localVol = 0.0;
  // double globalError;
  // double globalVol;
  // vector<double> error(localParticleNum);
  // for (int i = 0; i < localParticleNum; i++) {
  //   error[i] = 0.0;
  //   for (int axes = 0; axes < gradientComponentNum; axes++) {
  //     error[i] += pow(
  //         recoveredVelocityGradient[i][axes] -
  //         directVelocityGradient[i][axes], 2);
  //   }
  //   double vol =
  //       (__dim == 3)
  //           ? (particleSize[i][0] * particleSize[i][1] * particleSize[i][2])
  //           : (particleSize[i][0] * particleSize[i][1]);
  //   localVol += vol;
  //   localError += error[i] * vol;
  // }

  // MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM,
  //               MPI_COMM_WORLD);
  // MPI_Allreduce(&localVol, &globalVol, 1, MPI_DOUBLE, MPI_SUM,
  // MPI_COMM_WORLD); globalError /= globalVol; PetscPrintf(PETSC_COMM_WORLD,
  // "Total error for gradient of velocity: %f\n",
  //             globalError);

  // // mark stage
  // if (__adaptive_step < 0) {
  //   double alpha = pow(0.95, __adaptive_step);
  //   vector<int> splitTag;
  //   for (int i = 0; i < localParticleNum; i++) {
  //     if (error[i] > alpha * globalError) {
  //       splitTag.push_back(i);
  //     }
  //   }
  //   int localSplitParticleNum = splitTag.size();
  //   int globalSplitParticleNum;
  //   MPI_Allreduce(&localSplitParticleNum, &globalSplitParticleNum, 1,
  //   MPI_INT,
  //                 MPI_SUM, MPI_COMM_WORLD);
  //   PetscPrintf(PETSC_COMM_WORLD, "Total number of split particle: %d\n",
  //               globalSplitParticleNum);

  //   // refine stage
  //   SplitParticle(splitTag);

  //   BuildNeighborList();
  // }

  // if (__adaptive_step >= 0) return false;

  // __adaptive_step++;

  // return true;

  return false;
}