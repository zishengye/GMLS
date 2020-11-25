#include <limits>

#include "DivergenceFree.h"
#include "gmls_solver.h"

using namespace std;
using namespace Compadre;

double Wab(double r, double h) {
  int p = 4;
  return pow(1.0 - abs(r / h), p) * double((1.0 - abs(r / h)) > 0.0);
}

struct ErrorComb {
  double error;
  int rank;
};

bool pairCompare(const std::pair<int, double> &firstElem,
                 const std::pair<int, double> &secondElem) {
  return firstElem.second > secondElem.second;
}

bool GMLS_Solver::NeedRefinement() {
  if (__adaptiveRefinement != 0) {
    // prepare stage

    MPI_Barrier(MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "\nstart of adaptive refinement\n");

    static vector<int> &particleNum =
        __field.index.GetHandle("particle number");
    int &localParticleNum = particleNum[0];
    int &globalParticleNum = particleNum[1];

    GMLS &pressureBasis = *__gmls.GetHandle("pressure basis");
    GMLS &velocityBasis = *__gmls.GetHandle("velocity basis");

    static auto &backgroundSourceCoord =
        __background.vector.GetHandle("source coord");
    static auto &coord = __field.vector.GetHandle("coord");

    static auto &volume = __field.scalar.GetHandle("volume");
    static auto &adaptive_level = __field.index.GetHandle("adaptive level");

    static vector<int> &offset = __neighbor.index.GetHandle("recv offset");
    int neighborNum = pow(3, __dim);
    int totalNeighborParticleNum = offset[neighborNum];

    // communicate epsilon
    vector<double> recvEpsilon;
    DataSwapAmongNeighbor(__epsilon, recvEpsilon);

    vector<double> backgroundEpsilon;
    for (int i = 0; i < localParticleNum; i++) {
      backgroundEpsilon.push_back(__epsilon[i]);
    }

    for (int i = 0; i < totalNeighborParticleNum; i++) {
      backgroundEpsilon.push_back(recvEpsilon[i]);
    }

    vector<double> recvVolume;
    DataSwapAmongNeighbor(volume, recvVolume);
    vector<double> backgroundVolume;

    backgroundVolume.insert(backgroundVolume.end(), volume.begin(),
                            volume.end());
    backgroundVolume.insert(backgroundVolume.end(), recvVolume.begin(),
                            recvVolume.end());

    double localError = 0.0;
    double localVol = 0.0;
    double localDirectGradientNorm = 0.0;
    double globalDirectGradientNorm;
    double globalError;
    double globalVol;

    auto &error = __field.scalar.GetHandle("error");
    error.resize(localParticleNum);

    vector<int> velocityGradientIndex(pow(__dim, 3));
    for (int i = 0; i < __dim; i++) {
      for (int j = 0; j < __dim; j++) {
        for (int k = 0; k < __dim; k++) {
          velocityGradientIndex[(i * __dim + j) * __dim + k] =
              velocityBasis.getAlphaColumnOffset(
                  GradientOfVectorPointEvaluation, i, j, k, 0);
        }
      }
    }

    auto velocityNeighborListsLengths = velocityBasis.getNeighborListsLengths();
    auto velocityAlphas = velocityBasis.getAlphas();
    auto neighborLists = velocityBasis.getNeighborLists();

    // communicate velocity field
    static vector<vec3> &velocity = __field.vector.GetHandle("fluid velocity");
    vector<vec3> recvVelocity;
    DataSwapAmongNeighbor(velocity, recvVelocity);
    vector<vec3> backgroundVelocity;

    for (int i = 0; i < localParticleNum; i++) {
      backgroundVelocity.push_back(velocity[i]);
    }

    for (int i = 0; i < totalNeighborParticleNum; i++) {
      backgroundVelocity.push_back(recvVelocity[i]);
    }

    // communicate coeffients
    Kokkos::View<double **, Kokkos::DefaultExecutionSpace>
        backgroundVelocityDevice("background velocity",
                                 backgroundVelocity.size(), 3);
    Kokkos::View<double **>::HostMirror backgroundVelocityHost =
        Kokkos::create_mirror_view(backgroundVelocityDevice);

    for (size_t i = 0; i < backgroundVelocity.size(); i++) {
      backgroundVelocityHost(i, 0) = backgroundVelocity[i][0];
      backgroundVelocityHost(i, 1) = backgroundVelocity[i][1];
      backgroundVelocityHost(i, 2) = backgroundVelocity[i][2];
    }

    Kokkos::deep_copy(backgroundVelocityHost, backgroundVelocityDevice);

    Evaluator velocityEvaluator(&velocityBasis);

    auto coefficients =
        velocityEvaluator
            .applyFullPolynomialCoefficientsBasisToDataAllComponents<
                double **, Kokkos::HostSpace>(backgroundVelocityDevice);

    auto gradient =
        velocityEvaluator.applyAlphasToDataAllComponentsAllTargetSites<
            double **, Kokkos::HostSpace>(backgroundVelocityDevice,
                                          GradientOfVectorPointEvaluation);

    auto coefficientsSize = velocityBasis.getPolynomialCoefficientsSize();

    vector<vector<double>> coefficientsChunk(localParticleNum);
    for (int i = 0; i < localParticleNum; i++) {
      coefficientsChunk[i].resize(coefficientsSize);
      for (int j = 0; j < coefficientsSize; j++) {
        coefficientsChunk[i][j] = coefficients(i, j);
      }
    }

    vector<vector<double>> recvCoefficientsChunk;

    DataSwapAmongNeighbor(coefficientsChunk, recvCoefficientsChunk,
                          coefficientsSize);

    vector<vector<double>> backgroundCoefficients;
    for (int i = 0; i < localParticleNum; i++) {
      backgroundCoefficients.push_back(coefficientsChunk[i]);
    }

    for (int i = 0; i < totalNeighborParticleNum; i++) {
      backgroundCoefficients.push_back(recvCoefficientsChunk[i]);
    }

    // estimate stage
    vector<vector<double>> recoveredVelocityGradient;
    recoveredVelocityGradient.resize(localParticleNum);
    const int gradientComponentNum = pow(__dim, 2);
    for (int i = 0; i < localParticleNum; i++) {
      recoveredVelocityGradient[i].resize(gradientComponentNum);
      for (int axes1 = 0; axes1 < __dim; axes1++) {
        for (int axes2 = 0; axes2 < __dim; axes2++) {
          recoveredVelocityGradient[i][axes1 * __dim + axes2] = 0.0;
        }
      }
    }

    for (int i = 0; i < localParticleNum; i++) {
      for (int j = 0; j < neighborLists(i, 0); j++) {
        const int neighborParticleIndex = neighborLists(i, j + 1);

        vec3 dX = coord[i] - backgroundSourceCoord[neighborParticleIndex];
        for (int axes1 = 0; axes1 < __dim; axes1++) {
          for (int axes2 = 0; axes2 < __dim; axes2++) {
            if (__dim == 2)
              recoveredVelocityGradient[i][axes1 * __dim + axes2] +=
                  calDivFreeBasisGrad(
                      axes1, axes2, dX[0], dX[1], __polynomialOrder,
                      backgroundEpsilon[neighborParticleIndex],
                      backgroundCoefficients[neighborParticleIndex]);
            if (__dim == 3)
              recoveredVelocityGradient[i][axes1 * __dim + axes2] +=
                  calDivFreeBasisGrad(
                      axes1, axes2, dX[0], dX[1], dX[2], __polynomialOrder,
                      backgroundEpsilon[neighborParticleIndex],
                      backgroundCoefficients[neighborParticleIndex]);
          }
        }
      }

      for (int axes1 = 0; axes1 < __dim; axes1++) {
        for (int axes2 = 0; axes2 < __dim; axes2++) {
          recoveredVelocityGradient[i][axes1 * __dim + axes2] /=
              neighborLists(i, 0);
        }
      }
    }

    vector<vector<double>> recvRecoveredVelocityGradient;
    DataSwapAmongNeighbor(recoveredVelocityGradient,
                          recvRecoveredVelocityGradient, gradientComponentNum);

    vector<vector<double>> backgroundRecoveredVelocityGradient;

    for (int i = 0; i < localParticleNum; i++) {
      backgroundRecoveredVelocityGradient.push_back(
          recoveredVelocityGradient[i]);
    }
    for (int i = 0; i < totalNeighborParticleNum; i++) {
      backgroundRecoveredVelocityGradient.push_back(
          recvRecoveredVelocityGradient[i]);
    }

    for (int i = 0; i < localParticleNum; i++) {
      error[i] = 0.0;
      vector<double> reconstructedVelocityGradient(gradientComponentNum);
      double totalNeighborVol = 0.0;
      // loop over all neighbors
      for (int j = 0; j < velocityNeighborListsLengths(i); j++) {
        const int neighborParticleIndex = neighborLists(i, j + 1);

        vec3 dX = backgroundSourceCoord[neighborParticleIndex] - coord[i];

        if (dX.mag() < __epsilon[i]) {
          totalNeighborVol += backgroundVolume[neighborParticleIndex];
          for (int axes1 = 0; axes1 < __dim; axes1++) {
            for (int axes2 = 0; axes2 < __dim; axes2++) {
              if (__dim == 2)
                reconstructedVelocityGradient[axes1 * __dim + axes2] =
                    calDivFreeBasisGrad(axes1, axes2, dX[0], dX[1],
                                        __polynomialOrder, backgroundEpsilon[i],
                                        backgroundCoefficients[i]);
              if (__dim == 3)
                reconstructedVelocityGradient[axes1 * __dim + axes2] =
                    calDivFreeBasisGrad(axes1, axes2, dX[0], dX[1], dX[2],
                                        __polynomialOrder, backgroundEpsilon[i],
                                        backgroundCoefficients[i]);
            }
          }

          for (int axes1 = 0; axes1 < __dim; axes1++) {
            for (int axes2 = axes1; axes2 < __dim; axes2++) {
              if (axes1 == axes2)
                error[i] +=
                    pow(reconstructedVelocityGradient[axes1 * __dim + axes2] -
                            backgroundRecoveredVelocityGradient
                                [neighborParticleIndex][axes1 * __dim + axes2],
                        2) *
                    backgroundVolume[neighborParticleIndex];
              else {
                error[i] +=
                    pow(0.5 * (reconstructedVelocityGradient[axes1 * __dim +
                                                             axes2] -
                               backgroundRecoveredVelocityGradient
                                   [neighborParticleIndex]
                                   [axes1 * __dim + axes2] +
                               reconstructedVelocityGradient[axes2 * __dim +
                                                             axes1] -
                               backgroundRecoveredVelocityGradient
                                   [neighborParticleIndex]
                                   [axes2 * __dim + axes1]),
                        2) *
                    backgroundVolume[neighborParticleIndex];
              }
            }
          }
        }
      }

      error[i] = error[i] / totalNeighborVol;
      localError += error[i] * volume[i];

      for (int axes1 = 0; axes1 < __dim; axes1++) {
        for (int axes2 = axes1; axes2 < __dim; axes2++) {
          if (axes1 == axes2)
            localDirectGradientNorm +=
                pow(gradient(i, axes1 * __dim + axes2), 2) * volume[i];
          else {
            localDirectGradientNorm +=
                pow(0.5 * (gradient(i, axes1 * __dim + axes2) +
                           gradient(i, axes2 * __dim + axes1)),
                    2) *
                volume[i];
          }
        }
      }
    }

    for (int ite = 0; ite < 1; ite++) {
      vector<double> recvError;
      DataSwapAmongNeighbor(error, recvError);

      vector<double> backgroundError;
      backgroundError = error;
      backgroundError.insert(backgroundError.end(), recvError.begin(),
                             recvError.end());

      for (int i = 0; i < localParticleNum; i++) {
        error[i] = 0.0;
        double totalNeighborVol = 0.0;
        for (int j = 0; j < velocityNeighborListsLengths(i); j++) {
          const int neighborParticleIndex = neighborLists(i, j + 1);

          vec3 dX = backgroundSourceCoord[neighborParticleIndex] - coord[i];

          if (dX.mag() < __epsilon[i]) {
            double Wabij = Wab(dX.mag(), __epsilon[i]);

            error[i] += backgroundError[neighborParticleIndex] *
                        backgroundVolume[neighborParticleIndex] * Wabij;
            totalNeighborVol += backgroundVolume[neighborParticleIndex] * Wabij;
          }
        }
        error[i] /= totalNeighborVol;
      }
    }

    for (int i = 0; i < localParticleNum; i++) {
      error[i] *= volume[i];
    }

    // if (__adaptive_base_field == "Pressure") {
    //   vector<int> pressureGradientIndex;
    //   for (int i = 0; i < __dim; i++)
    //     pressureGradientIndex.push_back(pressureBasis.getAlphaColumnOffset(
    //         GradientOfScalarPointEvaluation, i, 0, 0, 0));

    //   auto pressureAlphas = pressureBasis.getAlphas();
    //   auto neighborLists = pressureBasis.getNeighborLists();

    //   auto &pressure = __field.scalar.GetHandle("fluid pressure");
    //   vector<double> recvPressure;
    //   DataSwapAmongNeighbor(pressure, recvPressure);
    //   vector<double> backgroundPressure = pressure;
    //   backgroundPressure.insert(backgroundPressure.end(),
    //   recvPressure.begin(),
    //                             recvPressure.end());

    //   Kokkos::View<double *, Kokkos::DefaultExecutionSpace>
    //       backgroundPressureDevice("background pressure",
    //                                backgroundPressure.size());
    //   Kokkos::View<double *>::HostMirror backgroundPressureHost =
    //       Kokkos::create_mirror_view(backgroundPressureDevice);

    //   for (size_t i = 0; i < backgroundPressure.size(); i++) {
    //     backgroundPressureHost(i) = backgroundPressure[i];
    //   }

    //   Kokkos::deep_copy(backgroundPressureHost, backgroundPressureDevice);

    //   Evaluator pressureEvaluator(&pressureBasis);

    //   auto coefficients =
    //       pressureEvaluator
    //           .applyFullPolynomialCoefficientsBasisToDataAllComponents<
    //               double **, Kokkos::HostSpace>(backgroundPressureDevice);

    //   auto gradient =
    //       pressureEvaluator.applyAlphasToDataAllComponentsAllTargetSites<
    //           double **, Kokkos::HostSpace>(backgroundPressureDevice,
    //                                         GradientOfScalarPointEvaluation);

    //   auto coefficientsSize =
    //   pressureBasis.getPolynomialCoefficientsSize();

    //   vector<vector<double>> coefficientsChunk(localParticleNum);
    //   for (int i = 0; i < localParticleNum; i++) {
    //     coefficientsChunk[i].resize(coefficientsSize);
    //     for (int j = 0; j < coefficientsSize; j++) {
    //       coefficientsChunk[i][j] = coefficients(i, j);
    //     }
    //   }

    //   vector<vector<double>> recvCoefficientsChunk;

    //   DataSwapAmongNeighbor(coefficientsChunk, recvCoefficientsChunk,
    //                         coefficientsSize);

    //   vector<vector<double>> backgroundCoefficients;
    //   for (int i = 0; i < localParticleNum; i++) {
    //     backgroundCoefficients.push_back(coefficientsChunk[i]);
    //   }

    //   for (int i = 0; i < totalNeighborParticleNum; i++) {
    //     backgroundCoefficients.push_back(recvCoefficientsChunk[i]);
    //   }

    //   // estimate stage
    //   vector<vec3> recoveredPressureGradient;
    //   recoveredPressureGradient.resize(localParticleNum);

    //   for (int i = 0; i < localParticleNum; i++) {
    //     for (int j = 0; j < neighborLists(i, 0); j++) {
    //       const int neighborParticleIndex = neighborLists(i, j + 1);

    //       vec3 dX = coord[i] -
    //       backgroundSourceCoord[neighborParticleIndex]; for (int axes1 = 0;
    //       axes1 < __dim; axes1++) {
    //         if (__dim == 2)
    //           recoveredPressureGradient[i][axes1] +=
    //               calStaggeredScalarGrad(
    //                   axes1, dX[0], dX[1], __polynomialOrder,
    //                   backgroundEpsilon[neighborParticleIndex],
    //                   backgroundCoefficients[neighborParticleIndex]) /
    //               neighborLists(i, 0);
    //       }
    //     }
    //   }
    // }

    MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localDirectGradientNorm, &globalDirectGradientNorm, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    globalError /= globalDirectGradientNorm;
    globalError = sqrt(globalError);
    PetscPrintf(
        PETSC_COMM_WORLD,
        "Total error for gradient of velocity: %f, with tolerance: %f\n",
        globalError, __adaptiveRefinementTolerance);

    if (globalError < __adaptiveRefinementTolerance ||
        __adaptive_step >= __maxAdaptiveLevel)
      return false;

    // mark stage
    double alpha = 0.9;

    vector<pair<int, double>> chopper;
    pair<int, double> toAdd;

    localError = 0.0;
    for (int i = 0; i < localParticleNum; i++) {
      toAdd = pair<int, double>(i, error[i]);
      chopper.push_back(toAdd);
      localError += error[i];
    }
    MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    std::sort(chopper.begin(), chopper.end(), pairCompare);

    vector<int> splitTag;
    vector<int> splitCandidateTag;

    // parallel selection
    int split_max_index = 0;

    double error_max, error_min, current_error_split, next_error;
    error_max = chopper[0].second;
    error_min = chopper[localParticleNum - 1].second;

    MPI_Allreduce(MPI_IN_PLACE, &error_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &error_min, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);

    current_error_split = (error_max + error_min) / 2.0;
    bool selection_finished = false;
    while (!selection_finished) {
      int ite = 0;
      double error_sum = 0.0;
      while (ite < localParticleNum) {
        if (chopper[ite].second > current_error_split) {
          error_sum += chopper[ite].second;
          next_error = error_min;
          ite++;
        } else {
          next_error = chopper[ite].second;
          break;
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &error_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &next_error, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);

      if ((error_sum < alpha * globalError) &&
          (error_sum + next_error >= alpha * globalError)) {
        selection_finished = true;
        split_max_index = ite;
      } else if (error_sum < alpha * globalError) {
        error_max = current_error_split;
        current_error_split = (error_min + error_max) / 2.0;
      } else {
        error_min = current_error_split;
        current_error_split = (error_min + error_max) / 2.0;
      }
    }

    splitCandidateTag.clear();
    for (int i = 0; i < split_max_index; i++) {
      splitCandidateTag.push_back(chopper[i].first);
    }

    vector<int> recvAdaptiveLevel, backgroundAdaptiveLevel;
    DataSwapAmongNeighbor(adaptive_level, recvAdaptiveLevel);

    backgroundAdaptiveLevel.insert(backgroundAdaptiveLevel.end(),
                                   adaptive_level.begin(),
                                   adaptive_level.end());
    backgroundAdaptiveLevel.insert(backgroundAdaptiveLevel.end(),
                                   recvAdaptiveLevel.begin(),
                                   recvAdaptiveLevel.end());

    for (int i = 0; i < splitCandidateTag.size(); i++) {
      bool isAdd = true;
      int index = splitCandidateTag[i];
      // for (int j = 0; j < velocityNeighborListsLengths(index); j++) {
      //   const int neighborParticleIndex = neighborLists(index, j + 1);
      //   if (backgroundAdaptiveLevel[neighborParticleIndex] <
      //       adaptive_level[index]) {
      //     isAdd = false;
      //   }
      // }
      if (isAdd) {
        splitTag.push_back(splitCandidateTag[i]);
      }
    }

    for (int i = 0; i < localParticleNum; i++) {
      error[i] = sqrt(error[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (__writeData)
      WriteDataAdaptiveStep();

    MPI_Barrier(MPI_COMM_WORLD);

    int localSplitParticleNum = splitTag.size();
    int globalSplitParticleNum;
    MPI_Allreduce(&localSplitParticleNum, &globalSplitParticleNum, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    __adaptive_step++;

    // refine stage
    SplitParticle(splitTag);

    BuildNeighborList();
  } else {
    return false;
  }

  return true;
}