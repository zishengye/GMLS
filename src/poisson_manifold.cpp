// #include "GMLS_solver.h"
// #include "manifold.h"
// #include "sparse_matrix.h"

// #define PI 3.1415926

// using namespace std;
// using namespace Compadre;

// void GMLS_Solver::PoissonEquationManifold() {
//   // create source and target coords
//   int numSourceCoords = __backgroundParticle.coord.size();
//   int numTargetCoords = __particle.X.size();
//   Kokkos::View<double **, Kokkos::DefaultExecutionSpace> sourceCoordsDevice(
//       "source coordinates", numSourceCoords, 3);
//   Kokkos::View<double **>::HostMirror sourceCoords =
//       Kokkos::create_mirror_view(sourceCoordsDevice);

//   Kokkos::View<double **, Kokkos::DefaultExecutionSpace> targetCoordsDevice(
//       "target coordinates", numTargetCoords, 3);
//   Kokkos::View<double **>::HostMirror targetCoords =
//       Kokkos::create_mirror_view(targetCoordsDevice);

//   for (size_t i = 0; i < __backgroundParticle.coord.size(); i++) {
//     for (int j = 0; j < 3; j++) {
//       sourceCoords(i, j) = __backgroundParticle.coord[i][j];
//     }
//   }

//   for (int i = 0; i < __particle.localParticleNum; i++) {
//     for (int j = 0; j < 3; j++) {
//       targetCoords(i, j) = __particle.X[i][j];
//     }
//   }

//   Kokkos::deep_copy(sourceCoordsDevice, sourceCoords);
//   Kokkos::deep_copy(targetCoordsDevice, targetCoords);

//   // neighbor search
//   auto pointCloudSearch(CreatePointCloudSearch(sourceCoords));

//   const int minNeighbors = Compadre::GMLS::getNP(__polynomialOrder, __dim);

//   double epsilonMultiplier;
//   epsilonMultiplier = 1.5;
//   int estimatedUpperBoundNumberNeighbors =
//       PI * pow(__cutoffDistance / __particleSize0[0], __dim) + 1;

//   Kokkos::View<int **, Kokkos::DefaultExecutionSpace> neighborListsDevice(
//       "neighbor lists", numTargetCoords, estimatedUpperBoundNumberNeighbors);
//   Kokkos::View<int **>::HostMirror neighborLists =
//       Kokkos::create_mirror_view(neighborListsDevice);

//   Kokkos::View<double *, Kokkos::DefaultExecutionSpace> epsilonDevice(
//       "h supports", numTargetCoords);
//   Kokkos::View<double *>::HostMirror epsilon =
//       Kokkos::create_mirror_view(epsilonDevice);

//   pointCloudSearch.generateNeighborListsFromKNNSearch(
//       true, targetCoords, neighborLists, epsilon, minNeighbors, __dim,
//       epsilonMultiplier, NULL, __cutoffDistance);

//   Kokkos::deep_copy(neighborListsDevice, neighborLists);
//   Kokkos::deep_copy(epsilonDevice, epsilon);

//   PetscPrintf(PETSC_COMM_WORLD, "\nSetup neighbor lists\n");

//   __eq.rhsScalar.resize(__particle.localParticleNum);
//   __eq.xScalar.resize(__particle.localParticleNum);
//   __particle.pressure.resize(__particle.localParticleNum);

//   if (__particle.scalarBasis != nullptr) {
//     delete __particle.scalarBasis;
//   }
//   __particle.scalarBasis =
//       new GMLS(ScalarTaylorPolynomial,
//                StaggeredEdgeAnalyticGradientIntegralSample,
//                __polynomialOrder,
//                __dim, "QR", "MANIFOLD", "NO_CONSTRAINT", __manifoldOrder);

//   GMLS &pressureBasis = *__particle.scalarBasis;

//   pressureBasis.setProblemData(neighborListsDevice, sourceCoordsDevice,
//                                targetCoordsDevice, epsilonDevice);

//   std::vector<TargetOperation> pressureOperation(1);
//   pressureOperation[0] = LaplacianOfScalarPointEvaluation;

//   pressureBasis.addTargets(pressureOperation);

//   pressureBasis.setWeightingType(WeightingFunctionType::Power);
//   pressureBasis.setWeightingPower(2);

//   pressureBasis.generateAlphas();

//   auto pressureAlphas = pressureBasis.getAlphas();

//   const int pressureLaplacianIndex = pressureBasis.getAlphaColumnOffset(
//       LaplacianOfScalarPointEvaluation, 0, 0, 0, 0);

//   auto pressureNeighborListsLengths =
//   pressureBasis.getNeighborListsLengths();

//   PetscPrintf(PETSC_COMM_WORLD, "\nGenerating Poisson Matrix...\n");

//   PetscSparseMatrix A(__particle.localParticleNum,
//                       __particle.globalParticleNum);
//   for (int i = 0; i < __particle.localParticleNum; i++) {
//     const int currentParticleLocalIndex = i;
//     const int currentParticleGlobalIndex = __particle.globalIndex[i];
//     if (__particle.particleType[i] != 0) {
//       A.increment(currentParticleLocalIndex,
//       currentParticleGlobalIndex, 1.0);
//     } else {
//       for (int j = 1; j < pressureNeighborListsLengths(i); j++) {
//         const int neighborParticleIndex =
//             __backgroundParticle.index[neighborLists(i, j + 1)];
//         A.increment(currentParticleLocalIndex, neighborParticleIndex,
//                     -pressureAlphas(i, pressureLaplacianIndex, j));
//         A.increment(currentParticleLocalIndex, currentParticleGlobalIndex,
//                     pressureAlphas(i, pressureLaplacianIndex, j));
//       }
//     }
//   }

//   MPI_Barrier(MPI_COMM_WORLD);
//   A.FinalAssemble();

//   PetscPrintf(PETSC_COMM_WORLD, "\nPoisson Matrix Assembled\n");

//   __eq.rhsScalar.resize(__particle.localParticleNum);
//   __eq.xScalar.resize(__particle.localParticleNum);
//   __particle.pressure.resize(__particle.localParticleNum);

//   for (int i = 0; i < __particle.localParticleNum; i++) {
//     if (__particle.particleType[i] == 0) {
//       double xScalar = __particle.X[i][0];
//       double y = __particle.X[i][1];
//       double g = 1 + 4 * xScalar * xScalar;
//       __eq.rhsScalar[i] = 2.0 / g - 8.0 * xScalar * xScalar / g / g + 2.0;
//     } else {
//       double xScalar = __particle.X[i][0];
//       double y = __particle.X[i][1];
//       __eq.rhsScalar[i] = pow(xScalar, 2) + pow(y, 2);
//     }
//   }

//   A.Solve(__eq.rhsScalar, __eq.xScalar);

//   for (int i = 0; i < __particle.localParticleNum; i++) {
//     __particle.pressure[i] = __eq.xScalar[i];
//   }
// }