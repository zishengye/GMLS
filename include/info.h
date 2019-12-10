#pragma once

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "vec3.h"

struct ParticleInfo {
  // geometry info
  std::vector<vec3> X;
  std::vector<vec3> X_origin;
  std::vector<int> particleType;
  std::vector<vec3> particleSize;
  std::vector<vec3> normal;
  std::vector<int> globalIndex;
  std::vector<double> vol;
  int localParticleNum;
  int globalParticleNum;
  std::vector<int> particleOffset;
  std::vector<double> d;
  std::vector<size_t> attachedRigidBodyIndex;

  // physical info
  std::vector<double> pressure;
  std::vector<double> velocity;
  std::vector<double> us;
  std::vector<double> us_old;
  std::vector<vec3> flux;

  // GMLS info
  // Compadre::GMLS *scalarBasis;
  // Compadre::GMLS *vectorBasis;
  // Compadre::GMLS *scalarNeumannBoundaryBasis;
  // Compadre::GMLS *vectorNeumannBoundaryBasis;
};

struct RigidBodyInfo {
  std::vector<vec3> Ci_X;
  std::vector<vec3> Ci_Theta;
  std::vector<vec3> Ci_V;
  std::vector<vec3> Ci_Omega;
  std::vector<vec3> Ci_F;
  std::vector<vec3> Ci_Torque;
  std::vector<double> Ci_R;
  std::vector<int> type;
};

struct EquationInfo {
  std::vector<vec3> F;
  std::vector<vec3> gradP;
  std::vector<double> dP;
  std::vector<double> rhsScalar;
  std::vector<double> rhsVector;
  std::vector<double> xScalar;
  std::vector<double> xVector;
};

struct neighborListInfo {
  std::vector<vec3> coord;
  std::vector<int> index;

  void clear() {
    coord.clear();
    index.clear();
  }
};