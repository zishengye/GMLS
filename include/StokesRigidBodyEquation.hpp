#ifndef _StokesRigidBodyEquation_Hpp_
#define _StokesRigidBodyEquation_Hpp_

#include "StokesEquation.hpp"

class StokesRigidBodyEquation : public StokesEquation {
protected:
  virtual void InitLinearSystem();
  virtual void ConstructLinearSystem();
  virtual void ConstructRhs();

  virtual void SolveEquation();
  virtual void CalculateError();

  virtual void Output();

  std::function<double(const unsigned int, const unsigned int)>
      rigidBodyVelocity_;
  std::function<double(const unsigned int, const unsigned int)>
      rigidBodyAngularVelocity_;
  std::function<double(const unsigned int, const unsigned int)> rigidBodyForce_;
  std::function<double(const unsigned int, const unsigned int)>
      rigidBodyMoment_;

public:
  StokesRigidBodyEquation();
  ~StokesRigidBodyEquation();

  virtual void Init();
};