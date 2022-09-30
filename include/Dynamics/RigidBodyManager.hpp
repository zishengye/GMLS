#ifndef _Dynamics_RigidBodyManager_Hpp_
#define _Dynamics_RigidBodyManager_Hpp_

#include <string>

#include "Core/Typedef.hpp"
#include "Dynamics/Quaternion.hpp"
#include "Math/Mat3.hpp"

namespace Dynamics {
class RigidBodyManager {
private:
  std::vector<Math::Vec3> position_;
  std::vector<Dynamics::Quaternion> orientation_;

  std::vector<Math::Vec3> velocity_;
  std::vector<Math::Vec3> angularVelocity_;

  std::vector<Math::Vec3> acceleration_;
  std::vector<Math::Vec3> angularAcceleration_;

  std::vector<Math::Vec3> force_;
  std::vector<Math::Vec3> torque_;

  std::vector<Scalar> mass_;
  std::vector<Math::Mat3> momentOfInertia_;

  std::vector<LocalIndex> type_;

  std::vector<std::vector<Scalar>> sizeList_;

public:
  RigidBodyManager();
  ~RigidBodyManager();

  void Init(std::string RigidBodyInputFileName);

  Size GetNumRigidBody();

  LocalIndex GetType(const LocalIndex index);

  Scalar GetSize(const LocalIndex typeIndex, const LocalIndex sizeIndex);

  const Math::Vec3 &GetPosition(const LocalIndex index);
  const Dynamics::Quaternion &GetOrientation(const LocalIndex index);

  const Math::Vec3 &GetVelocity(const LocalIndex index);
  const Math::Vec3 &GetAngularVelocity(const LocalIndex index);

  const Math::Vec3 &GetAcceleration(const LocalIndex index);
  const Math::Vec3 &GetAngularAcceleration(const LocalIndex index);

  const Math::Vec3 &GetForce(const LocalIndex index);
  const Math::Vec3 &GetTorque(const LocalIndex index);

  void SetPosition(const LocalIndex index, const Math::Vec3 &position);
  void SetOrientation(const LocalIndex index,
                      const Dynamics::Quaternion &orientation);

  void SetVelocity(const LocalIndex index, const Math::Vec3 &velocity);
  void SetAngularVelocity(const LocalIndex index,
                          const Math::Vec3 &angularVelocity);
};
} // namespace Dynamics

#endif