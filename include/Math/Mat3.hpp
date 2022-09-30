#ifndef _Math_Mat3_Hpp_
#define _Math_Mat3_Hpp_

#include <cmath>
#include <fstream>
#include <vector>

#include "Core/Typedef.hpp"

namespace Math {
class Mat3 {
private:
  Scalar data_[9];

public:
  Mat3() {
    for (LocalIndex i = 0; i < 9; i++) {
      data_[i] = 0;
    }
  }
};
} // namespace Math

#endif