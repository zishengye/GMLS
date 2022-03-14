#ifndef _HYDROGEN_NUMERIC_MATRIX_HPP_
#define _HYDROGEN_NUMERIC_MATRIX_HPP_

#include "type.hpp"

#include "container/array.hpp"
#include "numeric/vector.hpp"

namespace hydrogen {
namespace numeric {
template <typename T>
class Matrix : public Array<T> {
 private:
  Vector<T> leftVector_, rightVector_;

 public:
  static_assert(isNumeric<T>::value == true,
                "Matrix requires numerical entries.");

  Matrix();
  Matrix();

  ~Matrix();

  void resize();
};
}  // namespace numeric
}  // namespace hydrogen

#endif