#ifndef _HYDROGEN_NUMERIC_VECTOR_HPP_
#define _HYDROGEN_NUMERIC_VECTOR_HPP_

#include "type.hpp"

#include "container/array.hpp"

namespace hydrogen {
namespace numeric {
template <typename T>
class Vector {
 public:
  static_assert(is_floating_point<T>::value == true,
                "Vector requires numerical entries.");
};
}  // namespace numeric
}  // namespace hydrogen

#endif