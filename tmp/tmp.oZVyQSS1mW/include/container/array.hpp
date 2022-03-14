#ifndef _HYDROGEN_CONTAINER_ARRAY_HPP_
#define _HYDROGEN_CONTAINER_ARRAY_HPP_

#include <memory>
#include <vector>

#include "object.hpp"

namespace hydrogen {
namespace container {
template <class Element_, class ExecutionSpace_, int Rank_>
class ArrayBase : public Object<ExecutionSpace_> {};

template <class Element_,
          class ExecutionSpace_,
          class allocator = std::allocator<Element_>,
          int Rank_,
          class... Size_>
class Array {
 private:
  std::size_t arraySize_;

 public:
  Array();
  Array(std::size_t size);

  bool resize(std::size_t newSize);
  bool resize(std::vector<std::size_t> newSize);
};
}  // namespace container
}  // namespace hydrogen

#endif