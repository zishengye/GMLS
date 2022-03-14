#ifndef _HYDROGEN_CONTAINER_HPP_
#define _HYDROGEN_CONTAINER_HPP_

#include "type.hpp"

namespace hydrogen {
template <class Element>
class RandomAccessContainer {
 private:
 public:
  RandomAccessContainer();
  RandomAccessContainer(const size_t size);

  void resize(const size_t size);

  Element &operator()(const size_t index);
};

template <class Element>
class LinkContainer {};

template <class Element, class HashGenerator>
class HashTableContainer {};
}  // namespace hydrogen

#endif