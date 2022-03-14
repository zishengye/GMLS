#ifndef _HYDROGEN_ALGORITHM_SORT_HPP_
#define _HYDROGEN_ALGORITHM_SORT_HPP_

#include <algorithm>

namespace hydrogen {
template <class RandomAccessIterator>
void sort(RandomAccessIterator first, RandomAccessIterator last) {
  std::sort<RandomAccessIterator>(first, last);
}

template <class RandomAccessIterator, typename Compare>
void sort(RandomAccessIterator first,
          RandomAccessIterator last,
          Compare compare) {
}
}  // namespace hydrogen

#endif