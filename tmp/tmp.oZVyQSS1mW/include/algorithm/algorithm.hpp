#ifndef _HYDROGEN_ALGORITHM_ALGORITHM_HPP_
#define _HYDROGEN_ALGORITHM_ALGORITHM_HPP_

namespace hydrogen {
template <class RandomAccessIterator>
void sort(RandomAccessIterator first, RandomAccessIterator last);

template <class RandomAccessIterator, typename Compare>
void sort(RandomAccessIterator first,
          RandomAccessIterator last,
          Compare compare);

template <class RandomAccessIterator,
          typename ExecutionSpace,
          typename ExecutionPolicy>
void sort(RandomAccessIterator first, RandomAccessIterator last);

template <class RandomAccessIterator,
          typename ExecutionSpace,
          typename ExecutionPolicy,
          typename Compare>
void sort(RandomAccessIterator first,
          RandomAccessIterator last,
          Compare compare);

template <class RandomAccessIterator, typename ExecutionSpace>
void stable_sort(RandomAccessIterator first, RandomAccessIterator last);
}  // namespace hydrogen

#include "algorithm/sort.hpp"

#endif