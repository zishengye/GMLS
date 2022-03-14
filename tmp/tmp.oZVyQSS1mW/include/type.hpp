#ifndef _HYDROGEN_TYPE_HPP_
#define _HYDROGEN_TYPE_HPP_

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// integer definition
#ifdef USE_INT64_DEFAULT
typedef uint64_t UnsignedInteger;
typedef int64_t Integer;
#else
typedef uint32_t UnsignedInteger;
typedef int32_t Integer;
typedef uint64_t LongUnsignedInteger;
typedef int64_t LongInteger;
#endif

// floating point definition
typedef float SingleReal;
typedef double Real;
typedef long double LongReal;

// extra built-in type definition
typedef bool Boolean;
typedef wchar_t Letter;

typedef void *RawPointer;

typedef std::complex<float> SingleComplex;
typedef std::complex<double> Complex;
typedef std::complex<long double> LongComplex;

template <class T>
struct isNumeric
    : std::integral_constant<
          bool,
          std::is_same<float, typename std::remove_cv<T>::type>::value ||
              std::is_same<double, typename std::remove_cv<T>::type>::value ||
              std::is_same<long double,
                           typename std::remove_cv<T>::type>::value ||
              std::is_same<std::complex<float>,
                           typename std::remove_cv<T>::type>::value ||
              std::is_same<std::complex<double>,
                           typename std::remove_cv<T>::type>::value ||
              std::is_same<std::complex<long double>,
                           typename std::remove_cv<T>::type>::value> {};

#endif