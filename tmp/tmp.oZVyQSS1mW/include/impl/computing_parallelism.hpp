#ifndef _HYDROGEN_COMPUTING_PARALLELISM_HPP_
#define _HYDROGEN_COMPUTING_PARALLELISM_HPP_

// CPU side computing parallelism flags
#ifdef ENABLE_OPENMP
#define HYDROGEN_ENABLE_OPENMP
#endif

// GPU side computing parallelism flags
#ifdef ENABLE_CUDA
#define HYDROGEN_ENABLE_CUDA
#endif

#ifdef ENABLE_OPENCL
#define HYDROGEN_ENABLE_OPENCL
#endif

#endif