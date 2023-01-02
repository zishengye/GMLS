#ifndef _LinearAlgebra_Impl_Default_MapGraph_Hpp_
#define _LinearAlgebra_Impl_Default_MapGraph_Hpp_

#include "Core/Typedef.hpp"

namespace LinearAlgebra {
namespace Impl {
class MapGraph {
public:
  MapGraph();
  MapGraph(const LocalIndex localSize, Boolean defaultOrdering = true);
};
} // namespace Impl
} // namespace LinearAlgebra

#endif