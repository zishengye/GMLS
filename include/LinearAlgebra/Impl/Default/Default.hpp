#ifndef _LinearAlgebra_Impl_Default_Default_Hpp_
#define _LinearAlgebra_Impl_Default_Default_Hpp_

/*! \brief A common header for Default based objects.
 *! \date Dec 17, 2022
 *! \author Zisheng Ye <zisheng_ye@outlook.com>
 * This is the default implementation of linear algebra package of the solver.
 * It only depends on Kokkos and MPI.
 */

namespace LinearAlgebra {
namespace Impl {
class DefaultBackend;

class DefaultVector;
class DefaultVectorEntry;
class DefaultMatrix;
class DefaultBlockMatrix;
class DefaultLinearSolver;
} // namespace Impl
} // namespace LinearAlgebra

#endif