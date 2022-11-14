#ifndef _LinearAlgebra_LinearSolverDescriptor_Hpp_
#define _LinearAlgebra_LinearSolverDescriptor_Hpp_

#include <functional>

#include "Core/Typedef.hpp"
#include "LinearAlgebra/Vector.hpp"

namespace LinearAlgebra {
/*!
 * \brief Add linear system to the solver.
 * \date Oct, 1, 2022
 * \author Zisheng Ye <zisheng_ye@outlook.com>
 * \param[in] spd            Symmetric positive definite matrix indicator. 1
 *                           denotes spd matrix, 0 denotes symmetric matrix,
 *                           -1 denotes general matrix.
 * \param[in] outerIteration Multiple layer of iteration indicator. 0 denotes
 *                           the inner-most iteration, simple iterative method
 *                           like GMRES will be selected. 1 denotes outer
 *                           iteration, flexible iterative method like
 *                           flex-GMRES will be selected. -1 denotes
 *                           non-iterative solver is required.
 */

template <class LinearAlgebraBackend> struct LinearSolverDescriptor {
  LocalIndex outerIteration;
  LocalIndex spd;
  LocalIndex maxIter;
  Scalar relativeTol;
  Boolean setFromDatabase;
  Boolean customPreconditioner;

  std::function<Void(Vector<LinearAlgebraBackend> &,
                     Vector<LinearAlgebraBackend> &)>
      preconditioningIteration;
};
} // namespace LinearAlgebra

#endif