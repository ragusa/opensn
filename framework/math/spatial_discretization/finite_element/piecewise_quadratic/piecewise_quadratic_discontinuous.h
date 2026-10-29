// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_discontinuous.h"

namespace opensn
{

/**
 * Piecewise quadratic discontinuous (PWQD) finite element discretization on polyhedral cells:
 * nodes at the cell vertices and cell-edge midpoints, P2 on every side tetrahedron of the cell
 * with the face and cell sub-mesh nodes tied to the real nodes with quadratic precision (see
 * PieceWiseQuadraticPolyhedronMapping). The node ordering, sparsity pattern and DOF mapping are
 * those of PieceWiseLinearDiscontinuous (they only depend on the per-cell node count).
 *
 * Only 3D polyhedral cells are supported in this version; slab and polygon cells are rejected.
 * The default quadrature order is FOURTH because the mass matrix is of degree 4 on each side
 * tetrahedron.
 *
 * \ingroup doc_SpatialDiscretization
 */
class PieceWiseQuadraticDiscontinuous : public PieceWiseLinearDiscontinuous
{
private:
  explicit PieceWiseQuadraticDiscontinuous(const std::shared_ptr<Mesh>& grid,
                                           QuadratureOrder q_order);

  static CellMappingFactory MakePWQDFactory(PieceWiseQuadraticDiscontinuous& self);

public:
  /// Construct a shared object using the private constructor.
  static std::shared_ptr<PieceWiseQuadraticDiscontinuous>
  New(const std::shared_ptr<Mesh>& grid, QuadratureOrder q_order = QuadratureOrder::FOURTH);
};

} // namespace opensn
