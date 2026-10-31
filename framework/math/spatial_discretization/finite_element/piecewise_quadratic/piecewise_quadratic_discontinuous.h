// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_discontinuous.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/submesh_tying.h"

namespace opensn
{

/// Tying of the non-real sub-mesh nodes of PWQD: least-squares quadratic fits (the default) or
/// the quadratic serendipity functions on PWL coordinates (SerendipityTyingWeights).
enum class PWQDTying
{
  LEAST_SQUARES = 0,
  SERENDIPITY = 1
};

/**
 * Piecewise quadratic discontinuous (PWQD) finite element discretization on polyhedral and
 * polygonal cells: nodes at the cell vertices and cell-edge midpoints, P2 on every side
 * tetrahedron (3D) or side triangle (2D) of the cell with the face and cell sub-mesh nodes tied
 * to the real nodes with quadratic precision (see PieceWiseQuadraticPolyhedronMapping and
 * PieceWiseQuadraticPolygonMapping). The node ordering, sparsity pattern and DOF mapping are
 * those of PieceWiseLinearDiscontinuous (they only depend on the per-cell node count).
 *
 * Slab (1D) cells are rejected in this version. The default quadrature order is FOURTH because
 * the mass matrix is of degree 4 on each side simplex.
 *
 * \ingroup doc_SpatialDiscretization
 */
class PieceWiseQuadraticDiscontinuous : public PieceWiseLinearDiscontinuous
{
private:
  PieceWiseQuadraticDiscontinuous(const std::shared_ptr<Mesh>& grid,
                                  QuadratureOrder q_order,
                                  std::shared_ptr<const TyingWeightProvider> weights);

  static CellMappingFactory MakePWQDFactory(PieceWiseQuadraticDiscontinuous& self,
                                            std::shared_ptr<const TyingWeightProvider> weights);

  std::shared_ptr<const TyingWeightProvider> weights_;

public:
  /// Construct a shared object using the private constructor.
  static std::shared_ptr<PieceWiseQuadraticDiscontinuous>
  New(const std::shared_ptr<Mesh>& grid,
      QuadratureOrder q_order = QuadratureOrder::FOURTH,
      PWQDTying tying = PWQDTying::LEAST_SQUARES);

  /// The weight provider shared by all cell mappings.
  const TyingWeightProvider& GetTyingWeights() const { return *weights_; }
};

} // namespace opensn
