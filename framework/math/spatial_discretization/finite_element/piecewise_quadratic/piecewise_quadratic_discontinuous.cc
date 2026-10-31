// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/spatial_discretization/finite_element/piecewise_quadratic/piecewise_quadratic_discontinuous.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/piecewise_quadratic_polyhedron_mapping.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/piecewise_quadratic_polygon_mapping.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/serendipity_tying_weights.h"
#include "framework/mesh/mesh/mesh.h"
#include <stdexcept>

namespace opensn
{

PieceWiseQuadraticDiscontinuous::PieceWiseQuadraticDiscontinuous(
  const std::shared_ptr<Mesh>& grid,
  QuadratureOrder q_order,
  std::shared_ptr<const TyingWeightProvider> weights)
  : PieceWiseLinearDiscontinuous(grid,
                                 q_order,
                                 SpatialDiscretizationType::PIECEWISE_QUADRATIC_DISCONTINUOUS,
                                 MakePWQDFactory(*this, weights)),
    weights_(std::move(weights))
{
}

PieceWiseLinearBase::CellMappingFactory
PieceWiseQuadraticDiscontinuous::MakePWQDFactory(PieceWiseQuadraticDiscontinuous& self,
                                                 std::shared_ptr<const TyingWeightProvider> weights)
{
  // The factory is invoked from the base-class constructor; it only touches members that the
  // PieceWiseLinearBase constructor has already initialized (grid and quadratures) and its own
  // copy of the weight provider.
  return [&self, weights](const Cell& cell) -> std::unique_ptr<CellMapping>
  {
    switch (cell.GetType())
    {
      case CellType::POLYGON:
        return std::make_unique<PieceWiseQuadraticPolygonMapping>(cell,
                                                                  self.grid_,
                                                                  self.tri_quad_order_arbitrary_,
                                                                  self.line_quad_order_arbitrary_,
                                                                  2,
                                                                  weights);
      case CellType::POLYHEDRON:
        return std::make_unique<PieceWiseQuadraticPolyhedronMapping>(cell,
                                                                     self.grid_,
                                                                     self.tet_quad_order_arbitrary_,
                                                                     self.tri_quad_order_arbitrary_,
                                                                     2,
                                                                     weights);
      default:
        throw std::invalid_argument(
          "PieceWiseQuadraticDiscontinuous (PWQD): only polygonal (2D) and polyhedral (3D) cells "
          "are supported in this version; cell " +
          std::to_string(cell.global_id) + " is neither");
    }
  };
}

std::shared_ptr<PieceWiseQuadraticDiscontinuous>
PieceWiseQuadraticDiscontinuous::New(const std::shared_ptr<Mesh>& grid,
                                     QuadratureOrder q_order,
                                     PWQDTying tying)
{
  if (q_order < QuadratureOrder::FOURTH)
    throw std::invalid_argument("PieceWiseQuadraticDiscontinuous: the quadrature order must be at "
                                "least FOURTH (the mass matrix is of degree 4 on each side tet)");
  std::shared_ptr<const TyingWeightProvider> weights;
  if (tying == PWQDTying::SERENDIPITY)
    weights = std::make_shared<SerendipityTyingWeights>();
  else
    weights = std::make_shared<LeastSquaresTyingWeights>(2);
  return std::shared_ptr<PieceWiseQuadraticDiscontinuous>(
    new PieceWiseQuadraticDiscontinuous(grid, q_order, std::move(weights)));
}

} // namespace opensn
