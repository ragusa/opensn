// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/spatial_discretization/finite_element/piecewise_quadratic/piecewise_quadratic_discontinuous.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/piecewise_quadratic_polyhedron_mapping.h"
#include "framework/mesh/mesh/mesh.h"
#include <stdexcept>

namespace opensn
{

PieceWiseQuadraticDiscontinuous::PieceWiseQuadraticDiscontinuous(const std::shared_ptr<Mesh>& grid,
                                                                 QuadratureOrder q_order)
  : PieceWiseLinearDiscontinuous(grid,
                                 q_order,
                                 SpatialDiscretizationType::PIECEWISE_QUADRATIC_DISCONTINUOUS,
                                 MakePWQDFactory(*this))
{
}

PieceWiseLinearBase::CellMappingFactory
PieceWiseQuadraticDiscontinuous::MakePWQDFactory(PieceWiseQuadraticDiscontinuous& self)
{
  // The factory is invoked from the base-class constructor; it only touches members that the
  // PieceWiseLinearBase constructor has already initialized (grid and quadratures).
  return [&self](const Cell& cell) -> std::unique_ptr<CellMapping>
  {
    if (cell.GetType() != CellType::POLYHEDRON)
      throw std::invalid_argument(
        "PieceWiseQuadraticDiscontinuous (PWQD): only polyhedral (3D) cells are supported in this "
        "version; cell " +
        std::to_string(cell.global_id) + " is not a polyhedron");
    return std::make_unique<PieceWiseQuadraticPolyhedronMapping>(
      cell, self.grid_, self.tet_quad_order_arbitrary_, self.tri_quad_order_arbitrary_);
  };
}

std::shared_ptr<PieceWiseQuadraticDiscontinuous>
PieceWiseQuadraticDiscontinuous::New(const std::shared_ptr<Mesh>& grid, QuadratureOrder q_order)
{
  if (q_order < QuadratureOrder::FOURTH)
    throw std::invalid_argument("PieceWiseQuadraticDiscontinuous: the quadrature order must be at "
                                "least FOURTH (the mass matrix is of degree 4 on each side tet)");
  return std::shared_ptr<PieceWiseQuadraticDiscontinuous>(
    new PieceWiseQuadraticDiscontinuous(grid, q_order));
}

} // namespace opensn
