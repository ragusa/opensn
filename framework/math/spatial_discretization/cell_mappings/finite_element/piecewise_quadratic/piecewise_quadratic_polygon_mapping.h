// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/spatial_discretization/cell_mappings/cell_mapping.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/submesh_tying.h"
#include "framework/math/quadratures/spatial/triangle_quadrature.h"
#include "framework/math/quadratures/spatial/line_quadrature.h"
#include "framework/data_types/matrix3x3.h"
#include <memory>

namespace opensn
{

class Cell;

/**
 * Constrained sub-mesh (side-triangle) finite element of degree p on a polygonal cell, the 2D
 * counterpart of PieceWiseQuadraticPolyhedronMapping. With p = 2 and least-squares tying this is
 * the PWQD basis: nodes at the cell vertices and at the edge midpoints (2 n_v per cell), P2 on
 * every side triangle (v_k, v_{k+1}, x_c) of the cell, the sub-mesh nodes on the spokes to the
 * centroid and the centroid itself tied to the real nodes by a least-squares quadratic through
 * all of them (exact interpolation on triangles, where the basis is the Lagrange P2 element).
 * With p = 1 and average tying it reproduces the PWLD basis of PieceWiseLinearPolygonMapping
 * (used only as a regression check).
 *
 * The side-triangle geometry is the same as in PieceWiseLinearPolygonMapping: reference
 * coordinates (xi, eta) with x = v0 + J (xi, eta), J = [v1 - v0 | x_c - v0], the edge being
 * eta = 0. The reference element is the zeta = 0 face of the P_p reference tetrahedron.
 *
 * Traces are edge-local: the trace on an edge is the P_p Lagrange trace of the edge, which
 * involves only the two edge vertices and (p = 2) the edge midpoint, so the face node mapping
 * lists exactly the nodes that couple through that edge.
 *
 * \ingroup doc_CellMappings
 */
class PieceWiseQuadraticPolygonMapping : public CellMapping
{
public:
  PieceWiseQuadraticPolygonMapping(const Cell& poly_cell,
                                   std::shared_ptr<Mesh> ref_grid,
                                   const TriangleQuadrature& volume_quadrature,
                                   const LineQuadrature& surface_quadrature,
                                   unsigned int degree,
                                   std::shared_ptr<const TyingWeightProvider> weights);

  /// PWQD: degree 2, least-squares tying of degree 2.
  PieceWiseQuadraticPolygonMapping(const Cell& poly_cell,
                                   std::shared_ptr<Mesh> ref_grid,
                                   const TriangleQuadrature& volume_quadrature,
                                   const LineQuadrature& surface_quadrature);

  VolumetricFiniteElementData MakeVolumetricFiniteElementData() const override;

  SurfaceFiniteElementData MakeSurfaceFiniteElementData(size_t face_index) const override;

  double ShapeValue(size_t i, const Vector3& xyz) const override;

  Vector3 GradShapeValue(size_t i, const Vector3& xyz) const override;

  void ShapeValues(const Vector3& xyz, Vector<double>& shape_values) const override;

  void GradShapeValues(const Vector3& xyz, std::vector<Vector3>& gradshape_values) const override;

  const SubmeshTying& GetTying() const { return *tying_; }

  /// Side-triangle data: v0, J, detJ (twice the area), J^{-1}, J^{-T} (2x2 blocks of 3x3
  /// matrices with zero third row and column), the edge length and the outward edge normal.
  struct SideData
  {
    double detJ = 0.0;
    double detJ_surf = 0.0;
    Vector3 v0;
    Matrix3x3 J;
    Matrix3x3 Jinv;
    Matrix3x3 JTinv;
    Vector3 normal;
  };

  /// One side per face (edge), in face order.
  const std::vector<SideData>& GetSides() const { return sides_; }

private:
  /// Delegated-to constructor: the tying is built once by the public constructors.
  PieceWiseQuadraticPolygonMapping(const Cell& poly_cell,
                                   std::shared_ptr<Mesh> ref_grid,
                                   const TriangleQuadrature& volume_quadrature,
                                   const LineQuadrature& surface_quadrature,
                                   std::shared_ptr<const TyingWeightProvider> weights,
                                   std::unique_ptr<SubmeshTying> tying);

  /// Shape values and physical gradients of all nodes at the reference point (xi, eta, 0) of
  /// side s.
  void EvaluateInSide(size_t s,
                      const Vector3& xi,
                      std::vector<double>& values,
                      std::vector<Vector3>& grads) const;

  /// Finds the side triangle containing xyz (returns false if none) and its reference
  /// coordinates.
  bool LocateSide(const Vector3& xyz, size_t& side, Vector3& xi) const;

  std::shared_ptr<const TyingWeightProvider> weights_;
  std::unique_ptr<SubmeshTying> tying_;
  std::vector<SideData> sides_;

  const TriangleQuadrature& volume_quadrature_;
  const LineQuadrature& surface_quadrature_;
};

} // namespace opensn
