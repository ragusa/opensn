// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/spatial_discretization/cell_mappings/cell_mapping.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/submesh_tying.h"
#include "framework/math/quadratures/spatial/tetrahedra_quadrature.h"
#include "framework/math/quadratures/spatial/triangle_quadrature.h"
#include "framework/data_types/matrix3x3.h"
#include <memory>

namespace opensn
{

class Cell;

/**
 * Constrained sub-mesh (side-tetrahedron) finite element of degree p on a polyhedral cell. With
 * p = 2 and least-squares tying this is the PWQD basis: nodes at the cell vertices and at the
 * cell-edge midpoints, P2 on every side tetrahedron (v_k, x_f, v_{k+1}, x_c) of the cell, the face
 * and cell sub-mesh nodes tied to the real nodes with quadratic precision (see SubmeshTying). With
 * p = 1 and average tying it reproduces the PWLD basis of PieceWiseLinearPolyhedronMapping (used
 * only as a regression check).
 *
 * The side-tetrahedron geometry is the same as in PieceWiseLinearPolyhedronMapping: reference
 * coordinates (xi, eta, zeta) with x = v0 + J (xi, eta, zeta), J = [x_f - v0 | v1 - v0 | x_c - v0],
 * the face sub-triangle being zeta = 0.
 *
 * Traces are face-local: a basis function whose node is not on face f vanishes on f, so the face
 * node mapping lists exactly the nodes that couple through that face.
 *
 * \ingroup doc_CellMappings
 */
class PieceWiseQuadraticPolyhedronMapping : public CellMapping
{
public:
  PieceWiseQuadraticPolyhedronMapping(const Cell& polyh_cell,
                                      std::shared_ptr<Mesh> ref_grid,
                                      const TetrahedraQuadrature& volume_quadrature,
                                      const TriangleQuadrature& surface_quadrature,
                                      unsigned int degree,
                                      std::shared_ptr<const TyingWeightProvider> weights);

  /// PWQD: degree 2, least-squares tying of degree 2.
  PieceWiseQuadraticPolyhedronMapping(const Cell& polyh_cell,
                                      std::shared_ptr<Mesh> ref_grid,
                                      const TetrahedraQuadrature& volume_quadrature,
                                      const TriangleQuadrature& surface_quadrature);

  VolumetricFiniteElementData MakeVolumetricFiniteElementData() const override;

  SurfaceFiniteElementData MakeSurfaceFiniteElementData(size_t face_index) const override;

  double ShapeValue(size_t i, const Vector3& xyz) const override;

  Vector3 GradShapeValue(size_t i, const Vector3& xyz) const override;

  void ShapeValues(const Vector3& xyz, Vector<double>& shape_values) const override;

  void GradShapeValues(const Vector3& xyz, std::vector<Vector3>& gradshape_values) const override;

  const SubmeshTying& GetTying() const { return *tying_; }

  /// Side-tetrahedron data: v0, J, detJ, J^{-1}, J^{-T}, surface Jacobian of the zeta = 0 face.
  struct SideData
  {
    double detJ = 0.0;
    double detJ_surf = 0.0;
    Vector3 v0;
    Matrix3x3 J;
    Matrix3x3 Jinv;
    Matrix3x3 JTinv;
  };

  const std::vector<SideData>& GetSides() const { return sides_; }
  /// Sides (indices into GetSides()) that belong to face f, in face-loop order.
  const std::vector<size_t>& GetFaceSides(size_t f) const { return face_sides_[f]; }

private:
  /// Delegated-to constructor: the tying is built once by the public constructors.
  PieceWiseQuadraticPolyhedronMapping(const Cell& polyh_cell,
                                      std::shared_ptr<Mesh> ref_grid,
                                      const TetrahedraQuadrature& volume_quadrature,
                                      const TriangleQuadrature& surface_quadrature,
                                      std::shared_ptr<const TyingWeightProvider> weights,
                                      std::unique_ptr<SubmeshTying> tying);

  /// Shape values and physical gradients of all nodes at the reference point xi of side s.
  void EvaluateInSide(size_t s,
                      const Vector3& xi,
                      std::vector<double>& values,
                      std::vector<Vector3>& grads) const;

  /// Finds the side tet containing xyz (returns false if none) and its reference coordinates.
  bool LocateSide(const Vector3& xyz, size_t& side, Vector3& xi) const;

  std::shared_ptr<const TyingWeightProvider> weights_;
  std::unique_ptr<SubmeshTying> tying_;
  std::vector<SideData> sides_;
  std::vector<std::vector<size_t>> face_sides_;
  std::vector<Vector3> face_normals_;

  const TetrahedraQuadrature& volume_quadrature_;
  const TriangleQuadrature& surface_quadrature_;
};

} // namespace opensn
