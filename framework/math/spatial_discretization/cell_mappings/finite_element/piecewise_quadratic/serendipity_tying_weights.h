// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/submesh_tying.h"

namespace opensn
{

/**
 * Quadratic serendipity tying weights (the "PWQD-S" member of the PWQD family).
 *
 * The tied value is the quadratic serendipity function of Floater and Lai (SIAM J. Numer. Anal.
 * 54, 2016) built on the piecewise-linear (PWL) generalized barycentric coordinates phi_i of the
 * cell and on the vertex-star coordinates lambda_{i,j} (the barycentric coordinates of the
 * simplex formed by vertex i and its neighbours), evaluated at the tied node:
 *   F_i = phi_i lambda_{i,i},  F_{ij} = phi_i lambda_{i,j} + phi_j lambda_{j,i},
 *   L_i = F_i - 1/2 sum_{j ~ i} F_{ij},  L_{ij} = 2 F_{ij}.
 * These functions are nodal at the vertices and edge midpoints, reproduce every quadratic
 * (polar-form argument, needing only the partition of unity and linear precision of both
 * coordinate sets) and are piecewise quadratic on the side simplices, so the sub-mesh P2 element
 * tied with them is exactly the serendipity element. On a tetrahedron or a triangle it is the
 * Lagrange P2 element.
 *
 * On a planar face the trace is the two-dimensional element of the face polygon (PWL coordinates
 * of the face fan around the face centroid, vertex-star triangles), computed from the face's own
 * dofs, so it is identical from both sides. The construction needs every vertex of the cell to be
 * simple (three neighbours) and every face planar; where that fails (non-planar faces, pyramid
 * apexes, agglomerated cells) the least-squares weights of degree 2 are used for that face or
 * cell instead, which keeps the traces face-local and quadratic precision intact. Polygons need
 * nothing beyond positive side triangles.
 */
class SerendipityTyingWeights : public TyingWeightProvider
{
public:
  /// planarity_tol: largest out-of-plane coordinate of a face vertex, relative to the face
  /// diameter, for the face to count as planar.
  explicit SerendipityTyingWeights(double planarity_tol = 1.0e-9);

  std::string GetName() const override { return "serendipity"; }
  unsigned int GetMaxDegree() const override { return 2; }

  std::vector<double> FaceWeights(const TyingNodeSet& nodes,
                                  const Vector3& x_tied,
                                  const FaceFrame& frame) const override;
  std::vector<double>
  CellWeights(const TyingNodeSet& nodes, const Vector3& x_tied, double scale) const override;

  /// Number of tied rows that fell back to the least-squares weights so far.
  size_t GetNumFallbacks() const { return num_fallbacks_; }

private:
  double planarity_tol_;
  LeastSquaresTyingWeights fallback_;
  mutable size_t num_fallbacks_ = 0;
};

} // namespace opensn
