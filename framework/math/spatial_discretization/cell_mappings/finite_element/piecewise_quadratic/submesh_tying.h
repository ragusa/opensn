// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/vector3.h"
#include "framework/data_types/dense_matrix.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace opensn
{

class Cell;
class Mesh;

/**
 * Lagrange P_p basis on the reference tetrahedron with nodes given as barycentric multi-indices.
 *
 * Barycentric coordinates are lam = (1 - xi - eta - zeta, xi, eta, zeta). Node a = (a0, a1, a2, a3)
 * with sum(a) = p sits at lam = a / p. The shape function of node a is
 *   N_a(lam) = prod_i prod_{m=0}^{a_i - 1} (p lam_i - m) / (m + 1).
 * The nodes are ordered vertices first (a = p e_i, i = 0..3), then the remaining multi-indices in
 * lexicographic order. The nodes with a3 = 0 lie on the zeta = 0 face.
 *
 * \ingroup doc_CellMappings
 */
class TetLagrangeReference
{
public:
  explicit TetLagrangeReference(unsigned int p);

  unsigned int GetDegree() const { return p_; }
  size_t GetNumNodes() const { return nodes_.size(); }
  const std::vector<std::array<unsigned int, 4>>& GetNodes() const { return nodes_; }
  /// True for the nodes on the zeta = 0 face (a3 == 0).
  const std::vector<bool>& GetZetaZeroFaceMask() const { return on_zeta_zero_face_; }

  /// Reference coordinates (xi, eta, zeta) of node a: (a1, a2, a3) / p.
  Vector3 GetNodeReferenceLocation(size_t a) const;

  /// Shape function values at the reference point xi (size num_nodes).
  void Shape(const Vector3& xi, std::vector<double>& values) const;

  /// Reference gradients (d/dxi, d/deta, d/dzeta) at the reference point xi (size num_nodes).
  void Grad(const Vector3& xi, std::vector<Vector3>& grads) const;

private:
  unsigned int p_;
  std::vector<std::array<unsigned int, 4>> nodes_;
  std::vector<bool> on_zeta_zero_face_;
};

/**
 * Orthonormal frame of a face used for the face-local least-squares fits: origin, two in-plane
 * axes and the normal of the least-squares plane of the face vertices, every axis scaled by the
 * reciprocal of the face diameter so that frame coordinates are O(1).
 */
struct FaceFrame
{
  Vector3 origin;
  std::array<Vector3, 3> axes;

  /// Frame coordinates (s, t, nu) of the point x.
  Vector3 Map(const Vector3& x) const
  {
    const Vector3 d = x - origin;
    return {d.Dot(axes[0]), d.Dot(axes[1]), d.Dot(axes[2])};
  }
};

/**
 * Canonical frame of the face with the given vertex locations. The vertices must be passed in a
 * canonical order (sorted by global vertex id) so that both cells sharing a face obtain
 * bit-identical frames.
 */
FaceFrame MakeFaceFrame(const std::vector<Vector3>& face_vertices_sorted);

/// Key identifying a dof independently of the cell: (vid, vid) for a vertex, (min vid, max vid)
/// for an edge midpoint.
using SubmeshDofKey = std::pair<uint64_t, uint64_t>;

/**
 * The real nodes (dofs) a tied sub-mesh node may depend on, in a canonical order (sorted by dof
 * key, so that both cells sharing a face see the same list), with the topology a weight provider
 * may need: the vertex loops (by global vertex id) of the face, or of every face of the cell, and
 * the vertex average of the face or cell (OpenSn's face and cell centroids). For a polygon the
 * loops are its edges (two vertices each).
 */
struct TyingNodeSet
{
  std::vector<Vector3> x;
  std::vector<SubmeshDofKey> keys;
  std::vector<std::vector<uint64_t>> loops;
  Vector3 centroid;
};

/**
 * Provider of the weights that tie a non-real sub-mesh node to the real nodes (dofs).
 *
 * The inputs are given in a canonical order (dofs sorted by key) so that both cells sharing a face
 * compute the same floating-point face weights.
 */
class TyingWeightProvider
{
public:
  virtual ~TyingWeightProvider() = default;

  virtual std::string GetName() const = 0;

  /// Highest polynomial degree the weights reproduce.
  virtual unsigned int GetMaxDegree() const = 0;

  /// Weights over the real nodes of a face (nodes) for the tied point x_tied on that face.
  virtual std::vector<double>
  FaceWeights(const TyingNodeSet& nodes, const Vector3& x_tied, const FaceFrame& frame) const = 0;

  /// Weights over all real nodes of the cell (nodes) for the tied point x_tied; scale is the
  /// cell diameter.
  virtual std::vector<double>
  CellWeights(const TyingNodeSet& nodes, const Vector3& x_tied, double scale) const = 0;
};

/**
 * PWLD weights: every tied node is the plain average of the real nodes it is tied to. Linear
 * precision holds only when the tied node is the vertex average of those nodes (the face and cell
 * centroids of OpenSn's side tetrahedra), hence only p = 1 accepts this provider.
 */
class AverageTyingWeights : public TyingWeightProvider
{
public:
  std::string GetName() const override { return "average"; }
  unsigned int GetMaxDegree() const override { return 1; }
  std::vector<double> FaceWeights(const TyingNodeSet& nodes,
                                  const Vector3& x_tied,
                                  const FaceFrame& frame) const override;
  std::vector<double>
  CellWeights(const TyingNodeSet& nodes, const Vector3& x_tied, double scale) const override;
};

/**
 * Least-squares polynomial weights of degree q: the tied value is the least-squares polynomial of
 * degree q through the real nodes, evaluated at the tied node. On a face the fit space consists of
 * all monomials of degree <= q in the two in-plane frame coordinates plus the linear out-of-plane
 * term, so every 3D linear function is reproduced on any face and every 3D polynomial of degree
 * <= q on a planar face. In the cell the fit space is the full 3D polynomial space of degree <= q.
 * The fit is exact interpolation whenever the number of nodes equals the number of monomials
 * (triangular faces at q = 2; tetrahedra at q = 2). The pseudo-inverse drops singular values below
 * 1e-12 times the largest one (a planar face makes the out-of-plane column vanish).
 */
class LeastSquaresTyingWeights : public TyingWeightProvider
{
public:
  explicit LeastSquaresTyingWeights(unsigned int q);
  std::string GetName() const override { return "lsq" + std::to_string(q_); }
  unsigned int GetMaxDegree() const override { return q_; }
  std::vector<double> FaceWeights(const TyingNodeSet& nodes,
                                  const Vector3& x_tied,
                                  const FaceFrame& frame) const override;
  std::vector<double>
  CellWeights(const TyingNodeSet& nodes, const Vector3& x_tied, double scale) const override;

private:
  unsigned int q_;
};

/**
 * Weights w (size n) such that w^T values gives the least-squares fit, in the column space of the
 * n x m Vandermonde matrix A, evaluated at the point whose monomial row is m_row (size m):
 * w = pinv(A)^T m_row. The pseudo-inverse is computed by a one-sided Jacobi SVD and drops singular
 * values below rcond times the largest one. Exposed for testing.
 */
std::vector<double> LeastSquaresWeights(const DenseMatrix<double>& A,
                                        const std::vector<double>& m_row,
                                        double rcond = 1.0e-12);

/// Kind of a sub-mesh node of the constrained sub-mesh element.
enum class SubmeshNodeKind : int
{
  REAL = 0,
  FACE_TIED = 1,
  CELL_TIED = 2
};

/**
 * The constrained sub-mesh element of a polyhedral or polygonal cell: P_p Lagrange on OpenSn's
 * side tetrahedra (v_k, x_f, v_{k+1}, x_c) of a polyhedron, or on the side triangles
 * (v_k, v_{k+1}, x_c) of a polygon, with the non-real nodes tied to the real nodes (the dofs) by
 * fixed weights. PWLD is p = 1 with average weights; PWQD is p = 2 with least-squares weights.
 *
 * Sub-mesh nodes are identified across sides by the multiset of sub-vertices that support them
 * (cell vertices 'V', face centroids 'F', the cell centroid 'C'). Node classes: REAL (only 'V'
 * support: a cell vertex, or for p = 2 a cell-edge midpoint), FACE_TIED ('F' in the support and no
 * 'C'), CELL_TIED ('C' in the support). Dofs are the real nodes, ordered cell vertices (in
 * cell.vertex_ids order) then edge midpoints (in order of first appearance walking the face loops).
 * In 2D the faces are the edges, their midpoints are real nodes and there are no face-tied nodes.
 *
 * Side s of the cell is the side tet of face GetSideFace(s), edge GetSideEdge(s), with the
 * reference tet nodes 0..3 at (v_k, x_f, v_{k+1}, x_c), exactly as in
 * PieceWiseLinearPolyhedronMapping. For a polygon side s is the side triangle of face (edge) s
 * with the reference nodes 0..2 at (v_k, v_{k+1}, x_c), exactly as in
 * PieceWiseLinearPolygonMapping; it uses the nodes of the zeta = 0 face of the reference
 * tetrahedron (the P_p triangle), see GetSideReferenceNodes(). The tying operator T
 * (num_sub_nodes x num_dofs) has identity rows for real nodes; face-tied rows involve only the
 * dofs of that face; cell-tied rows involve all dofs.
 */
class SubmeshTying
{
public:
  using DofKey = SubmeshDofKey;

  SubmeshTying(const Mesh& grid,
               const Cell& cell,
               unsigned int p,
               const TyingWeightProvider& weights);

  unsigned int GetDegree() const { return p_; }
  /// 3 for a polyhedron, 2 for a polygon.
  unsigned int GetDimension() const { return dim_; }
  const TetLagrangeReference& GetReferenceElement() const { return ref_; }

  size_t GetNumDofs() const { return dof_locations_.size(); }
  const std::vector<Vector3>& GetDofLocations() const { return dof_locations_; }
  const std::vector<DofKey>& GetDofKeys() const { return dof_keys_; }
  /// Per face: the dofs with support on that face, the face vertices (loop order) followed by the
  /// face edge midpoints (loop order).
  const std::vector<std::vector<int>>& GetFaceDofs() const { return face_dofs_; }

  size_t GetNumSides() const { return side_face_.size(); }
  size_t GetSideFace(size_t s) const { return side_face_[s]; }
  size_t GetSideEdge(size_t s) const { return side_edge_[s]; }
  /// Reference-element nodes that make up a side: every node of the reference tetrahedron for a
  /// polyhedron, the nodes of its zeta = 0 face for a polygon. GetSideNodes(s)[k] and row k of
  /// GetSideTying(s) belong to reference node GetSideReferenceNodes()[k].
  const std::vector<size_t>& GetSideReferenceNodes() const { return side_reference_nodes_; }
  /// Sub-mesh node index of each reference node (GetSideReferenceNodes()) of side s.
  const std::vector<size_t>& GetSideNodes(size_t s) const { return side_nodes_[s]; }
  /// Rows of T restricted to the reference nodes of side s (num_side_nodes x num_dofs).
  const DenseMatrix<double>& GetSideTying(size_t s) const { return side_tying_[s]; }

  size_t GetNumSubNodes() const { return sub_node_locations_.size(); }
  const std::vector<Vector3>& GetSubNodeLocations() const { return sub_node_locations_; }
  const std::vector<SubmeshNodeKind>& GetSubNodeKinds() const { return sub_node_kinds_; }
  const DenseMatrix<double>& GetTyingOperator() const { return T_; }

private:
  unsigned int p_;
  unsigned int dim_ = 3;
  TetLagrangeReference ref_;
  std::vector<size_t> side_reference_nodes_;

  std::vector<Vector3> dof_locations_;
  std::vector<DofKey> dof_keys_;
  std::vector<std::vector<int>> face_dofs_;

  std::vector<size_t> side_face_;
  std::vector<size_t> side_edge_;
  std::vector<std::vector<size_t>> side_nodes_;
  std::vector<DenseMatrix<double>> side_tying_;

  std::vector<Vector3> sub_node_locations_;
  std::vector<SubmeshNodeKind> sub_node_kinds_;
  std::vector<int> sub_node_face_;
  DenseMatrix<double> T_;
};

} // namespace opensn
