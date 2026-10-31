// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/submesh_tying.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/mesh/cell.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace opensn
{

// ----------------------------------------------------------------------------------------------
// TetLagrangeReference
// ----------------------------------------------------------------------------------------------

TetLagrangeReference::TetLagrangeReference(unsigned int p) : p_(p)
{
  if (p < 1)
    throw std::invalid_argument("TetLagrangeReference: the degree must be at least 1");

  // Vertices first: a = p e_i
  for (unsigned int i = 0; i < 4; ++i)
  {
    std::array<unsigned int, 4> a = {0, 0, 0, 0};
    a[i] = p;
    nodes_.push_back(a);
  }
  // Then all remaining multi-indices with sum p, in lexicographic order
  for (unsigned int a0 = 0; a0 <= p; ++a0)
    for (unsigned int a1 = 0; a1 <= p - a0; ++a1)
      for (unsigned int a2 = 0; a2 <= p - a0 - a1; ++a2)
      {
        const unsigned int a3 = p - a0 - a1 - a2;
        const std::array<unsigned int, 4> a = {a0, a1, a2, a3};
        if (a0 == p or a1 == p or a2 == p or a3 == p)
          continue;
        nodes_.push_back(a);
      }

  on_zeta_zero_face_.reserve(nodes_.size());
  for (const auto& a : nodes_)
    on_zeta_zero_face_.push_back(a[3] == 0);
}

Vector3
TetLagrangeReference::GetNodeReferenceLocation(size_t a) const
{
  const auto& n = nodes_.at(a);
  const double p = static_cast<double>(p_);
  return {n[1] / p, n[2] / p, n[3] / p};
}

namespace
{

/// f(lam) = prod_{m < a_i} (p lam - m)/(m + 1) and its derivative with respect to lam.
inline void
FactorAndDerivative(unsigned int p, unsigned int a_i, double lam, double& f, double& df)
{
  f = 1.0;
  df = 0.0;
  for (unsigned int m = 0; m < a_i; ++m)
  {
    const double g = (p * lam - static_cast<double>(m)) / static_cast<double>(m + 1);
    const double dg = static_cast<double>(p) / static_cast<double>(m + 1);
    df = df * g + f * dg;
    f = f * g;
  }
}

} // namespace

void
TetLagrangeReference::Shape(const Vector3& xi, std::vector<double>& values) const
{
  const std::array<double, 4> lam = {1.0 - xi.x - xi.y - xi.z, xi.x, xi.y, xi.z};
  values.assign(nodes_.size(), 1.0);
  for (size_t j = 0; j < nodes_.size(); ++j)
    for (unsigned int i = 0; i < 4; ++i)
      if (nodes_[j][i] > 0)
      {
        double f, df;
        FactorAndDerivative(p_, nodes_[j][i], lam[i], f, df);
        values[j] *= f;
      }
}

void
TetLagrangeReference::Grad(const Vector3& xi, std::vector<Vector3>& grads) const
{
  const std::array<double, 4> lam = {1.0 - xi.x - xi.y - xi.z, xi.x, xi.y, xi.z};
  // d lam_i / d (xi, eta, zeta)
  const std::array<Vector3, 4> dlam = {Vector3{-1.0, -1.0, -1.0},
                                       Vector3{1.0, 0.0, 0.0},
                                       Vector3{0.0, 1.0, 0.0},
                                       Vector3{0.0, 0.0, 1.0}};

  grads.assign(nodes_.size(), Vector3{0.0, 0.0, 0.0});
  for (size_t j = 0; j < nodes_.size(); ++j)
  {
    std::array<double, 4> f = {1.0, 1.0, 1.0, 1.0};
    std::array<double, 4> df = {0.0, 0.0, 0.0, 0.0};
    for (unsigned int i = 0; i < 4; ++i)
      if (nodes_[j][i] > 0)
        FactorAndDerivative(p_, nodes_[j][i], lam[i], f[i], df[i]);

    for (unsigned int i = 0; i < 4; ++i)
    {
      if (nodes_[j][i] == 0)
        continue;
      double others = 1.0;
      for (unsigned int k = 0; k < 4; ++k)
        if (k != i)
          others *= f[k];
      grads[j] += dlam[i] * (df[i] * others);
    }
  }
}

// ----------------------------------------------------------------------------------------------
// Small dense linear algebra: one-sided Jacobi SVD
// ----------------------------------------------------------------------------------------------

namespace
{

/**
 * One-sided (Hestenes) Jacobi SVD of the n x m matrix A: on return the columns of `cols` are
 * U * Sigma (i.e. mutually orthogonal, with norms the singular values) and V holds the right
 * singular vectors (m x m, column j belongs to column j of `cols`).
 */
void
JacobiSVD(const DenseMatrix<double>& A,
          std::vector<std::vector<double>>& cols,
          std::vector<std::vector<double>>& V)
{
  const size_t n = A.Rows();
  const size_t m = A.Columns();
  cols.assign(m, std::vector<double>(n, 0.0));
  V.assign(m, std::vector<double>(m, 0.0));
  for (size_t j = 0; j < m; ++j)
  {
    for (size_t i = 0; i < n; ++i)
      cols[j][i] = A(i, j);
    V[j][j] = 1.0;
  }

  const double eps = 1.0e-15;
  for (unsigned int sweep = 0; sweep < 100; ++sweep)
  {
    bool rotated = false;
    for (size_t j = 0; j + 1 < m; ++j)
      for (size_t k = j + 1; k < m; ++k)
      {
        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
          alpha += cols[j][i] * cols[j][i];
          beta += cols[k][i] * cols[k][i];
          gamma += cols[j][i] * cols[k][i];
        }
        if (std::fabs(gamma) <= eps * std::sqrt(alpha * beta) or gamma == 0.0)
          continue;
        rotated = true;
        const double zeta = (beta - alpha) / (2.0 * gamma);
        const double t =
          (zeta >= 0.0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = c * t;
        for (size_t i = 0; i < n; ++i)
        {
          const double aj = cols[j][i];
          const double ak = cols[k][i];
          cols[j][i] = c * aj - s * ak;
          cols[k][i] = s * aj + c * ak;
        }
        for (size_t i = 0; i < m; ++i)
        {
          const double vj = V[j][i];
          const double vk = V[k][i];
          V[j][i] = c * vj - s * vk;
          V[k][i] = s * vj + c * vk;
        }
      }
    if (not rotated)
      break;
  }
}

/// Monomials of total degree <= q in the first `dim` components of y, ordered by total degree
/// and then lexicographically by exponent tuple.
std::vector<double>
Monomials(const Vector3& y, unsigned int dim, unsigned int q)
{
  std::vector<double> out;
  for (unsigned int deg = 0; deg <= q; ++deg)
  {
    if (dim == 2)
    {
      for (unsigned int e0 = 0; e0 <= deg; ++e0)
      {
        const unsigned int e1 = deg - e0;
        out.push_back(std::pow(y.x, e0) * std::pow(y.y, e1));
      }
    }
    else
    {
      for (unsigned int e0 = 0; e0 <= deg; ++e0)
        for (unsigned int e1 = 0; e1 <= deg - e0; ++e1)
        {
          const unsigned int e2 = deg - e0 - e1;
          out.push_back(std::pow(y.x, e0) * std::pow(y.y, e1) * std::pow(y.z, e2));
        }
    }
  }
  return out;
}

/// Face fit space: in-plane monomials of degree <= q plus the linear out-of-plane term.
std::vector<double>
FaceMonomials(const Vector3& y, unsigned int q)
{
  auto out = Monomials(y, 2, q);
  out.push_back(y.z);
  return out;
}

double
MaxPairwiseDistance(const std::vector<Vector3>& points)
{
  double d2 = 0.0;
  for (size_t i = 0; i < points.size(); ++i)
    for (size_t j = i + 1; j < points.size(); ++j)
      d2 = std::max(d2, (points[i] - points[j]).NormSquare());
  return std::sqrt(d2);
}

} // namespace

std::vector<double>
LeastSquaresWeights(const DenseMatrix<double>& A, const std::vector<double>& m_row, double rcond)
{
  const size_t n = A.Rows();
  const size_t m = A.Columns();
  if (m_row.size() != m)
    throw std::invalid_argument("LeastSquaresWeights: monomial row size does not match");

  std::vector<std::vector<double>> cols, V;
  JacobiSVD(A, cols, V);

  std::vector<double> sigma(m, 0.0);
  double sigma_max = 0.0;
  for (size_t j = 0; j < m; ++j)
  {
    double s2 = 0.0;
    for (size_t i = 0; i < n; ++i)
      s2 += cols[j][i] * cols[j][i];
    sigma[j] = std::sqrt(s2);
    sigma_max = std::max(sigma_max, sigma[j]);
  }

  // w = U Sigma^+ V^T m_row = sum_j cols_j (V^T m_row)_j / sigma_j^2
  std::vector<double> w(n, 0.0);
  for (size_t j = 0; j < m; ++j)
  {
    if (sigma[j] <= rcond * sigma_max or sigma[j] == 0.0)
      continue;
    double vtm = 0.0;
    for (size_t i = 0; i < m; ++i)
      vtm += V[j][i] * m_row[i];
    const double coef = vtm / (sigma[j] * sigma[j]);
    for (size_t i = 0; i < n; ++i)
      w[i] += cols[j][i] * coef;
  }
  return w;
}

// ----------------------------------------------------------------------------------------------
// Face frame
// ----------------------------------------------------------------------------------------------

FaceFrame
MakeFaceFrame(const std::vector<Vector3>& face_vertices_sorted)
{
  const size_t n = face_vertices_sorted.size();
  if (n < 3)
    throw std::invalid_argument("MakeFaceFrame: a face needs at least three vertices");

  FaceFrame frame;
  frame.origin = Vector3{0.0, 0.0, 0.0};
  for (const auto& x : face_vertices_sorted)
    frame.origin += x;
  frame.origin /= static_cast<double>(n);

  DenseMatrix<double> P(n, 3, 0.0);
  for (size_t i = 0; i < n; ++i)
  {
    const Vector3 d = face_vertices_sorted[i] - frame.origin;
    P(i, 0) = d.x;
    P(i, 1) = d.y;
    P(i, 2) = d.z;
  }

  std::vector<std::vector<double>> cols, V;
  JacobiSVD(P, cols, V);

  // Order the right singular vectors by decreasing singular value: two in-plane axes, then the
  // normal of the least-squares plane.
  std::array<double, 3> sigma = {0.0, 0.0, 0.0};
  for (size_t j = 0; j < 3; ++j)
    for (size_t i = 0; i < n; ++i)
      sigma[j] += cols[j][i] * cols[j][i];
  std::array<size_t, 3> order = {0, 1, 2};
  std::sort(
    order.begin(), order.end(), [&sigma](size_t a, size_t b) { return sigma[a] > sigma[b]; });

  const double diam = MaxPairwiseDistance(face_vertices_sorted);
  for (size_t k = 0; k < 3; ++k)
  {
    const auto& v = V[order[k]];
    Vector3 axis{v[0], v[1], v[2]};
    axis.Normalize();
    frame.axes[k] = axis / diam;
  }
  // Make the normal exactly orthogonal to the in-plane axes (it already is to round-off; this
  // keeps the frame right-handed and well defined when the third singular value vanishes).
  Vector3 e0 = frame.axes[0] * diam;
  Vector3 e1 = frame.axes[1] * diam;
  Vector3 e2 = e0.Cross(e1);
  e2.Normalize();
  frame.axes[2] = e2 / diam;
  return frame;
}

// ----------------------------------------------------------------------------------------------
// Weight providers
// ----------------------------------------------------------------------------------------------

std::vector<double>
AverageTyingWeights::FaceWeights(const TyingNodeSet& nodes,
                                 const Vector3& /*x_tied*/,
                                 const FaceFrame& /*frame*/) const
{
  return std::vector<double>(nodes.x.size(), 1.0 / static_cast<double>(nodes.x.size()));
}

std::vector<double>
AverageTyingWeights::CellWeights(const TyingNodeSet& nodes,
                                 const Vector3& /*x_tied*/,
                                 double /*scale*/) const
{
  return std::vector<double>(nodes.x.size(), 1.0 / static_cast<double>(nodes.x.size()));
}

LeastSquaresTyingWeights::LeastSquaresTyingWeights(unsigned int q) : q_(q)
{
  if (q < 1)
    throw std::invalid_argument("LeastSquaresTyingWeights: the degree must be at least 1");
}

std::vector<double>
LeastSquaresTyingWeights::FaceWeights(const TyingNodeSet& nodes,
                                      const Vector3& x_tied,
                                      const FaceFrame& frame) const
{
  const auto& nodes_x = nodes.x;
  const size_t n = nodes_x.size();
  const auto m_row = FaceMonomials(frame.Map(x_tied), q_);
  DenseMatrix<double> A(n, m_row.size(), 0.0);
  for (size_t i = 0; i < n; ++i)
  {
    const auto row = FaceMonomials(frame.Map(nodes_x[i]), q_);
    for (size_t j = 0; j < row.size(); ++j)
      A(i, j) = row[j];
  }
  return LeastSquaresWeights(A, m_row);
}

std::vector<double>
LeastSquaresTyingWeights::CellWeights(const TyingNodeSet& nodes,
                                      const Vector3& x_tied,
                                      double scale) const
{
  const auto& nodes_x = nodes.x;
  const size_t n = nodes_x.size();
  Vector3 origin{0.0, 0.0, 0.0};
  for (const auto& x : nodes_x)
    origin += x;
  origin /= static_cast<double>(n);

  const auto m_row = Monomials((x_tied - origin) / scale, 3, q_);
  DenseMatrix<double> A(n, m_row.size(), 0.0);
  for (size_t i = 0; i < n; ++i)
  {
    const auto row = Monomials((nodes_x[i] - origin) / scale, 3, q_);
    for (size_t j = 0; j < row.size(); ++j)
      A(i, j) = row[j];
  }
  return LeastSquaresWeights(A, m_row);
}

// ----------------------------------------------------------------------------------------------
// SubmeshTying
// ----------------------------------------------------------------------------------------------

namespace
{

/// A sub-vertex of the side-tet sub-mesh: kind 0 = cell vertex (id = local vertex index),
/// 1 = face centroid (id = face index), 2 = cell centroid; mult = barycentric multiplicity.
struct SupportEntry
{
  int kind;
  int id;
  int mult;
  bool operator<(const SupportEntry& o) const
  {
    if (kind != o.kind)
      return kind < o.kind;
    if (id != o.id)
      return id < o.id;
    return mult < o.mult;
  }
  bool operator==(const SupportEntry& o) const
  {
    return kind == o.kind and id == o.id and mult == o.mult;
  }
};

} // namespace

SubmeshTying::SubmeshTying(const Mesh& grid,
                           const Cell& cell,
                           unsigned int p,
                           const TyingWeightProvider& weights)
  : p_(p), ref_(p)
{
  if (p < 1)
    throw std::invalid_argument("SubmeshTying: the degree must be at least 1");
  if (p > 2)
    throw std::invalid_argument("SubmeshTying: degrees above 2 are not supported (face-interior "
                                "real nodes are not implemented)");
  if (dynamic_cast<const AverageTyingWeights*>(&weights) != nullptr and p != 1)
    throw std::invalid_argument("SubmeshTying: average weights are only linear-precise for p = 1");
  if (weights.GetMaxDegree() < p)
    throw std::invalid_argument("SubmeshTying: weight provider " + weights.GetName() +
                                " reproduces degree " + std::to_string(weights.GetMaxDegree()) +
                                " < p = " + std::to_string(p));

  const CellType cell_type = cell.GetType();
  if (cell_type != CellType::POLYHEDRON and cell_type != CellType::POLYGON)
    throw std::invalid_argument("SubmeshTying: only polygon (2D) and polyhedron (3D) cells are "
                                "supported; cell " +
                                std::to_string(cell.global_id) + " is neither");
  dim_ = (cell_type == CellType::POLYHEDRON) ? 3 : 2;

  const size_t n_v = cell.vertex_ids.size();
  const size_t n_f = cell.faces.size();
  // A polygon face is an edge (two vertices, one cell edge, one side triangle)
  auto num_face_edges = [this](const CellFace& face)
  { return (dim_ == 3) ? face.vertex_ids.size() : size_t{1}; };

  std::map<uint64_t, int> vloc;
  for (size_t i = 0; i < n_v; ++i)
    vloc[cell.vertex_ids[i]] = static_cast<int>(i);

  // Cell edges as (min vid, max vid), numbered in order of first appearance along the face loops.
  std::map<DofKey, int> edge_index;
  std::vector<DofKey> edges;
  for (const auto& face : cell.faces)
  {
    const size_t ne = face.vertex_ids.size();
    for (size_t e = 0; e < num_face_edges(face); ++e)
    {
      const uint64_t a = face.vertex_ids[e];
      const uint64_t b = face.vertex_ids[(e + 1) % ne];
      const DofKey key = a < b ? DofKey{a, b} : DofKey{b, a};
      if (edge_index.find(key) == edge_index.end())
      {
        edge_index[key] = static_cast<int>(edges.size());
        edges.push_back(key);
      }
    }
  }
  const size_t n_e = edges.size();

  // Dofs: vertices (cell order), then edge midpoints (p >= 2)
  dof_locations_.reserve(n_v + (p >= 2 ? n_e : 0));
  dof_keys_.reserve(dof_locations_.capacity());
  for (size_t i = 0; i < n_v; ++i)
  {
    dof_locations_.push_back(grid.GlobalVertex(cell.vertex_ids[i]));
    dof_keys_.emplace_back(cell.vertex_ids[i], cell.vertex_ids[i]);
  }
  if (p >= 2)
    for (const auto& key : edges)
    {
      dof_locations_.push_back((grid.GlobalVertex(key.first) + grid.GlobalVertex(key.second)) *
                               0.5);
      dof_keys_.push_back(key);
    }
  const size_t N = dof_locations_.size();

  // Face dofs: face vertices (loop order) then face edge midpoints (loop order)
  face_dofs_.resize(n_f);
  for (size_t f = 0; f < n_f; ++f)
  {
    const auto& face = cell.faces[f];
    const size_t ne = face.vertex_ids.size();
    for (const auto vid : face.vertex_ids)
      face_dofs_[f].push_back(vloc.at(vid));
    if (p >= 2)
      for (size_t e = 0; e < num_face_edges(face); ++e)
      {
        const uint64_t a = face.vertex_ids[e];
        const uint64_t b = face.vertex_ids[(e + 1) % ne];
        const DofKey key = a < b ? DofKey{a, b} : DofKey{b, a};
        face_dofs_[f].push_back(static_cast<int>(n_v) + edge_index.at(key));
      }
  }

  // Sub-mesh nodes: enumerate the reference nodes of every side. A polyhedron side is a tet
  // (v_k, x_f, v_{k+1}, x_c) using every reference node; a polygon side is the triangle
  // (v_k, v_{k+1}, x_c) using the nodes of the zeta = 0 face of the reference tet.
  const auto& ref_nodes = ref_.GetNodes();
  side_reference_nodes_.clear();
  for (size_t j = 0; j < ref_nodes.size(); ++j)
    if (dim_ == 3 or ref_nodes[j][3] == 0)
      side_reference_nodes_.push_back(j);
  const size_t n_side_nodes = side_reference_nodes_.size();
  std::map<std::vector<SupportEntry>, size_t> key_to_index;
  std::vector<std::vector<SupportEntry>> sub_node_keys;

  for (size_t f = 0; f < n_f; ++f)
  {
    const auto& face = cell.faces[f];
    const size_t ne = face.vertex_ids.size();
    for (size_t e = 0; e < num_face_edges(face); ++e)
    {
      const uint64_t v0 = face.vertex_ids[e];
      const uint64_t v1 = face.vertex_ids[(e + 1) % ne];
      std::array<Vector3, 4> X;
      std::array<SupportEntry, 4> subv;
      if (dim_ == 3)
      {
        X = {grid.GlobalVertex(v0), face.centroid, grid.GlobalVertex(v1), cell.centroid};
        subv = {SupportEntry{0, vloc.at(v0), 0},
                SupportEntry{1, static_cast<int>(f), 0},
                SupportEntry{0, vloc.at(v1), 0},
                SupportEntry{2, 0, 0}};
      }
      else
      {
        // The fourth entry is never used: the zeta = 0 nodes have a[3] == 0
        X = {grid.GlobalVertex(v0), grid.GlobalVertex(v1), cell.centroid, cell.centroid};
        subv = {SupportEntry{0, vloc.at(v0), 0},
                SupportEntry{0, vloc.at(v1), 0},
                SupportEntry{2, 0, 0},
                SupportEntry{2, 0, 0}};
      }

      side_face_.push_back(f);
      side_edge_.push_back(e);
      std::vector<size_t> nodes_of_side(n_side_nodes);
      for (size_t j = 0; j < n_side_nodes; ++j)
      {
        const auto& a = ref_nodes[side_reference_nodes_[j]];
        std::vector<SupportEntry> key;
        Vector3 x{0.0, 0.0, 0.0};
        for (unsigned int i = 0; i < 4; ++i)
        {
          if (a[i] == 0)
            continue;
          key.push_back(SupportEntry{subv[i].kind, subv[i].id, static_cast<int>(a[i])});
          x += X[i] * (static_cast<double>(a[i]) / static_cast<double>(p));
        }
        std::sort(key.begin(), key.end());

        auto it = key_to_index.find(key);
        size_t idx;
        if (it == key_to_index.end())
        {
          idx = sub_node_keys.size();
          key_to_index[key] = idx;
          sub_node_keys.push_back(key);
          sub_node_locations_.push_back(x);
          bool has_c = false, has_f = false;
          for (const auto& k : key)
          {
            has_c = has_c or (k.kind == 2);
            has_f = has_f or (k.kind == 1);
          }
          if (has_c)
          {
            sub_node_kinds_.push_back(SubmeshNodeKind::CELL_TIED);
            sub_node_face_.push_back(-1);
          }
          else if (has_f)
          {
            sub_node_kinds_.push_back(SubmeshNodeKind::FACE_TIED);
            sub_node_face_.push_back(static_cast<int>(f));
          }
          else
          {
            sub_node_kinds_.push_back(SubmeshNodeKind::REAL);
            sub_node_face_.push_back(-1);
          }
        }
        else
          idx = it->second;
        nodes_of_side[j] = idx;
      }
      side_nodes_.push_back(std::move(nodes_of_side));
    }
  }
  const size_t n_sub = sub_node_locations_.size();

  // Real sub-mesh node -> dof
  std::vector<int> real_to_dof(n_sub, -1);
  for (size_t i = 0; i < n_sub; ++i)
  {
    if (sub_node_kinds_[i] != SubmeshNodeKind::REAL)
      continue;
    const auto& key = sub_node_keys[i];
    if (key.size() == 1)
      real_to_dof[i] = key[0].id;
    else if (key.size() == 2)
    {
      const uint64_t a = cell.vertex_ids[key[0].id];
      const uint64_t b = cell.vertex_ids[key[1].id];
      const DofKey ek = a < b ? DofKey{a, b} : DofKey{b, a};
      real_to_dof[i] = static_cast<int>(n_v) + edge_index.at(ek);
    }
    else
      throw std::logic_error("SubmeshTying: unexpected real node support");
  }

  // Canonical orderings (by dof key) for the weight computations
  auto sorted_by_key = [this](std::vector<int> dofs)
  {
    std::sort(
      dofs.begin(), dofs.end(), [this](int a, int b) { return dof_keys_[a] < dof_keys_[b]; });
    return dofs;
  };
  std::vector<int> all_dofs(N);
  for (size_t j = 0; j < N; ++j)
    all_dofs[j] = static_cast<int>(j);
  const auto all_sorted = sorted_by_key(all_dofs);

  // Face frames are needed for the face-tied nodes, which exist only in 3D
  std::vector<FaceFrame> frames;
  frames.reserve(dim_ == 3 ? n_f : 0);
  for (size_t f = 0; f < n_f and dim_ == 3; ++f)
  {
    auto vids = cell.faces[f].vertex_ids;
    std::sort(vids.begin(), vids.end());
    std::vector<Vector3> fx;
    fx.reserve(vids.size());
    for (const auto vid : vids)
      fx.push_back(grid.GlobalVertex(vid));
    frames.push_back(MakeFaceFrame(fx));
  }

  std::vector<Vector3> cell_vertices;
  cell_vertices.reserve(n_v);
  for (size_t i = 0; i < n_v; ++i)
    cell_vertices.push_back(dof_locations_[i]);
  const double diameter = MaxPairwiseDistance(cell_vertices);

  // Tying operator
  T_ = DenseMatrix<double>(n_sub, N, 0.0);
  for (size_t i = 0; i < n_sub; ++i)
  {
    switch (sub_node_kinds_[i])
    {
      case SubmeshNodeKind::REAL:
        T_(i, real_to_dof[i]) = 1.0;
        break;
      case SubmeshNodeKind::FACE_TIED:
      {
        const int f = sub_node_face_[i];
        const auto fd = sorted_by_key(face_dofs_[f]);
        TyingNodeSet nodes;
        nodes.x.reserve(fd.size());
        nodes.keys.reserve(fd.size());
        for (const auto d : fd)
        {
          nodes.x.push_back(dof_locations_[d]);
          nodes.keys.push_back(dof_keys_[d]);
        }
        nodes.loops = {cell.faces[f].vertex_ids};
        nodes.centroid = cell.faces[f].centroid;
        const auto w = weights.FaceWeights(nodes, sub_node_locations_[i], frames[f]);
        for (size_t k = 0; k < fd.size(); ++k)
          T_(i, fd[k]) = w[k];
        break;
      }
      case SubmeshNodeKind::CELL_TIED:
      {
        TyingNodeSet nodes;
        nodes.x.reserve(N);
        nodes.keys.reserve(N);
        for (const auto d : all_sorted)
        {
          nodes.x.push_back(dof_locations_[d]);
          nodes.keys.push_back(dof_keys_[d]);
        }
        for (const auto& face : cell.faces)
          nodes.loops.push_back(face.vertex_ids);
        nodes.centroid = cell.centroid;
        const auto w = weights.CellWeights(nodes, sub_node_locations_[i], diameter);
        for (size_t k = 0; k < N; ++k)
          T_(i, all_sorted[k]) = w[k];
        break;
      }
    }
  }

  // Per-side restriction of T
  side_tying_.reserve(side_nodes_.size());
  for (const auto& nodes_of_side : side_nodes_)
  {
    DenseMatrix<double> Ts(n_side_nodes, N, 0.0);
    for (size_t t = 0; t < n_side_nodes; ++t)
      for (size_t j = 0; j < N; ++j)
        Ts(t, j) = T_(nodes_of_side[t], j);
    side_tying_.push_back(std::move(Ts));
  }
}

} // namespace opensn
