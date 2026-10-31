// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/serendipity_tying_weights.h"
#include "framework/data_types/matrix3x3.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace opensn
{

namespace
{

/// Vertices, edges and vertex neighbours of a node set, by node-set index.
struct NodeSetTopology
{
  std::vector<size_t> vertex_nodes;              ///< node-set index of local vertex v
  std::map<uint64_t, size_t> vid_to_local;       ///< global vertex id -> local vertex v
  std::vector<std::array<size_t, 2>> edges;      ///< local vertices of edge e
  std::vector<size_t> edge_nodes;                ///< node-set index of edge e
  std::vector<std::vector<size_t>> neighbours;   ///< local vertices adjacent to local vertex v
  std::vector<std::vector<size_t>> vertex_edges; ///< edges incident to local vertex v
};

NodeSetTopology
BuildTopology(const TyingNodeSet& nodes)
{
  NodeSetTopology top;
  for (size_t k = 0; k < nodes.keys.size(); ++k)
    if (nodes.keys[k].first == nodes.keys[k].second)
    {
      top.vid_to_local[nodes.keys[k].first] = top.vertex_nodes.size();
      top.vertex_nodes.push_back(k);
    }
  const size_t n_v = top.vertex_nodes.size();
  top.neighbours.resize(n_v);
  top.vertex_edges.resize(n_v);
  for (size_t k = 0; k < nodes.keys.size(); ++k)
    if (nodes.keys[k].first != nodes.keys[k].second)
    {
      const auto a = top.vid_to_local.find(nodes.keys[k].first);
      const auto b = top.vid_to_local.find(nodes.keys[k].second);
      if (a == top.vid_to_local.end() or b == top.vid_to_local.end())
        throw std::logic_error("SerendipityTyingWeights: edge node without its vertices");
      const size_t e = top.edges.size();
      top.edges.push_back({a->second, b->second});
      top.edge_nodes.push_back(k);
      top.neighbours[a->second].push_back(b->second);
      top.neighbours[b->second].push_back(a->second);
      top.vertex_edges[a->second].push_back(e);
      top.vertex_edges[b->second].push_back(e);
    }
  return top;
}

/// Lagrange serendipity values from the PWL coordinates phi (per local vertex) and the vertex-star
/// coordinates star[v] = (lambda_{v,v}, lambda_{v,n} for n in neighbours[v]), returned per node-set
/// index.
std::vector<double>
LagrangeFromCoordinates(const NodeSetTopology& top,
                        size_t num_nodes,
                        const std::vector<double>& phi,
                        const std::vector<std::vector<double>>& star)
{
  const size_t n_v = top.vertex_nodes.size();
  auto lambda = [&top, &star](size_t v, size_t neighbour)
  {
    const auto& nb = top.neighbours[v];
    for (size_t k = 0; k < nb.size(); ++k)
      if (nb[k] == neighbour)
        return star[v][k + 1];
    throw std::logic_error("SerendipityTyingWeights: not a neighbour");
  };
  std::vector<double> F_edge(top.edges.size(), 0.0);
  for (size_t e = 0; e < top.edges.size(); ++e)
  {
    const auto [i, j] = top.edges[e];
    F_edge[e] = phi[i] * lambda(i, j) + phi[j] * lambda(j, i);
  }
  std::vector<double> w(num_nodes, 0.0);
  for (size_t v = 0; v < n_v; ++v)
  {
    double L = phi[v] * star[v][0];
    for (const auto e : top.vertex_edges[v])
      L -= 0.5 * F_edge[e];
    w[top.vertex_nodes[v]] = L;
  }
  for (size_t e = 0; e < top.edges.size(); ++e)
    w[top.edge_nodes[e]] = 2.0 * F_edge[e];
  return w;
}

/// Barycentric coordinates (l1, l2) of x in the triangle (p0, p1, p2) of the plane z = 0; l0 is
/// 1 - l1 - l2.
std::array<double, 3>
Barycentric2D(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& x)
{
  const double a = p1.x - p0.x, b = p2.x - p0.x, c = p1.y - p0.y, d = p2.y - p0.y;
  const double det = a * d - b * c;
  if (det == 0.0)
    throw std::invalid_argument("SerendipityTyingWeights: degenerate triangle");
  const double rx = x.x - p0.x, ry = x.y - p0.y;
  const double l1 = (d * rx - b * ry) / det;
  const double l2 = (-c * rx + a * ry) / det;
  return {1.0 - l1 - l2, l1, l2};
}

/// Barycentric coordinates of x in the tetrahedron (p0, p1, p2, p3).
std::array<double, 4>
Barycentric3D(
  const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& x)
{
  Matrix3x3 J;
  J.SetColJVec(0, p1 - p0);
  J.SetColJVec(1, p2 - p0);
  J.SetColJVec(2, p3 - p0);
  if (J.Det() == 0.0)
    throw std::invalid_argument("SerendipityTyingWeights: degenerate tetrahedron");
  const Vector3 l = J.Inverse() * (x - p0);
  return {1.0 - l.x - l.y - l.z, l.x, l.y, l.z};
}

constexpr double kInsideTol = 1.0e-10;
/// A vertex star whose simplex volume is below this fraction of diameter^dim is degenerate (a
/// vertex collinear or coplanar with its neighbours, as at a hanging node).
constexpr double kStarTol = 1.0e-12;

double
Diameter(const std::vector<Vector3>& v)
{
  double d2 = 0.0;
  for (size_t i = 0; i < v.size(); ++i)
    for (size_t j = i + 1; j < v.size(); ++j)
      d2 = std::max(d2, (v[i] - v[j]).NormSquare());
  return std::sqrt(d2);
}

/// True when every vertex has the expected number of neighbours and its star simplex is not
/// degenerate.
bool
StarsNondegenerate(const NodeSetTopology& top, const std::vector<Vector3>& v, unsigned int dim)
{
  const double diam = Diameter(v);
  for (size_t i = 0; i < v.size(); ++i)
  {
    const auto& nb = top.neighbours[i];
    if (nb.size() != dim)
      return false;
    double volume = 0.0;
    if (dim == 2)
    {
      const Vector3 a = v[nb[0]] - v[i], b = v[nb[1]] - v[i];
      volume = std::fabs(a.x * b.y - a.y * b.x);
    }
    else
    {
      Matrix3x3 J;
      J.SetColJVec(0, v[nb[0]] - v[i]);
      J.SetColJVec(1, v[nb[1]] - v[i]);
      J.SetColJVec(2, v[nb[2]] - v[i]);
      volume = std::fabs(J.Det());
    }
    if (volume <= kStarTol * std::pow(diam, static_cast<double>(dim)))
      return false;
  }
  return true;
}

/**
 * Serendipity weights of a polygon in the plane z = 0: vertices v (local order), edges from the
 * topology, PWL coordinates on the fan of side triangles (v_a, v_b, c) around the centroid c,
 * vertex-star coordinates from the triangle (v, its two neighbours).
 */
std::vector<double>
Serendipity2D(const NodeSetTopology& top,
              size_t num_nodes,
              const std::vector<Vector3>& v,
              const Vector3& c,
              const Vector3& x)
{
  const size_t n_v = v.size();
  // PWL coordinates: locate x in a fan triangle (v_a, v_b, c); phi_i = l_a d_ia + l_b d_ib + l_c /
  // n_v
  std::vector<double> phi(n_v, 0.0);
  bool found = false;
  for (const auto& [a, b] : top.edges)
  {
    const auto l = Barycentric2D(v[a], v[b], c, x);
    if (l[0] >= -kInsideTol and l[1] >= -kInsideTol and l[2] >= -kInsideTol)
    {
      phi[a] += l[0];
      phi[b] += l[1];
      for (size_t i = 0; i < n_v; ++i)
        phi[i] += l[2] / static_cast<double>(n_v);
      found = true;
      break;
    }
  }
  if (not found)
    throw std::logic_error("SerendipityTyingWeights: tied point outside the polygon");

  std::vector<std::vector<double>> star(n_v);
  for (size_t i = 0; i < n_v; ++i)
  {
    const auto& nb = top.neighbours[i];
    if (nb.size() != 2)
      throw std::invalid_argument("SerendipityTyingWeights: a polygon vertex must have two "
                                  "neighbours");
    const auto l = Barycentric2D(v[i], v[nb[0]], v[nb[1]], x);
    star[i] = {l[0], l[1], l[2]};
  }
  return LagrangeFromCoordinates(top, num_nodes, phi, star);
}

/**
 * Serendipity weights of a polyhedron with simple vertices: PWL coordinates on the side
 * tetrahedra (v_k, x_f, v_{k+1}, x_c) of the face loops, vertex-star coordinates from the
 * tetrahedron (v, its three neighbours).
 */
std::vector<double>
Serendipity3D(const NodeSetTopology& top,
              size_t num_nodes,
              const std::vector<Vector3>& v,
              const std::vector<std::vector<size_t>>& loops,
              const Vector3& c,
              const Vector3& x)
{
  const size_t n_v = v.size();
  std::vector<double> phi(n_v, 0.0);
  bool found = false;
  for (const auto& loop : loops)
  {
    const size_t nf = loop.size();
    Vector3 xf{0.0, 0.0, 0.0};
    for (const auto i : loop)
      xf += v[i];
    xf /= static_cast<double>(nf);
    for (size_t k = 0; k < nf and not found; ++k)
    {
      const size_t a = loop[k], b = loop[(k + 1) % nf];
      const auto l = Barycentric3D(v[a], xf, v[b], c, x);
      if (l[0] >= -kInsideTol and l[1] >= -kInsideTol and l[2] >= -kInsideTol and
          l[3] >= -kInsideTol)
      {
        phi[a] += l[0];
        phi[b] += l[2];
        for (const auto i : loop)
          phi[i] += l[1] / static_cast<double>(nf);
        for (size_t i = 0; i < n_v; ++i)
          phi[i] += l[3] / static_cast<double>(n_v);
        found = true;
      }
    }
    if (found)
      break;
  }
  if (not found)
    throw std::logic_error("SerendipityTyingWeights: tied point outside the polyhedron");

  std::vector<std::vector<double>> star(n_v);
  for (size_t i = 0; i < n_v; ++i)
  {
    const auto& nb = top.neighbours[i];
    if (nb.size() != 3)
      throw std::invalid_argument("SerendipityTyingWeights: vertex is not simple");
    const auto l = Barycentric3D(v[i], v[nb[0]], v[nb[1]], v[nb[2]], x);
    star[i] = {l[0], l[1], l[2], l[3]};
  }
  return LagrangeFromCoordinates(top, num_nodes, phi, star);
}

/// Largest out-of-plane coordinate of the vertices of a loop relative to its diameter.
double
Nonplanarity(const std::vector<Vector3>& loop_vertices)
{
  if (loop_vertices.size() < 4)
    return 0.0;
  const auto frame = MakeFaceFrame(loop_vertices);
  double worst = 0.0;
  for (const auto& p : loop_vertices)
    worst = std::max(worst, std::fabs(frame.Map(p).z));
  return worst;
}

} // namespace

SerendipityTyingWeights::SerendipityTyingWeights(double planarity_tol)
  : planarity_tol_(planarity_tol), fallback_(2)
{
}

std::vector<double>
SerendipityTyingWeights::FaceWeights(const TyingNodeSet& nodes,
                                     const Vector3& x_tied,
                                     const FaceFrame& frame) const
{
  const auto top = BuildTopology(nodes);
  // Non-planar face: the star coordinates do not vanish on it, so the trace would not be
  // face-local; use the least-squares fit
  double worst = 0.0;
  for (const auto k : top.vertex_nodes)
    worst = std::max(worst, std::fabs(frame.Map(nodes.x[k]).z));
  if (worst > planarity_tol_)
  {
    ++num_fallbacks_;
    return fallback_.FaceWeights(nodes, x_tied, frame);
  }
  // The two-dimensional element of the face polygon in the frame coordinates
  std::vector<Vector3> v;
  v.reserve(top.vertex_nodes.size());
  for (const auto k : top.vertex_nodes)
  {
    const Vector3 s = frame.Map(nodes.x[k]);
    v.push_back({s.x, s.y, 0.0});
  }
  if (not StarsNondegenerate(top, v, 2))
  {
    ++num_fallbacks_;
    return fallback_.FaceWeights(nodes, x_tied, frame);
  }
  const Vector3 c3 = frame.Map(nodes.centroid);
  const Vector3 x3 = frame.Map(x_tied);
  return Serendipity2D(top, nodes.x.size(), v, {c3.x, c3.y, 0.0}, {x3.x, x3.y, 0.0});
}

std::vector<double>
SerendipityTyingWeights::CellWeights(const TyingNodeSet& nodes,
                                     const Vector3& x_tied,
                                     double scale) const
{
  const auto top = BuildTopology(nodes);
  std::vector<Vector3> v;
  v.reserve(top.vertex_nodes.size());
  for (const auto k : top.vertex_nodes)
    v.push_back(nodes.x[k]);

  const bool polygon = not nodes.loops.empty() and nodes.loops.front().size() == 2;
  if (polygon)
  {
    if (not StarsNondegenerate(top, v, 2))
    {
      ++num_fallbacks_;
      return fallback_.CellWeights(nodes, x_tied, scale);
    }
    return Serendipity2D(top, nodes.x.size(), v, nodes.centroid, x_tied);
  }

  // Polyhedron: every vertex simple with a non-degenerate star and every face planar, else
  // least squares
  bool applicable = StarsNondegenerate(top, v, 3);
  std::vector<std::vector<size_t>> loops;
  loops.reserve(nodes.loops.size());
  for (const auto& loop : nodes.loops)
  {
    std::vector<size_t> local;
    std::vector<Vector3> lx;
    for (const auto vid : loop)
    {
      local.push_back(top.vid_to_local.at(vid));
      lx.push_back(v[local.back()]);
    }
    applicable = applicable and Nonplanarity(lx) <= planarity_tol_;
    loops.push_back(std::move(local));
  }
  if (not applicable)
  {
    ++num_fallbacks_;
    return fallback_.CellWeights(nodes, x_tied, scale);
  }
  return Serendipity3D(top, nodes.x.size(), v, loops, nodes.centroid, x_tied);
}

} // namespace opensn
