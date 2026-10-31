// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

// Unit tests of the serendipity tying of the PWQD basis (SerendipityTyingWeights): polynomial
// and nodal identities, the classical serendipity values at cell and face centres of
// parallelepipeds and squares, equality with the least-squares tying on simplices (both are
// exact interpolation there), the least-squares fallback on non-planar faces and degenerate
// vertex stars, and trace consistency across faces.

#include "test/unit/common/mesh_builders.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/submesh_tying.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/serendipity_tying_weights.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/piecewise_quadratic_polyhedron_mapping.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/piecewise_quadratic_polygon_mapping.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_quadratic/piecewise_quadratic_discontinuous.h"
#include "framework/math/spatial_discretization/finite_element/finite_element_data.h"
#include "framework/math/quadratures/spatial/tetrahedra_quadrature.h"
#include "framework/math/quadratures/spatial/triangle_quadrature.h"
#include "framework/math/quadratures/spatial/line_quadrature.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/mesh/cell.h"
#include "framework/runtime.h"
#include <gtest/gtest.h>
#include <filesystem>

using namespace opensn;
namespace fs = std::filesystem;

namespace
{

std::vector<double>
Quadratics(const Vector3& x)
{
  return {1.0, x.x, x.y, x.z, x.x * x.x, x.x * x.y, x.x * x.z, x.y * x.y, x.y * x.z, x.z * x.z};
}

std::vector<double>
Linears(const Vector3& x)
{
  return {1.0, x.x, x.y, x.z};
}

std::shared_ptr<Mesh>
AssetMesh(const std::string& name)
{
  return BuildMeshFromFile(fs::path(OPENSN_TEST_ROOT) / "assets" / "mesh" / name);
}

/// Mapping of a cell with the given weights, 3D or 2D.
std::unique_ptr<CellMapping>
MakeMapping(const Cell& cell,
            const std::shared_ptr<Mesh>& grid,
            const TetrahedraQuadrature& tet,
            const TriangleQuadrature& tri,
            const LineQuadrature& line,
            std::shared_ptr<const TyingWeightProvider> weights)
{
  if (cell.GetType() == CellType::POLYHEDRON)
    return std::make_unique<PieceWiseQuadraticPolyhedronMapping>(cell, grid, tet, tri, 2, weights);
  return std::make_unique<PieceWiseQuadraticPolygonMapping>(cell, grid, tri, line, 2, weights);
}

const SubmeshTying&
TyingOf(const CellMapping& m)
{
  if (const auto* p = dynamic_cast<const PieceWiseQuadraticPolyhedronMapping*>(&m))
    return p->GetTying();
  return dynamic_cast<const PieceWiseQuadraticPolygonMapping&>(m).GetTying();
}

/// Max error of T applied to the monomials of degree <= q (q = 1 or 2) at every sub-mesh node.
double
ReproductionError(const SubmeshTying& tying, unsigned int q)
{
  const auto& T = tying.GetTyingOperator();
  const auto& dof_x = tying.GetDofLocations();
  const auto& sub_x = tying.GetSubNodeLocations();
  double worst = 0.0;
  for (size_t i = 0; i < tying.GetNumSubNodes(); ++i)
  {
    const auto exact = q == 2 ? Quadratics(sub_x[i]) : Linears(sub_x[i]);
    std::vector<double> tied(exact.size(), 0.0);
    for (size_t j = 0; j < tying.GetNumDofs(); ++j)
    {
      const auto qj = q == 2 ? Quadratics(dof_x[j]) : Linears(dof_x[j]);
      for (size_t k = 0; k < exact.size(); ++k)
        tied[k] += T(i, j) * qj[k];
    }
    for (size_t k = 0; k < exact.size(); ++k)
      worst = std::max(worst, std::fabs(tied[k] - exact[k]));
  }
  return worst;
}

/// Max |b_i(x_j) - delta_ij| over the dof locations.
double
NodalError(const CellMapping& m)
{
  const auto& X = m.GetNodeLocations();
  double worst = 0.0;
  for (size_t j = 0; j < X.size(); ++j)
    for (size_t i = 0; i < X.size(); ++i)
      worst = std::max(worst, std::fabs(m.ShapeValue(i, X[j]) - (i == j ? 1.0 : 0.0)));
  return worst;
}

/// Max difference between the tying operators of two mappings of the same cell.
double
TyingDifference(const SubmeshTying& a, const SubmeshTying& b)
{
  const auto& Ta = a.GetTyingOperator();
  const auto& Tb = b.GetTyingOperator();
  double worst = 0.0;
  for (size_t i = 0; i < a.GetNumSubNodes(); ++i)
    for (size_t j = 0; j < a.GetNumDofs(); ++j)
      worst = std::max(worst, std::fabs(Ta(i, j) - Tb(i, j)));
  return worst;
}

struct Fixtures
{
  TetrahedraQuadrature tet{QuadratureOrder::FOURTH};
  TriangleQuadrature tri{QuadratureOrder::FOURTH};
  LineQuadrature line{QuadratureOrder::FOURTH};
  std::shared_ptr<SerendipityTyingWeights> serendipity =
    std::make_shared<SerendipityTyingWeights>();
  std::shared_ptr<LeastSquaresTyingWeights> lsq = std::make_shared<LeastSquaresTyingWeights>(2);
};

} // namespace

TEST(PWQDSerendipity, QuadraticPrecisionAndNodalOnPlanarCells)
{
  Fixtures fx;
  const std::vector<std::pair<std::string, std::shared_ptr<Mesh>>> meshes = {
    {"orthogonal hexes", BuildBoxMesh(1.0, 2, 0.0)},
    {"non-uniform hexes", BuildOrthogonalMesh({{0.0, 0.3, 1.0}, {0.0, 0.5, 1.0}, {0.0, 0.4, 1.0}})},
    {"sheared hexes", AssetMesh("pwqd_hex_sheared_2.vtu")},
    {"tets", AssetMesh("pwqd_tets_2.vtu")},
    {"quads", BuildSquareMesh(1.0, 2, 0.0)},
    {"triangles", AssetMesh("square_mesh2x2_small.obj")},
    {"polygons", AssetMesh("quad_mesh_poly_mix.obj")}};
  for (const auto& [name, grid] : meshes)
  {
    const size_t fallbacks_before = fx.serendipity->GetNumFallbacks();
    for (const auto& cell : grid->GetLocalCells())
    {
      auto m = MakeMapping(*cell, grid, fx.tet, fx.tri, fx.line, fx.serendipity);
      EXPECT_LT(ReproductionError(TyingOf(*m), 2), 1e-11) << name;
      EXPECT_LT(NodalError(*m), 1e-12) << name;
      // Face locality: functions of nodes off a face vanish on it
      const size_t N = m->GetNumNodes();
      for (size_t f = 0; f < cell->faces.size(); ++f)
      {
        std::vector<bool> on_face(N, false);
        for (size_t fi = 0; fi < m->GetNumFaceNodes(f); ++fi)
          on_face[m->MapFaceNode(f, fi)] = true;
        const auto srf = m->MakeSurfaceFiniteElementData(f);
        for (const auto qp : srf.GetQuadraturePointIndices())
          for (size_t i = 0; i < N; ++i)
            if (not on_face[i])
              EXPECT_NEAR(srf.ShapeValue(i, qp), 0.0, 1e-14) << name;
      }
    }
    // All these cells have planar faces and simple vertices: no fallback
    EXPECT_EQ(fx.serendipity->GetNumFallbacks(), fallbacks_before) << name;
  }
}

TEST(PWQDSerendipity, ClassicalSerendipityValuesAtCentres)
{
  // At the centre of a parallelepiped the 20-node serendipity element has the values -1/4 at
  // the corners and 1/4 at the edge midpoints; at the centre of a parallelogram face (and of a
  // rectangular cell) the 8-node element has -1/4 at the corners and 1/2 at the edge midpoints.
  // The construction is affine-invariant, so boxes of any aspect ratio give the cube's values
  // (the sheared asset is trapezoidal, not a parallelepiped, and is checked elsewhere).
  Fixtures fx;
  const std::vector<std::pair<std::string, std::shared_ptr<Mesh>>> meshes = {
    {"cubes", BuildBoxMesh(1.0, 2, 0.0)},
    {"boxes", BuildOrthogonalMesh({{0.0, 0.3, 1.0}, {0.0, 0.5, 1.0}, {0.0, 0.4, 1.0}})},
    {"squares", BuildSquareMesh(1.0, 2, 0.0)},
    {"rectangles", BuildOrthogonalMesh({{0.0, 0.3, 1.0}, {0.0, 0.5, 1.0}})}};
  for (const auto& [name, grid] : meshes)
    for (const auto& cell : grid->GetLocalCells())
    {
      auto m = MakeMapping(*cell, grid, fx.tet, fx.tri, fx.line, fx.serendipity);
      const auto& tying = TyingOf(*m);
      const auto& T = tying.GetTyingOperator();
      const auto& sub_x = tying.GetSubNodeLocations();
      const size_t n_v = cell->vertex_ids.size();
      const double vertex_value = -0.25;
      const double edge_value = cell->GetType() == CellType::POLYHEDRON ? 0.25 : 0.5;
      size_t checked = 0;
      for (size_t i = 0; i < tying.GetNumSubNodes(); ++i)
      {
        if ((sub_x[i] - cell->centroid).NormSquare() > 1e-24)
          continue;
        for (size_t j = 0; j < tying.GetNumDofs(); ++j)
          EXPECT_NEAR(T(i, j), j < n_v ? vertex_value : edge_value, 1e-13)
            << name << " cell " << cell->global_id << " dof " << j;
        ++checked;
      }
      EXPECT_EQ(checked, 1u);
      if (cell->GetType() != CellType::POLYHEDRON)
        continue;
      // Face centres: the 8-node values over the face's own dofs
      for (size_t f = 0; f < cell->faces.size(); ++f)
      {
        const auto& fdofs = tying.GetFaceDofs()[f];
        for (size_t i = 0; i < tying.GetNumSubNodes(); ++i)
        {
          if ((sub_x[i] - cell->faces[f].centroid).NormSquare() > 1e-24)
            continue;
          for (size_t j = 0; j < tying.GetNumDofs(); ++j)
          {
            const bool on_face =
              std::find(fdofs.begin(), fdofs.end(), static_cast<int>(j)) != fdofs.end();
            const double expected = not on_face ? 0.0 : (j < n_v ? -0.25 : 0.5);
            EXPECT_NEAR(T(i, j), expected, 1e-13) << name << " face " << f << " dof " << j;
          }
        }
      }
    }
}

TEST(PWQDSerendipity, EqualsLeastSquaresOnSimplicesAndDiffersOnHexes)
{
  Fixtures fx;
  for (const auto& name : {"pwqd_tets_2.vtu", "square_mesh2x2_small.obj"})
  {
    auto grid = AssetMesh(name);
    for (const auto& cell : grid->GetLocalCells())
    {
      auto a = MakeMapping(*cell, grid, fx.tet, fx.tri, fx.line, fx.serendipity);
      auto b = MakeMapping(*cell, grid, fx.tet, fx.tri, fx.line, fx.lsq);
      EXPECT_LT(TyingDifference(TyingOf(*a), TyingOf(*b)), 1e-12) << name;
    }
  }
  auto hexes = BuildBoxMesh(1.0, 2, 0.0);
  for (const auto& cell : hexes->GetLocalCells())
  {
    auto a = MakeMapping(*cell, hexes, fx.tet, fx.tri, fx.line, fx.serendipity);
    auto b = MakeMapping(*cell, hexes, fx.tet, fx.tri, fx.line, fx.lsq);
    EXPECT_GT(TyingDifference(TyingOf(*a), TyingOf(*b)), 1e-2);
  }
}

TEST(PWQDSerendipity, FallsBackToLeastSquares)
{
  Fixtures fx;
  // Non-planar faces (interior-perturbed hexes): linear precision and face locality are kept,
  // the fallback is used, and every fallback row equals the least-squares row
  auto perturbed = AssetMesh("pwqd_hex_perturbed_2.vtu");
  const size_t before = fx.serendipity->GetNumFallbacks();
  for (const auto& cell : perturbed->GetLocalCells())
  {
    auto a = MakeMapping(*cell, perturbed, fx.tet, fx.tri, fx.line, fx.serendipity);
    auto b = MakeMapping(*cell, perturbed, fx.tet, fx.tri, fx.line, fx.lsq);
    EXPECT_LT(ReproductionError(TyingOf(*a), 1), 1e-12);
    EXPECT_LT(NodalError(*a), 1e-12);
    const auto& ta = TyingOf(*a);
    const auto& tb = TyingOf(*b);
    const auto& Ta = ta.GetTyingOperator();
    const auto& Tb = tb.GetTyingOperator();
    // Cell-tied rows: every interior-perturbed cell has a non-planar face, so they fall back
    bool has_nonplanar = false;
    for (const auto& face : cell->faces)
    {
      std::vector<Vector3> fx_;
      for (const auto vid : face.vertex_ids)
        fx_.push_back(perturbed->GlobalVertex(vid));
      const auto frame = MakeFaceFrame(fx_);
      for (const auto& p : fx_)
        has_nonplanar = has_nonplanar or std::fabs(frame.Map(p).z) > 1e-9;
    }
    if (has_nonplanar)
      for (size_t i = 0; i < ta.GetNumSubNodes(); ++i)
        if (ta.GetSubNodeKinds()[i] == SubmeshNodeKind::CELL_TIED)
          for (size_t j = 0; j < ta.GetNumDofs(); ++j)
            EXPECT_NEAR(Ta(i, j), Tb(i, j), 1e-14);
  }
  EXPECT_GT(fx.serendipity->GetNumFallbacks(), before);

  // Degenerate vertex star: a square with a fifth vertex in the middle of one edge
  // (hanging node) falls back for the whole cell and keeps quadratic precision
  const Vector3 corners[5] = {
    {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}};
  TyingNodeSet nodes;
  for (uint64_t v = 0; v < 5; ++v)
  {
    nodes.x.push_back(corners[v]);
    nodes.keys.emplace_back(v, v);
  }
  for (uint64_t v = 0; v < 5; ++v)
  {
    const uint64_t w = (v + 1) % 5;
    nodes.x.push_back((corners[v] + corners[w]) * 0.5);
    nodes.keys.emplace_back(std::min(v, w), std::max(v, w));
    nodes.loops.push_back({v, w});
  }
  nodes.centroid = Vector3{0.5, 0.4, 0.0};
  const size_t before2 = fx.serendipity->GetNumFallbacks();
  const auto w_s = fx.serendipity->CellWeights(nodes, nodes.centroid, std::sqrt(2.0));
  const auto w_l = fx.lsq->CellWeights(nodes, nodes.centroid, std::sqrt(2.0));
  EXPECT_EQ(fx.serendipity->GetNumFallbacks(), before2 + 1);
  for (size_t k = 0; k < w_s.size(); ++k)
    EXPECT_NEAR(w_s[k], w_l[k], 1e-14);
}

TEST(PWQDSerendipity, DiscretizationTraceConsistency)
{
  const std::vector<std::pair<std::string, std::shared_ptr<Mesh>>> meshes = {
    {"hexes", BuildOrthogonalMesh({{0.0, 0.3, 1.0}, {0.0, 0.5, 1.0}, {0.0, 0.4, 1.0}})},
    {"tets", AssetMesh("pwqd_tets_2.vtu")},
    {"perturbed hexes", AssetMesh("pwqd_hex_perturbed_2.vtu")},
    {"polygons", AssetMesh("quad_mesh_poly_mix.obj")}};
  for (const auto& [name, grid] : meshes)
  {
    auto sdm =
      PieceWiseQuadraticDiscontinuous::New(grid, QuadratureOrder::FOURTH, PWQDTying::SERENDIPITY);
    EXPECT_EQ(sdm->GetTyingWeights().GetName(), "serendipity");
    const auto adj = sdm->MakeInternalFaceNodeMappings();
    for (const auto& cell : grid->GetLocalCells())
    {
      const auto& cm = sdm->GetCellMapping(*cell);
      for (size_t f = 0; f < cell->faces.size(); ++f)
      {
        const auto& face = cell->faces[f];
        if (not face.has_neighbor)
          continue;
        const auto& adj_cell = grid->GetGlobalCell(face.neighbor_id);
        const auto& acm = sdm->GetCellMapping(adj_cell);
        const size_t af = Mesh::MapCellFace(*cell, adj_cell, f);
        const auto srf = cm.MakeSurfaceFiniteElementData(f);
        const auto asrf = acm.MakeSurfaceFiniteElementData(af);
        for (const auto qp : srf.GetQuadraturePointIndices())
        {
          int aqp = -1;
          for (const auto q2 : asrf.GetQuadraturePointIndices())
            if ((srf.QPointXYZ(qp) - asrf.QPointXYZ(q2)).NormSquare() < 1e-24)
            {
              aqp = static_cast<int>(q2);
              break;
            }
          ASSERT_GE(aqp, 0) << name;
          for (size_t fi = 0; fi < cm.GetNumFaceNodes(f); ++fi)
          {
            const int i = cm.MapFaceNode(f, fi);
            const int ai = adj[cell->local_id][f][fi];
            EXPECT_NEAR(srf.ShapeValue(i, qp), asrf.ShapeValue(ai, aqp), 1e-13) << name;
          }
        }
      }
    }
  }
}
