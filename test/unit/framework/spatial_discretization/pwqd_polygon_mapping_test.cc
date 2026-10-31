// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

// Unit tests of the PWQD polygon (2D) mapping. Every check is a polynomial or geometric
// identity, or an equality with an independently constructed element (PWLD at p = 1, Lagrange
// P2 on triangles).

#include "test/unit/common/mesh_builders.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/submesh_tying.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/piecewise_quadratic_polygon_mapping.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_linear/piecewise_linear_polygon_mapping.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_quadratic/piecewise_quadratic_discontinuous.h"
#include "framework/math/spatial_discretization/finite_element/finite_element_data.h"
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

/// The 6 monomials of total degree <= 2 in (x, y) at x.
std::vector<double>
Quadratics2D(const Vector3& x)
{
  return {1.0, x.x, x.y, x.x * x.x, x.x * x.y, x.y * x.y};
}

std::shared_ptr<Mesh>
AssetMesh(const std::string& name)
{
  return BuildMeshFromFile(fs::path(OPENSN_TEST_ROOT) / "assets" / "mesh" / name);
}

struct NamedMesh
{
  std::string name;
  std::shared_ptr<Mesh> grid;
};

std::vector<NamedMesh>
TestMeshes()
{
  return {{"orthogonal quads", BuildSquareMesh(1.0, 2, 0.0)},
          {"non-uniform quads", BuildOrthogonalMesh({{0.0, 0.3, 1.0}, {0.0, 0.5, 1.0}})},
          {"triangles", AssetMesh("square_mesh2x2_small.obj")},
          {"triangles to hexagons", AssetMesh("quad_mesh_poly_mix.obj")}};
}

/// Cholesky factorization succeeds iff the symmetric matrix is positive definite.
bool
IsSPD(const DenseMatrix<double>& M, double sym_tol)
{
  const size_t n = M.Rows();
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      if (std::fabs(M(i, j) - M(j, i)) > sym_tol)
        return false;
  DenseMatrix<double> L(n, n, 0.0);
  for (size_t j = 0; j < n; ++j)
  {
    double d = M(j, j);
    for (size_t k = 0; k < j; ++k)
      d -= L(j, k) * L(j, k);
    if (d <= 0.0)
      return false;
    L(j, j) = std::sqrt(d);
    for (size_t i = j + 1; i < n; ++i)
    {
      double s = M(i, j);
      for (size_t k = 0; k < j; ++k)
        s -= L(i, k) * L(j, k);
      L(i, j) = s / L(j, j);
    }
  }
  return true;
}

/// Edge length of face f of a polygon.
double
EdgeLength(const Mesh& grid, const Cell& cell, size_t f)
{
  const auto& face = cell.faces[f];
  return (grid.GlobalVertex(face.vertex_ids[1]) - grid.GlobalVertex(face.vertex_ids[0])).Norm();
}

} // namespace

TEST(PWQDPolygonMapping, DimensionCount)
{
  for (const auto& tm : TestMeshes())
  {
    ASSERT_EQ(tm.grid->GetDimension(), 2u) << tm.name;
    TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
    LineQuadrature line_quad(QuadratureOrder::FOURTH);
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      PieceWiseQuadraticPolygonMapping m(*cell, tm.grid, tri_quad, line_quad);
      const size_t n_v = cell->vertex_ids.size();
      const size_t n_f = cell->faces.size();
      ASSERT_EQ(n_f, n_v) << tm.name;
      EXPECT_EQ(m.GetNumNodes(), 2 * n_v) << tm.name;
      EXPECT_EQ(m.GetTying().GetDimension(), 2u);
      EXPECT_EQ(m.GetTying().GetSideReferenceNodes().size(), 6u);
      const auto& X = m.GetNodeLocations();
      // Vertex nodes first, at the vertex locations
      for (size_t i = 0; i < n_v; ++i)
        EXPECT_NEAR((X[i] - tm.grid->GlobalVertex(cell->vertex_ids[i])).Norm(), 0.0, 1e-15);
      // Three face nodes per edge: its two vertices and its midpoint
      for (size_t f = 0; f < n_f; ++f)
      {
        ASSERT_EQ(m.GetNumFaceNodes(f), 3u) << tm.name;
        const auto& face = cell->faces[f];
        const Vector3 a = tm.grid->GlobalVertex(face.vertex_ids[0]);
        const Vector3 b = tm.grid->GlobalVertex(face.vertex_ids[1]);
        EXPECT_NEAR((X[m.MapFaceNode(f, 0)] - a).Norm(), 0.0, 1e-15);
        EXPECT_NEAR((X[m.MapFaceNode(f, 1)] - b).Norm(), 0.0, 1e-15);
        EXPECT_NEAR((X[m.MapFaceNode(f, 2)] - (a + b) * 0.5).Norm(), 0.0, 1e-15);
        EXPECT_GE(m.MapFaceNode(f, 2), static_cast<int>(n_v));
      }
    }
  }
}

TEST(PWQDPolygonMapping, TyingReproducesQuadratics)
{
  for (const auto& tm : TestMeshes())
  {
    TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
    LineQuadrature line_quad(QuadratureOrder::FOURTH);
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      PieceWiseQuadraticPolygonMapping m(*cell, tm.grid, tri_quad, line_quad);
      const auto& tying = m.GetTying();
      const auto& T = tying.GetTyingOperator();
      const auto& dof_x = tying.GetDofLocations();
      const auto& sub_x = tying.GetSubNodeLocations();
      const size_t N = tying.GetNumDofs();
      size_t n_tied = 0;
      for (size_t i = 0; i < tying.GetNumSubNodes(); ++i)
      {
        EXPECT_NE(tying.GetSubNodeKinds()[i], SubmeshNodeKind::FACE_TIED) << tm.name;
        n_tied += tying.GetSubNodeKinds()[i] == SubmeshNodeKind::CELL_TIED ? 1 : 0;
        const auto exact = Quadratics2D(sub_x[i]);
        std::vector<double> tied(6, 0.0);
        for (size_t j = 0; j < N; ++j)
        {
          const auto qj = Quadratics2D(dof_x[j]);
          for (size_t k = 0; k < 6; ++k)
            tied[k] += T(i, j) * qj[k];
        }
        for (size_t k = 0; k < 6; ++k)
          EXPECT_NEAR(tied[k], exact[k], 1e-11)
            << tm.name << " sub-node " << i << " monomial " << k;
      }
      // Cell-tied nodes: the spoke midpoints and the centroid
      EXPECT_EQ(n_tied, cell->vertex_ids.size() + 1) << tm.name;
    }
  }
}

TEST(PWQDPolygonMapping, FiniteElementDataIdentities)
{
  for (const auto& tm : TestMeshes())
  {
    TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
    LineQuadrature line_quad(QuadratureOrder::FOURTH);
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      PieceWiseQuadraticPolygonMapping m(*cell, tm.grid, tri_quad, line_quad);
      const size_t N = m.GetNumNodes();
      const auto vol = m.MakeVolumetricFiniteElementData();

      double area = 0.0;
      DenseMatrix<double> M(N, N, 0.0);
      std::vector<Vector3> intV_grad(N, Vector3{0.0, 0.0, 0.0});
      for (const auto qp : vol.GetQuadraturePointIndices())
      {
        double sum = 0.0;
        Vector3 gsum{0.0, 0.0, 0.0};
        for (size_t i = 0; i < N; ++i)
        {
          sum += vol.ShapeValue(i, qp);
          gsum += vol.ShapeGrad(i, qp);
          EXPECT_EQ(vol.ShapeGrad(i, qp).z, 0.0);
        }
        EXPECT_NEAR(sum, 1.0, 1e-13) << tm.name;
        EXPECT_NEAR(gsum.Norm(), 0.0, 1e-12) << tm.name;
        area += vol.JxW(qp);
        for (size_t i = 0; i < N; ++i)
        {
          intV_grad[i] += vol.ShapeGrad(i, qp) * vol.JxW(qp);
          for (size_t j = 0; j < N; ++j)
            M(i, j) += vol.ShapeValue(i, qp) * vol.ShapeValue(j, qp) * vol.JxW(qp);
        }
      }
      EXPECT_NEAR(area, cell->volume, 1e-13 * cell->volume) << tm.name;
      double msum = 0.0;
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
          msum += M(i, j);
      EXPECT_NEAR(msum, cell->volume, 1e-12 * cell->volume) << tm.name;
      EXPECT_TRUE(IsSPD(M, 1e-14)) << tm.name;

      // Edges: partition of unity, edge lengths, divergence theorem int grad b_j = oint b_j n
      std::vector<Vector3> intS_b_n(N, Vector3{0.0, 0.0, 0.0});
      for (size_t f = 0; f < cell->faces.size(); ++f)
      {
        const auto srf = m.MakeSurfaceFiniteElementData(f);
        double length = 0.0;
        for (const auto qp : srf.GetQuadraturePointIndices())
        {
          double sum = 0.0;
          for (size_t i = 0; i < N; ++i)
            sum += srf.ShapeValue(i, qp);
          EXPECT_NEAR(sum, 1.0, 1e-13) << tm.name;
          length += srf.JxW(qp);
          for (size_t i = 0; i < N; ++i)
            intS_b_n[i] += srf.Normal(qp) * (srf.ShapeValue(i, qp) * srf.JxW(qp));
        }
        const double exact = EdgeLength(*tm.grid, *cell, f);
        EXPECT_NEAR(length, exact, 1e-13 * exact) << tm.name;
      }
      for (size_t j = 0; j < N; ++j)
        EXPECT_NEAR((intV_grad[j] - intS_b_n[j]).Norm(), 0.0, 1e-12) << tm.name << " node " << j;
    }
  }
}

TEST(PWQDPolygonMapping, TraceLocality)
{
  for (const auto& tm : TestMeshes())
  {
    TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
    LineQuadrature line_quad(QuadratureOrder::FOURTH);
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      PieceWiseQuadraticPolygonMapping m(*cell, tm.grid, tri_quad, line_quad);
      const size_t N = m.GetNumNodes();
      for (size_t f = 0; f < cell->faces.size(); ++f)
      {
        std::vector<bool> on_face(N, false);
        for (size_t fi = 0; fi < m.GetNumFaceNodes(f); ++fi)
          on_face[m.MapFaceNode(f, fi)] = true;
        const auto srf = m.MakeSurfaceFiniteElementData(f);
        EXPECT_EQ(srf.GetNumNodes(), m.GetNumFaceNodes(f));
        for (const auto qp : srf.GetQuadraturePointIndices())
          for (size_t i = 0; i < N; ++i)
            if (not on_face[i])
              EXPECT_NEAR(srf.ShapeValue(i, qp), 0.0, 1e-14) << tm.name;
      }
    }
  }
}

TEST(PWQDPolygonMapping, TraceConsistencyAcrossEdges)
{
  for (const auto& tm : TestMeshes())
  {
    auto sdm = PieceWiseQuadraticDiscontinuous::New(tm.grid);
    // Node locations of shared edges agree (throws otherwise)
    const auto adj = sdm->MakeInternalFaceNodeMappings();
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      const auto& cm = sdm->GetCellMapping(*cell);
      for (size_t f = 0; f < cell->faces.size(); ++f)
      {
        const auto& face = cell->faces[f];
        if (not face.has_neighbor)
          continue;
        const auto& adj_cell = tm.grid->GetGlobalCell(face.neighbor_id);
        const auto& acm = sdm->GetCellMapping(adj_cell);
        const size_t af = Mesh::MapCellFace(*cell, adj_cell, f);
        const auto srf = cm.MakeSurfaceFiniteElementData(f);
        const auto asrf = acm.MakeSurfaceFiniteElementData(af);
        // Match quadrature points by location (the edge runs the opposite way)
        for (const auto qp : srf.GetQuadraturePointIndices())
        {
          int aqp = -1;
          for (const auto q2 : asrf.GetQuadraturePointIndices())
            if ((srf.QPointXYZ(qp) - asrf.QPointXYZ(q2)).NormSquare() < 1e-24)
            {
              aqp = static_cast<int>(q2);
              break;
            }
          ASSERT_GE(aqp, 0) << tm.name << ": edge quadrature points do not coincide";
          EXPECT_NEAR(srf.JxW(qp), asrf.JxW(aqp), 1e-14);
          for (size_t fi = 0; fi < cm.GetNumFaceNodes(f); ++fi)
          {
            const int i = cm.MapFaceNode(f, fi);
            const int ai = adj[cell->local_id][f][fi];
            EXPECT_NEAR(srf.ShapeValue(i, qp), asrf.ShapeValue(ai, aqp), 1e-13) << tm.name;
          }
        }
      }
    }
  }
}

TEST(PWQDPolygonMapping, ReducesToPWLDAtDegreeOne)
{
  auto average = std::make_shared<AverageTyingWeights>();
  for (const auto& tm : TestMeshes())
  {
    TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
    LineQuadrature line_quad(QuadratureOrder::FOURTH);
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      PieceWiseLinearPolygonMapping pwld(*cell, tm.grid, tri_quad, line_quad);
      PieceWiseQuadraticPolygonMapping p1(*cell, tm.grid, tri_quad, line_quad, 1, average);
      ASSERT_EQ(p1.GetNumNodes(), pwld.GetNumNodes());
      const size_t N = p1.GetNumNodes();
      for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR((p1.GetNodeLocations()[i] - pwld.GetNodeLocations()[i]).Norm(), 0.0, 0.0);
      EXPECT_EQ(p1.GetFaceNodeMappings(), pwld.GetFaceNodeMappings());

      const auto a = p1.MakeVolumetricFiniteElementData();
      const auto b = pwld.MakeVolumetricFiniteElementData();
      ASSERT_EQ(a.GetQuadraturePointIndices().size(), b.GetQuadraturePointIndices().size());
      for (const auto qp : a.GetQuadraturePointIndices())
      {
        EXPECT_NEAR((a.QPointXYZ(qp) - b.QPointXYZ(qp)).Norm(), 0.0, 1e-14);
        EXPECT_NEAR(a.JxW(qp), b.JxW(qp), 1e-15);
        for (size_t i = 0; i < N; ++i)
        {
          EXPECT_NEAR(a.ShapeValue(i, qp), b.ShapeValue(i, qp), 1e-13) << tm.name;
          EXPECT_NEAR((a.ShapeGrad(i, qp) - b.ShapeGrad(i, qp)).Norm(), 0.0, 1e-12) << tm.name;
        }
      }
      for (size_t f = 0; f < cell->faces.size(); ++f)
      {
        const auto sa = p1.MakeSurfaceFiniteElementData(f);
        const auto sb = pwld.MakeSurfaceFiniteElementData(f);
        ASSERT_EQ(sa.GetQuadraturePointIndices().size(), sb.GetQuadraturePointIndices().size());
        EXPECT_EQ(sa.GetNumNodes(), sb.GetNumNodes());
        for (const auto qp : sa.GetQuadraturePointIndices())
        {
          EXPECT_NEAR((sa.QPointXYZ(qp) - sb.QPointXYZ(qp)).Norm(), 0.0, 1e-14);
          EXPECT_NEAR(sa.JxW(qp), sb.JxW(qp), 1e-15);
          EXPECT_NEAR((sa.Normal(qp) - sb.Normal(qp)).Norm(), 0.0, 0.0);
          for (size_t i = 0; i < N; ++i)
            EXPECT_NEAR(sa.ShapeValue(i, qp), sb.ShapeValue(i, qp), 1e-13) << tm.name;
        }
      }
      // Point evaluation at a random interior point agrees too
      const Vector3 x = cell->centroid * 0.7 + tm.grid->GlobalVertex(cell->vertex_ids[0]) * 0.3;
      for (size_t i = 0; i < N; ++i)
      {
        EXPECT_NEAR(p1.ShapeValue(i, x), pwld.ShapeValue(i, x), 1e-13) << tm.name;
        EXPECT_NEAR((p1.GradShapeValue(i, x) - pwld.GradShapeValue(i, x)).Norm(), 0.0, 1e-12);
      }
    }
  }
}

TEST(PWQDPolygonMapping, EqualsLagrangeP2OnTriangles)
{
  auto grid = AssetMesh("square_mesh2x2_small.obj");
  TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
  LineQuadrature line_quad(QuadratureOrder::FOURTH);
  // Lagrange P2 on a triangle: vertices lam_i (2 lam_i - 1), edges (i, j) 4 lam_i lam_j
  const std::array<std::pair<int, int>, 6> nodes = {
    std::pair{0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {2, 0}};
  for (const auto& cell : grid->GetLocalCells())
  {
    ASSERT_EQ(cell->vertex_ids.size(), 3u);
    PieceWiseQuadraticPolygonMapping m(*cell, grid, tri_quad, line_quad);
    ASSERT_EQ(m.GetNumNodes(), 6u);

    // Affine map of the whole triangle: x = X0 + A (lam1, lam2)
    const Vector3 X0 = grid->GlobalVertex(cell->vertex_ids[0]);
    Matrix3x3 A;
    A.SetColJVec(0, grid->GlobalVertex(cell->vertex_ids[1]) - X0);
    A.SetColJVec(1, grid->GlobalVertex(cell->vertex_ids[2]) - X0);
    A.SetIJ(2, 2, 1.0);
    const Matrix3x3 Ainv = A.Inverse();
    // Gradients of the barycentric coordinates: rows of A^{-1}, lam0 = 1 - lam1 - lam2
    std::array<Vector3, 3> dlam;
    dlam[1] = Vector3{Ainv.GetIJ(0, 0), Ainv.GetIJ(0, 1), 0.0};
    dlam[2] = Vector3{Ainv.GetIJ(1, 0), Ainv.GetIJ(1, 1), 0.0};
    dlam[0] = (dlam[1] + dlam[2]) * -1.0;

    // Match the PWQD nodes to the Lagrange nodes by location
    std::vector<int> pwqd_to_lagrange(6, -1);
    for (size_t i = 0; i < 6; ++i)
      for (size_t a = 0; a < 6; ++a)
      {
        const Vector3 xa = (grid->GlobalVertex(cell->vertex_ids[nodes[a].first]) +
                            grid->GlobalVertex(cell->vertex_ids[nodes[a].second])) *
                           0.5;
        if ((xa - m.GetNodeLocations()[i]).NormSquare() < 1e-24)
          pwqd_to_lagrange[i] = static_cast<int>(a);
      }
    for (size_t i = 0; i < 6; ++i)
      ASSERT_GE(pwqd_to_lagrange[i], 0);

    auto lagrange = [&](const Vector3& x, std::vector<double>& N, std::vector<Vector3>& dN)
    {
      const Vector3 l12 = Ainv * (x - X0);
      const std::array<double, 3> lam = {1.0 - l12.x - l12.y, l12.x, l12.y};
      N.assign(6, 0.0);
      dN.assign(6, Vector3{0.0, 0.0, 0.0});
      for (size_t a = 0; a < 6; ++a)
      {
        const int i = nodes[a].first, j = nodes[a].second;
        if (i == j)
        {
          N[a] = lam[i] * (2.0 * lam[i] - 1.0);
          dN[a] = dlam[i] * (4.0 * lam[i] - 1.0);
        }
        else
        {
          N[a] = 4.0 * lam[i] * lam[j];
          dN[a] = dlam[i] * (4.0 * lam[j]) + dlam[j] * (4.0 * lam[i]);
        }
      }
    };

    std::vector<double> N;
    std::vector<Vector3> dN;
    const auto vol = m.MakeVolumetricFiniteElementData();
    for (const auto qp : vol.GetQuadraturePointIndices())
    {
      lagrange(vol.QPointXYZ(qp), N, dN);
      for (size_t i = 0; i < 6; ++i)
      {
        EXPECT_NEAR(vol.ShapeValue(i, qp), N[pwqd_to_lagrange[i]], 1e-12);
        EXPECT_NEAR((vol.ShapeGrad(i, qp) - dN[pwqd_to_lagrange[i]]).Norm(), 0.0, 1e-11);
      }
    }
    for (size_t f = 0; f < 3; ++f)
    {
      const auto srf = m.MakeSurfaceFiniteElementData(f);
      for (const auto qp : srf.GetQuadraturePointIndices())
      {
        lagrange(srf.QPointXYZ(qp), N, dN);
        for (size_t i = 0; i < 6; ++i)
          EXPECT_NEAR(srf.ShapeValue(i, qp), N[pwqd_to_lagrange[i]], 1e-12);
      }
    }
  }
}

TEST(PWQDPolygonMapping, Discretization)
{
  for (const auto& tm : TestMeshes())
  {
    auto sdm = PieceWiseQuadraticDiscontinuous::New(tm.grid);
    EXPECT_EQ(sdm->GetType(), SpatialDiscretizationType::PIECEWISE_QUADRATIC_DISCONTINUOUS);
    size_t local_dofs = 0;
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      EXPECT_EQ(sdm->GetCellNumNodes(*cell), 2 * cell->vertex_ids.size()) << tm.name;
      local_dofs += sdm->GetCellNumNodes(*cell);
    }
    EXPECT_EQ(sdm->GetNumLocalNodes(), local_dofs) << tm.name;
    // Ghost cells have mappings too (needed by the sweep and the DSA at > 1 rank)
    for (const auto ghost_id : tm.grid->GetGhostGlobalIDs())
    {
      const auto& ghost = tm.grid->GetGlobalCell(ghost_id);
      EXPECT_EQ(sdm->GetCellMapping(ghost).GetNumNodes(), 2 * ghost.vertex_ids.size()) << tm.name;
    }
  }
  // Slab cells are rejected
  auto line = BuildLineMesh(1.0, 4, 0.0);
  EXPECT_THROW(PieceWiseQuadraticDiscontinuous::New(line), std::invalid_argument);
}
