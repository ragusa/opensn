// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

// Unit tests of the PWQD (piecewise quadratic discontinuous) polyhedron mapping and its
// discretization. Every check is a polynomial or geometric identity, or an equality with an
// independently constructed element (PWLD at p = 1, Lagrange P2 on tetrahedra).

#include "test/unit/common/mesh_builders.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/submesh_tying.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/piecewise_quadratic_polyhedron_mapping.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_linear/piecewise_linear_polyhedron_mapping.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_quadratic/piecewise_quadratic_discontinuous.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_discontinuous.h"
#include "framework/math/spatial_discretization/finite_element/finite_element_data.h"
#include "framework/math/spatial_discretization/finite_element/unit_cell_matrices.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/mesh/cell.h"
#include "framework/runtime.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <map>
#include <random>

using namespace opensn;
namespace fs = std::filesystem;

namespace
{

/// The 10 monomials of total degree <= 2 at x.
std::vector<double>
Quadratics(const Vector3& x)
{
  return {1.0, x.x, x.y, x.z, x.x * x.x, x.x * x.y, x.x * x.z, x.y * x.y, x.y * x.z, x.z * x.z};
}

/// The 4 monomials of degree <= 1 at x.
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

struct NamedMesh
{
  std::string name;
  std::shared_ptr<Mesh> grid;
  bool planar_faces;
};

std::vector<NamedMesh>
TestMeshes()
{
  return {{"orthogonal hexes", BuildBoxMesh(1.0, 2, 0.0), true},
          {"non-uniform hexes",
           BuildOrthogonalMesh({{0.0, 0.3, 1.0}, {0.0, 0.5, 1.0}, {0.0, 0.4, 1.0}}),
           true},
          {"sheared hexes", AssetMesh("pwqd_hex_sheared_2.vtu"), true},
          {"tets", AssetMesh("pwqd_tets_2.vtu"), true},
          {"perturbed hexes", AssetMesh("pwqd_hex_perturbed_2.vtu"), false}};
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

} // namespace

TEST(PWQDMapping, ReferenceElementP2)
{
  TetLagrangeReference ref(2);
  ASSERT_EQ(ref.GetNumNodes(), 10u);
  // Nodal (Kronecker) property
  for (size_t a = 0; a < 10; ++a)
  {
    std::vector<double> N;
    ref.Shape(ref.GetNodeReferenceLocation(a), N);
    for (size_t b = 0; b < 10; ++b)
      EXPECT_NEAR(N[b], a == b ? 1.0 : 0.0, 1e-14);
  }
  // Partition of unity, zero gradient sum, quadratic reproduction at random points
  std::mt19937 gen(7);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int trial = 0; trial < 20; ++trial)
  {
    Vector3 xi{u(gen), u(gen), u(gen)};
    xi /= (xi.x + xi.y + xi.z + u(gen) + 0.01); // inside the reference tet
    std::vector<double> N;
    std::vector<Vector3> dN;
    ref.Shape(xi, N);
    ref.Grad(xi, dN);
    double sum = 0.0;
    Vector3 gsum{0.0, 0.0, 0.0};
    Vector3 x_interp{0.0, 0.0, 0.0};
    double q_interp = 0.0;
    Vector3 dq_interp{0.0, 0.0, 0.0};
    for (size_t a = 0; a < 10; ++a)
    {
      sum += N[a];
      gsum += dN[a];
      const Vector3 xa = ref.GetNodeReferenceLocation(a);
      x_interp += xa * N[a];
      const double qa = xa.x * xa.y + xa.z * xa.z; // a quadratic
      q_interp += qa * N[a];
      dq_interp += dN[a] * qa;
    }
    EXPECT_NEAR(sum, 1.0, 1e-14);
    EXPECT_NEAR(gsum.Norm(), 0.0, 1e-14);
    EXPECT_NEAR((x_interp - xi).Norm(), 0.0, 1e-14);
    EXPECT_NEAR(q_interp, xi.x * xi.y + xi.z * xi.z, 1e-14);
    EXPECT_NEAR((dq_interp - Vector3{xi.y, xi.x, 2.0 * xi.z}).Norm(), 0.0, 1e-13);
  }
  // zeta = 0 face mask: 6 nodes
  size_t n_face = 0;
  for (const bool b : ref.GetZetaZeroFaceMask())
    n_face += b ? 1 : 0;
  EXPECT_EQ(n_face, 6u);
}

TEST(PWQDMapping, LeastSquaresWeights)
{
  // Square, full-rank Vandermonde: the weights interpolate exactly (Lagrange property)
  const std::vector<double> xs = {0.0, 0.5, 1.0};
  DenseMatrix<double> A(3, 3, 0.0);
  for (size_t i = 0; i < 3; ++i)
  {
    A(i, 0) = 1.0;
    A(i, 1) = xs[i];
    A(i, 2) = xs[i] * xs[i];
  }
  const double xt = 0.3;
  const auto w = LeastSquaresWeights(A, {1.0, xt, xt * xt});
  // Interpolating a quadratic f(x) = 2 - x + 3x^2 at xt
  double f = 0.0;
  for (size_t i = 0; i < 3; ++i)
    f += w[i] * (2.0 - xs[i] + 3.0 * xs[i] * xs[i]);
  EXPECT_NEAR(f, 2.0 - xt + 3.0 * xt * xt, 1e-13);

  // Rank-deficient (an identically zero column): the fit ignores it and stays exact
  DenseMatrix<double> B(3, 4, 0.0);
  for (size_t i = 0; i < 3; ++i)
  {
    B(i, 0) = 1.0;
    B(i, 1) = xs[i];
    B(i, 2) = 0.0;
    B(i, 3) = xs[i] * xs[i];
  }
  const auto wb = LeastSquaresWeights(B, {1.0, xt, 0.7, xt * xt});
  double fb = 0.0;
  for (size_t i = 0; i < 3; ++i)
    fb += wb[i] * (2.0 - xs[i] + 3.0 * xs[i] * xs[i]);
  EXPECT_NEAR(fb, 2.0 - xt + 3.0 * xt * xt, 1e-13);

  // Overdetermined, consistent data: the least-squares fit reproduces a polynomial in the space
  const std::vector<double> ys = {0.0, 0.2, 0.5, 0.9, 1.0};
  DenseMatrix<double> C(5, 2, 0.0);
  for (size_t i = 0; i < 5; ++i)
  {
    C(i, 0) = 1.0;
    C(i, 1) = ys[i];
  }
  const auto wc = LeastSquaresWeights(C, {1.0, xt});
  double fc = 0.0;
  for (size_t i = 0; i < 5; ++i)
    fc += wc[i] * (4.0 - 2.5 * ys[i]);
  EXPECT_NEAR(fc, 4.0 - 2.5 * xt, 1e-13);
}

TEST(PWQDMapping, DimensionCount)
{
  for (const auto& tm : TestMeshes())
  {
    TetrahedraQuadrature tet_quad(QuadratureOrder::FOURTH);
    TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      PieceWiseQuadraticPolyhedronMapping m(*cell, tm.grid, tet_quad, tri_quad);
      const size_t n_v = cell->vertex_ids.size();
      const size_t n_f = cell->faces.size();
      EXPECT_EQ(m.GetNumNodes(), 2 * n_v + n_f - 2) << tm.name;
      for (size_t f = 0; f < n_f; ++f)
        EXPECT_EQ(m.GetNumFaceNodes(f), 2 * cell->faces[f].vertex_ids.size()) << tm.name;
      // Vertex nodes first, at the vertex locations
      const auto& X = m.GetNodeLocations();
      for (size_t i = 0; i < n_v; ++i)
        EXPECT_NEAR((X[i] - tm.grid->GlobalVertex(cell->vertex_ids[i])).Norm(), 0.0, 1e-15);
    }
  }
}

TEST(PWQDMapping, TyingReproducesPolynomials)
{
  for (const auto& tm : TestMeshes())
  {
    TetrahedraQuadrature tet_quad(QuadratureOrder::FOURTH);
    TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      PieceWiseQuadraticPolyhedronMapping m(*cell, tm.grid, tet_quad, tri_quad);
      const auto& tying = m.GetTying();
      const auto& T = tying.GetTyingOperator();
      const auto& dof_x = tying.GetDofLocations();
      const auto& sub_x = tying.GetSubNodeLocations();
      const size_t N = tying.GetNumDofs();
      // Quadratics on planar-faced cells, linears everywhere
      const size_t n_mono = tm.planar_faces ? 10 : 4;
      const double tol = tm.planar_faces ? 1e-11 : 1e-12;
      for (size_t i = 0; i < tying.GetNumSubNodes(); ++i)
      {
        const auto exact = tm.planar_faces ? Quadratics(sub_x[i]) : Linears(sub_x[i]);
        std::vector<double> tied(n_mono, 0.0);
        for (size_t j = 0; j < N; ++j)
        {
          const auto qj = tm.planar_faces ? Quadratics(dof_x[j]) : Linears(dof_x[j]);
          for (size_t k = 0; k < n_mono; ++k)
            tied[k] += T(i, j) * qj[k];
        }
        for (size_t k = 0; k < n_mono; ++k)
          EXPECT_NEAR(tied[k], exact[k], tol) << tm.name << " sub-node " << i << " monomial " << k;
      }
    }
  }
}

TEST(PWQDMapping, FiniteElementDataIdentities)
{
  for (const auto& tm : TestMeshes())
  {
    TetrahedraQuadrature tet_quad(QuadratureOrder::FOURTH);
    TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      PieceWiseQuadraticPolyhedronMapping m(*cell, tm.grid, tet_quad, tri_quad);
      const size_t N = m.GetNumNodes();
      const auto vol = m.MakeVolumetricFiniteElementData();

      double volume = 0.0;
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
        }
        EXPECT_NEAR(sum, 1.0, 1e-13) << tm.name;
        EXPECT_NEAR(gsum.Norm(), 0.0, 1e-12) << tm.name;
        volume += vol.JxW(qp);
        for (size_t i = 0; i < N; ++i)
        {
          intV_grad[i] += vol.ShapeGrad(i, qp) * vol.JxW(qp);
          for (size_t j = 0; j < N; ++j)
            M(i, j) += vol.ShapeValue(i, qp) * vol.ShapeValue(j, qp) * vol.JxW(qp);
        }
      }
      EXPECT_NEAR(volume, cell->volume, 1e-13 * cell->volume) << tm.name;
      double msum = 0.0;
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
          msum += M(i, j);
      EXPECT_NEAR(msum, cell->volume, 1e-12 * cell->volume) << tm.name;
      EXPECT_TRUE(IsSPD(M, 1e-14)) << tm.name;

      // Surface: partition of unity, face areas, divergence theorem int grad b_j = oint b_j n.
      // OpenSn integrates a face with the face-average normal and the sub-triangle Jacobian
      // projected onto the plane of that normal (PieceWiseLinearPolyhedronMapping's convention,
      // which PWQD inherits; see ReducesToPWLDAtDegreeOne). On a non-planar face that measure
      // is not the true surface measure, so the area and divergence identities are checked on
      // planar-faced cells only.
      std::vector<Vector3> intS_b_n(N, Vector3{0.0, 0.0, 0.0});
      for (size_t f = 0; f < cell->faces.size(); ++f)
      {
        const auto srf = m.MakeSurfaceFiniteElementData(f);
        double area = 0.0;
        for (const auto qp : srf.GetQuadraturePointIndices())
        {
          double sum = 0.0;
          for (size_t i = 0; i < N; ++i)
            sum += srf.ShapeValue(i, qp);
          EXPECT_NEAR(sum, 1.0, 1e-13) << tm.name;
          area += srf.JxW(qp);
          for (size_t i = 0; i < N; ++i)
            intS_b_n[i] += srf.Normal(qp) * (srf.ShapeValue(i, qp) * srf.JxW(qp));
        }
        if (tm.planar_faces)
          EXPECT_NEAR(area, cell->faces[f].area, 1e-12 * cell->faces[f].area) << tm.name;
      }
      if (tm.planar_faces)
        for (size_t j = 0; j < N; ++j)
          EXPECT_NEAR((intV_grad[j] - intS_b_n[j]).Norm(), 0.0, 1e-12) << tm.name << " node " << j;
    }
  }
}

TEST(PWQDMapping, TraceLocality)
{
  for (const auto& tm : TestMeshes())
  {
    TetrahedraQuadrature tet_quad(QuadratureOrder::FOURTH);
    TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      PieceWiseQuadraticPolyhedronMapping m(*cell, tm.grid, tet_quad, tri_quad);
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

TEST(PWQDMapping, TraceConsistencyAcrossFaces)
{
  for (const auto& tm : TestMeshes())
  {
    auto sdm = PieceWiseQuadraticDiscontinuous::New(tm.grid);
    // Node locations of shared faces agree (throws otherwise)
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
        // Match quadrature points by location (the sub-triangle loops run opposite ways)
        for (const auto qp : srf.GetQuadraturePointIndices())
        {
          int aqp = -1;
          for (const auto q2 : asrf.GetQuadraturePointIndices())
            if ((srf.QPointXYZ(qp) - asrf.QPointXYZ(q2)).NormSquare() < 1e-24)
            {
              aqp = static_cast<int>(q2);
              break;
            }
          ASSERT_GE(aqp, 0) << tm.name << ": face quadrature points do not coincide";
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

TEST(PWQDMapping, ReducesToPWLDAtDegreeOne)
{
  auto average = std::make_shared<AverageTyingWeights>();
  for (const auto& tm : TestMeshes())
  {
    TetrahedraQuadrature tet_quad(QuadratureOrder::FOURTH);
    TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
    for (const auto& cell : tm.grid->GetLocalCells())
    {
      PieceWiseLinearPolyhedronMapping pwld(*cell, tm.grid, tet_quad, tri_quad);
      PieceWiseQuadraticPolyhedronMapping p1(*cell, tm.grid, tet_quad, tri_quad, 1, average);
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
    }
  }
}

TEST(PWQDMapping, EqualsLagrangeP2OnTetrahedra)
{
  auto grid = AssetMesh("pwqd_tets_2.vtu");
  TetrahedraQuadrature tet_quad(QuadratureOrder::FOURTH);
  TriangleQuadrature tri_quad(QuadratureOrder::FOURTH);
  TetLagrangeReference ref(2);
  for (const auto& cell : grid->GetLocalCells())
  {
    ASSERT_EQ(cell->vertex_ids.size(), 4u);
    PieceWiseQuadraticPolyhedronMapping m(*cell, grid, tet_quad, tri_quad);
    ASSERT_EQ(m.GetNumNodes(), 10u);

    // Affine map of the whole tetrahedron: x = X0 + A xi
    const Vector3 X0 = grid->GlobalVertex(cell->vertex_ids[0]);
    Matrix3x3 A;
    for (int k = 0; k < 3; ++k)
      A.SetColJVec(k, grid->GlobalVertex(cell->vertex_ids[k + 1]) - X0);
    const Matrix3x3 Ainv = A.Inverse();
    const Matrix3x3 ATinv = A.Transpose().Inverse();

    // Lagrange P2 node a sits at X0 + A * (a1, a2, a3)/2; match the PWQD nodes by location
    std::vector<int> pwqd_to_lagrange(10, -1);
    for (size_t i = 0; i < 10; ++i)
      for (size_t a = 0; a < 10; ++a)
      {
        const Vector3 xa = X0 + A * ref.GetNodeReferenceLocation(a);
        if ((xa - m.GetNodeLocations()[i]).NormSquare() < 1e-24)
          pwqd_to_lagrange[i] = static_cast<int>(a);
      }
    for (size_t i = 0; i < 10; ++i)
      ASSERT_GE(pwqd_to_lagrange[i], 0);

    const auto vol = m.MakeVolumetricFiniteElementData();
    std::vector<double> N;
    std::vector<Vector3> dN;
    for (const auto qp : vol.GetQuadraturePointIndices())
    {
      const Vector3 xi = Ainv * (vol.QPointXYZ(qp) - X0);
      ref.Shape(xi, N);
      ref.Grad(xi, dN);
      for (size_t i = 0; i < 10; ++i)
      {
        const int a = pwqd_to_lagrange[i];
        EXPECT_NEAR(vol.ShapeValue(i, qp), N[a], 1e-12);
        EXPECT_NEAR((vol.ShapeGrad(i, qp) - ATinv * dN[a]).Norm(), 0.0, 1e-11);
      }
    }
    for (size_t f = 0; f < 4; ++f)
    {
      const auto srf = m.MakeSurfaceFiniteElementData(f);
      for (const auto qp : srf.GetQuadraturePointIndices())
      {
        const Vector3 xi = Ainv * (srf.QPointXYZ(qp) - X0);
        ref.Shape(xi, N);
        for (size_t i = 0; i < 10; ++i)
          EXPECT_NEAR(srf.ShapeValue(i, qp), N[pwqd_to_lagrange[i]], 1e-12);
      }
    }
  }
}

TEST(PWQDMapping, Discretization)
{
  auto grid = BuildBoxMesh(1.0, 2, 0.0);
  auto sdm = PieceWiseQuadraticDiscontinuous::New(grid);
  EXPECT_EQ(sdm->GetType(), SpatialDiscretizationType::PIECEWISE_QUADRATIC_DISCONTINUOUS);
  const auto& uk = sdm->UNITARY_UNKNOWN_MANAGER;
  EXPECT_EQ(sdm->GetNumLocalDOFs(uk), 20 * grid->GetLocalCellCount());
  EXPECT_EQ(sdm->GetNumGlobalDOFs(uk), 20 * grid->GetGlobalNumberOfCells());
  // Unit cell matrices come out of the generic integrator
  for (const auto& cell : grid->GetLocalCells())
  {
    const auto ucm = ComputeUnitCellIntegrals(*sdm, *cell);
    EXPECT_EQ(ucm.intV_shapeI_shapeJ.Rows(), 20u);
    double v = 0.0;
    for (size_t i = 0; i < 20; ++i)
      v += ucm.intV_shapeI(i);
    EXPECT_NEAR(v, cell->volume, 1e-13);
  }
  // Insufficient quadrature order and non-polyhedral cells are rejected
  EXPECT_THROW(PieceWiseQuadraticDiscontinuous::New(grid, QuadratureOrder::SECOND),
               std::invalid_argument);
  auto square = BuildSquareMesh(1.0, 2, 0.0);
  EXPECT_THROW(PieceWiseQuadraticDiscontinuous::New(square), std::invalid_argument);
}
