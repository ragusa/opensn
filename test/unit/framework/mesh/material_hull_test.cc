// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "test/unit/framework/mesh/hull_test_utils.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/raytrace/material_hull.h"
#include "framework/runtime.h"
#include <algorithm>
#include <filesystem>
#include <tuple>

using namespace opensn;

namespace
{

double
TriangleArea(const HullTriangle& triangle)
{
  const auto& v = triangle.vertices;
  return 0.5 * (v[1] - v[0]).Cross(v[2] - v[0]).Norm();
}

std::shared_ptr<Mesh>
BuildTetMesh()
{
  auto grid = BuildMeshFromFile(std::filesystem::path(OPENSN_TEST_ROOT) / "assets" / "mesh" /
                                "uncollided_cube_coarse.msh");
  grid->SetUniformBlockID(0);
  return grid;
}

} // namespace

// These tests run on every rank, and at 1, 2 and 4 ranks in CI. Each expectation is derived from
// the geometry and not from the partitioning, so passing at every rank count shows that the hull
// is the same for every partitioning and on every rank.

TEST(MaterialHullTest, HomogeneousMeshGivesOnlyBoundaryFaces)
{
  const unsigned int n = 3;
  const auto grid = BuildBoxMesh(2.0, n, -1.0);
  grid->SetUniformBlockID(7);
  const auto hull = MaterialHull::Extract(*grid);

  ASSERT_EQ(hull->GetBlockIDs(), std::vector<unsigned int>({7}));
  EXPECT_EQ(hull->GetNumMaterials(), 1U);
  EXPECT_EQ(hull->GetMaterialIndex(7), 0U);
  EXPECT_THROW(hull->GetMaterialIndex(3), std::invalid_argument);

  // Each of the 6 n^2 boundary quadrilaterals is split into four triangles.
  ASSERT_EQ(hull->GetTriangles().size(), 6U * n * n * 4U);
  double area = 0.0;
  for (const auto& triangle : hull->GetTriangles())
  {
    EXPECT_EQ(triangle.back_material, 0U);
    EXPECT_EQ(triangle.front_material, hull->GetExteriorIndex());
    area += TriangleArea(triangle);

    // The normal points out of the domain, which is centered at the origin.
    const auto& v = triangle.vertices;
    EXPECT_GT((v[1] - v[0]).Cross(v[2] - v[0]).Dot(v[0] + v[1] + v[2]), 0.0);
  }
  EXPECT_NEAR(area, 24.0, 1.0e-12);
  EXPECT_NEAR(hull->GetSize(), 2.0 * std::sqrt(3.0), 1.0e-14);
  EXPECT_TRUE(hull_test::IsClosedSurface(*hull, 0));
}

TEST(MaterialHullTest, InterfacesMatchIndependentConstruction)
{
  const auto grid = hull_test::BuildDoglegMesh();
  const auto hull = MaterialHull::Extract(*grid);
  const auto model = hull_test::MakeDoglegVoxelModel();

  ASSERT_EQ(hull->GetBlockIDs(), model.BlockIDs());

  // The coordinates are multiples of 1/4, so the vertices and face centroids are exact and the
  // triangles can be compared exactly with those written down from the voxel description.
  const auto expected = hull_test::MakeTriangleKeys(model.Triangles(true));
  const auto actual = hull_test::MakeTriangleKeys(hull->GetTriangles());
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_TRUE(actual == expected);

  // No triangle is emitted twice, and none separates a material from itself.
  EXPECT_TRUE(std::adjacent_find(actual.begin(), actual.end()) == actual.end());
  for (const auto& triangle : hull->GetTriangles())
    EXPECT_NE(triangle.back_material, triangle.front_material);

  // The duct has 10 voxels of side 0.5, hence 60 voxel faces, of which 26 are between two duct
  // voxels. Of the other 34, 4 are on the domain boundary (2 at each end of the duct), 2 face the
  // column and 28 face the rest. Each voxel face has an area of 0.25.
  EXPECT_NEAR(hull_test::InterfaceArea(*hull, 0, 1), 7.0, 1.0e-12);
  EXPECT_NEAR(hull_test::InterfaceArea(*hull, 1, 2), 0.5, 1.0e-12);
  EXPECT_NEAR(hull_test::InterfaceArea(*hull, 1, hull->GetExteriorIndex()), 1.0, 1.0e-12);
  for (std::uint32_t material = 0; material < hull->GetNumMaterials(); ++material)
    EXPECT_TRUE(hull_test::IsClosedSurface(*hull, material));
}

TEST(MaterialHullTest, TrianglesAreSortedByOwner)
{
  const auto grid = hull_test::BuildDoglegMesh();
  const auto hull = MaterialHull::Extract(*grid);
  const auto& triangles = hull->GetTriangles();
  EXPECT_TRUE(std::is_sorted(
    triangles.begin(),
    triangles.end(),
    [](const HullTriangle& a, const HullTriangle& b)
    { return std::tie(a.owner_cell, a.face, a.side) < std::tie(b.owner_cell, b.face, b.side); }));
  for (const auto& triangle : triangles)
    EXPECT_EQ(triangle.boundary_id == MaterialHull::NO_BOUNDARY,
              triangle.front_material != hull->GetExteriorIndex());
}

TEST(MaterialHullTest, AllFacesSelectionKeepsEveryFace)
{
  const unsigned int n = 4;
  const auto grid = hull_test::BuildDoglegMesh();
  const auto hull = MaterialHull::Extract(*grid, MaterialHull::FaceSelection::ALL_FACES);

  // 3 n^2 (n + 1) faces, four triangles each.
  EXPECT_EQ(hull->GetTriangles().size(), 3U * n * n * (n + 1U) * 4U);
  EXPECT_EQ(hull->GetBlockIDs(), std::vector<unsigned int>({0, 1, 2}));
}

TEST(MaterialHullTest, TriangularFacesAreSplitAroundTheirCentroid)
{
  const auto grid = BuildTetMesh();
  const auto hull = MaterialHull::Extract(*grid);

  // Count the boundary faces of the tetrahedra over all ranks.
  std::size_t local_boundary_faces = 0;
  for (const auto& cell : grid->GetLocalCells())
    for (const auto& face : cell->faces)
      if (not face.has_neighbor)
      {
        ASSERT_EQ(face.vertex_ids.size(), 3U);
        ++local_boundary_faces;
      }
  std::size_t boundary_faces = 0;
  mpi_comm.all_reduce(local_boundary_faces, boundary_faces, mpi::op::sum<std::size_t>());

  // Three side triangles per face, as for the piecewise-linear discretization.
  ASSERT_EQ(hull->GetTriangles().size(), 3U * boundary_faces);
  double area = 0.0;
  for (std::size_t t = 0; t < hull->GetTriangles().size(); ++t)
  {
    const auto& triangle = hull->GetTriangles()[t];
    EXPECT_EQ(triangle.side, t % 3U);
    area += TriangleArea(triangle);
    // The third vertex is the face centroid, shared by the three triangles of the face.
    EXPECT_EQ((triangle.vertices[2] - hull->GetTriangles()[t - t % 3U].vertices[2]).NormSquare(),
              0.0);
  }
  // The mesh is a cube of side 0.032.
  EXPECT_NEAR(area, 6.0 * 0.032 * 0.032, 1.0e-15);
  EXPECT_TRUE(hull_test::IsClosedSurface(*hull, 0));
}

TEST(MaterialHullTest, TetrahedralInterfaceIsClosed)
{
  const auto grid = BuildTetMesh();
  // An interface that follows the faces of the tetrahedra near the plane x = 0.016.
  auto params = RPPLogicalVolume::GetInputParameters();
  ParameterBlock block;
  block.AddParameter("xmin", -1.0);
  block.AddParameter("xmax", 0.016);
  block.AddParameter("infy", true);
  block.AddParameter("infz", true);
  params.AssignParameters(block);
  grid->SetBlockIDFromLogicalVolume(RPPLogicalVolume(params), 5, true);
  const auto hull = MaterialHull::Extract(*grid);

  ASSERT_EQ(hull->GetBlockIDs(), std::vector<unsigned int>({0, 5}));
  EXPECT_GT(hull_test::InterfaceArea(*hull, 0, 1), 0.032 * 0.032);
  EXPECT_TRUE(hull_test::IsClosedSurface(*hull, 0));
  EXPECT_TRUE(hull_test::IsClosedSurface(*hull, 1));
  EXPECT_NEAR(hull_test::InterfaceArea(*hull, 0, 2) + hull_test::InterfaceArea(*hull, 1, 2),
              6.0 * 0.032 * 0.032,
              1.0e-15);
}

TEST(MaterialHullTest, RejectsInvalidInput)
{
  const auto planar_grid = BuildSquareMesh(2.0, 2, -1.0);
  EXPECT_THROW(MaterialHull::Extract(*planar_grid), std::invalid_argument);

  HullTriangle triangle;
  triangle.vertices = {Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0)};
  triangle.back_material = 0;
  triangle.front_material = 3;
  EXPECT_THROW(MaterialHull({triangle}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(MaterialHull({}, {1, 0}), std::invalid_argument);
  EXPECT_THROW(MaterialHull({}, {1, 1}), std::invalid_argument);

  const MaterialHull empty({}, {});
  EXPECT_EQ(empty.GetNumMaterials(), 0U);
  EXPECT_EQ(empty.GetSize(), 0.0);
}

TEST(MaterialHullTest, EndPointExclusions)
{
  const auto model = hull_test::MakeDoglegVoxelModel();
  const auto hull = model.Hull(true);

  // A point inside a voxel touches no triangle.
  EXPECT_TRUE(hull->FindEndPointExclusions(model.Center(0, 0, 0)).empty());

  // A corner of the domain touches one fan triangle of each of the three boundary faces there.
  const auto corner = hull->FindEndPointExclusions(Vector3(-1.0, -1.0, -1.0));
  EXPECT_EQ(corner.size(), 6U);
  EXPECT_TRUE(std::is_sorted(corner.begin(), corner.end()));
  for (const auto t : corner)
    EXPECT_EQ(hull->GetTriangles()[t].front_material, hull->GetExteriorIndex());

  // The centre of a boundary face touches its four fan triangles.
  EXPECT_EQ(hull->FindEndPointExclusions(Vector3(-1.0, -0.75, -0.75)).size(), 4U);

  // A point on the face between the duct (voxel 0,1,1) and the rest (voxel 0,0,1).
  EXPECT_THROW(hull->FindEndPointExclusions(Vector3(-0.8, -0.5, -0.3)), std::invalid_argument);
  EXPECT_FALSE(hull->FindTrianglesAt(Vector3(-0.8, -0.5, -0.3)).empty());
}
