// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "test/unit/framework/mesh/hull_test_utils.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/raytrace/bvh_segment_tracer.h"
#include "framework/mesh/raytrace/direction_grid_tracer.h"
#include "framework/mesh/raytrace/material_hull.h"
#include "framework/mesh/raytrace/raytracer.h"
#include "framework/mesh/raytrace/segment_tracer.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_discontinuous.h"
#include "framework/math/spatial_discretization/cell_mappings/cell_mapping.h"
#include "framework/runtime.h"
#include <filesystem>
#include <random>

using namespace opensn;

namespace
{

const char* const backends[] = {"brute_force", "bvh"};

/// Result of tracing one segment.
struct TracedSegment
{
  std::vector<HullCrossing> crossings;
  std::vector<double> lengths;
  std::uint32_t start_material = SegmentTracer::INVALID_MATERIAL;
};

TracedSegment
TraceOne(const SegmentTracer& tracer,
         const Vector3& start,
         const Vector3& end,
         const std::uint32_t end_material,
         const std::vector<std::uint32_t>& excluded = {})
{
  TracedSegment result;
  HullCrossings crossings;
  std::vector<std::uint32_t> start_materials;
  tracer.TraceLengths(
    {{start, end}}, excluded, end_material, result.lengths, start_materials, crossings);
  result.crossings = crossings.crossings;
  result.start_material = start_materials.front();
  return result;
}

/// Material index of the hull at a point of a voxel model; the exterior index outside it.
std::uint32_t
MaterialIndex(const hull_test::VoxelModel& model, const Vector3& point)
{
  const int material = model.MaterialAt(point);
  return static_cast<std::uint32_t>(material < 0 ? model.NumMaterials() : material);
}

/**
 * Traces a segment through the hull of a voxel model and compares the length in each material
 * with the lengths obtained by clipping the segment against every voxel.
 */
void
ExpectMatchesVoxelModel(const SegmentTracer& tracer,
                        const hull_test::VoxelModel& model,
                        const Vector3& start,
                        const Vector3& end,
                        const double tolerance,
                        const std::vector<std::uint32_t>& excluded = {})
{
  const auto traced = TraceOne(tracer, start, end, MaterialIndex(model, end), excluded);
  const auto expected = model.Lengths(start, end);
  ASSERT_EQ(traced.lengths.size(), expected.size());
  EXPECT_EQ(traced.start_material, MaterialIndex(model, start))
    << "from " << start.PrintStr() << " to " << end.PrintStr();
  for (std::size_t m = 0; m < expected.size(); ++m)
    EXPECT_NEAR(traced.lengths[m], expected[m], tolerance)
      << "material " << m << " from " << start.PrintStr() << " to " << end.PrintStr();
}

void
ExpectSameCrossings(const std::vector<HullCrossing>& a, const std::vector<HullCrossing>& b)
{
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    EXPECT_EQ(a[i].fraction, b[i].fraction);
    EXPECT_EQ(a[i].triangle, b[i].triangle);
    EXPECT_EQ(a[i].before, b[i].before);
    EXPECT_EQ(a[i].after, b[i].after);
  }
}

/// Points drawn uniformly in a box, with a fixed seed.
class RandomPoints
{
public:
  RandomPoints(const Vector3& lower, const Vector3& upper, unsigned int seed)
    : lower_(lower), upper_(upper), generator_(seed)
  {
  }

  Vector3 operator()()
  {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double u = uniform(generator_);
    const double v = uniform(generator_);
    const double w = uniform(generator_);
    return {lower_.x + u * (upper_.x - lower_.x),
            lower_.y + v * (upper_.y - lower_.y),
            lower_.z + w * (upper_.z - lower_.z)};
  }

private:
  Vector3 lower_;
  Vector3 upper_;
  std::mt19937 generator_;
};

/// Local cell that contains a point, or nullptr.
const Cell*
FindLocalCell(const Mesh& grid, const Vector3& point)
{
  for (const auto& cell : grid.GetLocalCells())
    if (grid.CheckPointInsideCell(*cell, point))
      return cell.get();
  return nullptr;
}

/// Block id of the cell that contains each point, over all ranks.
std::vector<unsigned int>
FindBlockIDs(const Mesh& grid, const std::vector<Vector3>& points)
{
  std::vector<unsigned int> local(points.size(), 0);
  for (std::size_t p = 0; p < points.size(); ++p)
    if (const Cell* cell = FindLocalCell(grid, points[p]))
      local[p] = cell->block_id;
  std::vector<unsigned int> global(points.size(), 0);
  mpi_comm.all_reduce(local, global, mpi::op::max<unsigned int>());
  return global;
}

/**
 * Length of a segment in each block id, obtained by walking from cell to cell with the mesh ray
 * tracer. Serial only.
 */
std::map<unsigned int, double>
WalkCells(const std::shared_ptr<Mesh>& grid, const Vector3& start, const Vector3& end)
{
  RayTracer ray_tracer(grid);
  std::map<unsigned int, double> lengths;
  const Cell* cell = FindLocalCell(*grid, start);
  if (cell == nullptr)
    throw std::logic_error("The start point is in no cell.");

  const double length = (end - start).Norm();
  Vector3 direction = (end - start) / length;
  Vector3 position = start;
  double remaining = length;
  for (int step = 0; step < 100000; ++step)
  {
    const auto trace = ray_tracer.TraceRay(*cell, position, direction);
    if (trace.particle_lost)
      throw std::logic_error("The cell walk lost the ray.");
    const double distance = std::min(trace.distance_to_surface, remaining);
    lengths[cell->block_id] += distance;
    remaining -= distance;
    if (remaining <= 1.0e-13 * length)
      return lengths;
    position = trace.pos_f;
    cell = &grid->GetGlobalCell(trace.destination_face_neighbor);
  }
  throw std::logic_error("The cell walk did not end.");
}

std::shared_ptr<Mesh>
BuildTwoMaterialTetMesh()
{
  auto grid = BuildMeshFromFile(std::filesystem::path(OPENSN_TEST_ROOT) / "assets" / "mesh" /
                                "uncollided_cube_coarse.msh");
  grid->SetUniformBlockID(0);
  // An interface that follows the faces of the tetrahedra near the plane x = 0.016.
  auto params = RPPLogicalVolume::GetInputParameters();
  ParameterBlock block;
  block.AddParameter("xmin", -1.0);
  block.AddParameter("xmax", 0.016);
  block.AddParameter("infy", true);
  block.AddParameter("infz", true);
  params.AssignParameters(block);
  grid->SetBlockIDFromLogicalVolume(RPPLogicalVolume(params), 5, true);
  return grid;
}

} // namespace

TEST(SegmentTracerTest, TriangleCrossingDirectionAndFraction)
{
  // The normal of this triangle is +z.
  const std::array<Vector3, 3> triangle = {
    Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0)};
  double fraction = -1.0;
  bool forward = false;

  // Along the normal: from the back side to the front side.
  const HullSegmentFrame up({Vector3(0.25, 0.25, -1.0), Vector3(0.25, 0.25, 3.0)});
  ASSERT_TRUE(IntersectSegmentTriangle(up, triangle, fraction, forward));
  EXPECT_DOUBLE_EQ(fraction, 0.25);
  EXPECT_TRUE(forward);

  const HullSegmentFrame down({Vector3(0.25, 0.25, 3.0), Vector3(0.25, 0.25, -1.0)});
  ASSERT_TRUE(IntersectSegmentTriangle(down, triangle, fraction, forward));
  EXPECT_DOUBLE_EQ(fraction, 0.75);
  EXPECT_FALSE(forward);

  // Oblique, with the largest direction component along each axis in turn.
  const HullSegmentFrame oblique_x({Vector3(-0.25, 0.25, -0.25), Vector3(1.75, 0.25, 0.75)});
  ASSERT_TRUE(IntersectSegmentTriangle(oblique_x, triangle, fraction, forward));
  EXPECT_DOUBLE_EQ(fraction, 0.25);
  EXPECT_TRUE(forward);
  const HullSegmentFrame oblique_y({Vector3(0.25, 1.25, 0.5), Vector3(0.25, -2.75, -1.5)});
  ASSERT_TRUE(IntersectSegmentTriangle(oblique_y, triangle, fraction, forward));
  EXPECT_DOUBLE_EQ(fraction, 0.25);
  EXPECT_FALSE(forward);

  // Beside the triangle, short of it, past it, within its plane, and of zero length.
  const HullSegmentFrame beside({Vector3(2.0, 2.0, -1.0), Vector3(2.0, 2.0, 1.0)});
  EXPECT_FALSE(IntersectSegmentTriangle(beside, triangle, fraction, forward));
  const HullSegmentFrame before({Vector3(0.25, 0.25, -2.0), Vector3(0.25, 0.25, -1.0)});
  EXPECT_FALSE(IntersectSegmentTriangle(before, triangle, fraction, forward));
  const HullSegmentFrame past({Vector3(0.25, 0.25, 1.0), Vector3(0.25, 0.25, 2.0)});
  EXPECT_FALSE(IntersectSegmentTriangle(past, triangle, fraction, forward));
  const HullSegmentFrame in_plane({Vector3(-1.0, 0.25, 0.0), Vector3(2.0, 0.25, 0.0)});
  EXPECT_FALSE(IntersectSegmentTriangle(in_plane, triangle, fraction, forward));
  const HullSegmentFrame point({Vector3(0.25, 0.25, 0.0), Vector3(0.25, 0.25, 0.0)});
  EXPECT_FALSE(point.valid);
  EXPECT_FALSE(IntersectSegmentTriangle(point, triangle, fraction, forward));
}

TEST(SegmentTracerTest, SlabsMatchAnalyticLengths)
{
  // Three slabs of thickness 1, 1 and 1 along z, in a 1 x 1 x 3 box.
  hull_test::VoxelModel model(1, 1, 3, Vector3(0.0, 0.0, 0.0), Vector3(1.0, 1.0, 1.0));
  model.At(0, 0, 1) = 1;
  model.At(0, 0, 2) = 2;
  for (const auto* backend : backends)
  {
    const auto tracer = SegmentTracer::Create(backend, model.Hull(true));

    // Normal to the slabs: 0.75 in the first, 1 in the second, 0.5 in the third.
    auto traced = TraceOne(*tracer, Vector3(0.3, 0.4, 0.25), Vector3(0.3, 0.4, 2.5), 2);
    EXPECT_EQ(traced.start_material, 0U);
    ASSERT_EQ(traced.crossings.size(), 2U);
    EXPECT_NEAR(traced.lengths[0], 0.75, 1.0e-15);
    EXPECT_NEAR(traced.lengths[1], 1.0, 1.0e-15);
    EXPECT_NEAR(traced.lengths[2], 0.5, 1.0e-15);
    EXPECT_NEAR(traced.lengths[3], 0.0, 1.0e-15);

    // Oblique: the lengths scale with 1 / cos(theta) = L / dz.
    const Vector3 start(0.1, 0.2, 0.5);
    const Vector3 end(0.9, 0.7, 2.25);
    const double secant = (end - start).Norm() / 1.75;
    traced = TraceOne(*tracer, start, end, 2);
    EXPECT_EQ(traced.start_material, 0U);
    EXPECT_NEAR(traced.lengths[0], 0.5 * secant, 1.0e-14);
    EXPECT_NEAR(traced.lengths[1], 1.0 * secant, 1.0e-14);
    EXPECT_NEAR(traced.lengths[2], 0.25 * secant, 1.0e-14);

    // From outside the box: the part before the box is in the exterior.
    traced = TraceOne(*tracer, Vector3(0.5, 0.5, -2.0), Vector3(0.5, 0.5, 1.5), 1);
    EXPECT_EQ(traced.start_material, 3U);
    EXPECT_NEAR(traced.lengths[3], 2.0, 1.0e-15);
    EXPECT_NEAR(traced.lengths[0], 1.0, 1.0e-15);
    EXPECT_NEAR(traced.lengths[1], 0.5, 1.0e-15);
  }
}

TEST(SegmentTracerTest, EdgesAndVerticesOfAClosedSurfaceAreCrossedOnce)
{
  // A unit cube. The segments join points of a lattice that is offset by one half, so many of
  // them pass exactly through a vertex, an edge, a face diagonal or a face centre of the cube,
  // and none lies within a face.
  const hull_test::VoxelModel model(1, 1, 1, Vector3(0.0, 0.0, 0.0), Vector3(1.0, 1.0, 1.0));
  std::vector<Vector3> points;
  for (int k = -2; k <= 2; ++k)
    for (int j = -2; j <= 2; ++j)
      for (int i = -2; i <= 2; ++i)
        points.emplace_back(0.5 + i, 0.5 + j, 0.5 + k);

  for (const bool fan : {false, true})
  {
    const auto brute_force = SegmentTracer::Create("brute_force", model.Hull(fan));
    const auto bvh = SegmentTracer::Create("bvh", model.Hull(fan));
    std::size_t through_cube = 0;
    for (std::size_t a = 0; a < points.size(); ++a)
      for (std::size_t b = 0; b < points.size(); ++b)
      {
        if (a == b)
          continue;
        const auto& start = points[a];
        const auto& end = points[b];
        const auto traced = TraceOne(*brute_force, start, end, MaterialIndex(model, end));
        const auto expected = model.Lengths(start, end);

        ASSERT_EQ(traced.start_material, MaterialIndex(model, start))
          << "from " << start.PrintStr() << " to " << end.PrintStr();
        EXPECT_NEAR(traced.lengths[0], expected[0], 1.0e-13);
        EXPECT_NEAR(traced.lengths[1], expected[1], 1.0e-13);

        // The surface is entered and left once per pass through the cube. A segment that only
        // touches it may be reported as entering and leaving at the same point.
        const bool starts_inside = model.MaterialAt(start) == 0;
        const bool ends_inside = model.MaterialAt(end) == 0;
        if (starts_inside != ends_inside)
          EXPECT_EQ(traced.crossings.size(), 1U);
        else if (expected[0] > 1.0e-12 and not starts_inside)
        {
          EXPECT_EQ(traced.crossings.size(), 2U);
          ++through_cube;
        }
        else
          EXPECT_TRUE(traced.crossings.empty() or traced.crossings.size() == 2U);

        ExpectSameCrossings(traced.crossings,
                            TraceOne(*bvh, start, end, MaterialIndex(model, end)).crossings);
      }
    EXPECT_GT(through_cube, 100U);
  }
}

TEST(SegmentTracerTest, VoxelCenterSegmentsMatchAnalyticLengths)
{
  // Segments between the centres of the voxels of a non-convex, three-material problem. Many
  // pass exactly through hull vertices and edges, including the concave edge of the duct and
  // the edges where three materials meet.
  const auto model = hull_test::MakeDoglegVoxelModel();
  std::vector<Vector3> centers;
  for (int k = 0; k < 4; ++k)
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i)
        centers.push_back(model.Center(i, j, k));

  for (const bool fan : {false, true})
    for (const auto* backend : backends)
    {
      const auto tracer = SegmentTracer::Create(backend, model.Hull(fan));
      for (std::size_t a = 0; a < centers.size(); ++a)
        for (std::size_t b = 0; b < centers.size(); ++b)
          if (a != b)
            ExpectMatchesVoxelModel(*tracer, model, centers[a], centers[b], 1.0e-13);
    }
}

TEST(SegmentTracerTest, RandomSegmentsMatchAnalyticLengths)
{
  const auto model = hull_test::MakeDoglegVoxelModel();
  for (const auto* backend : backends)
  {
    const auto tracer = SegmentTracer::Create(backend, model.Hull(true));

    // Both end points in the domain.
    RandomPoints inside(Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0), 2026);
    for (int s = 0; s < 2000; ++s)
    {
      const Vector3 start = inside();
      const Vector3 end = inside();
      ExpectMatchesVoxelModel(*tracer, model, start, end, 1.0e-13);
    }

    // End points in a larger box, so that some segments start or end in the exterior.
    RandomPoints around(Vector3(-2.0, -2.0, -2.0), Vector3(2.0, 2.0, 2.0), 106524);
    for (int s = 0; s < 2000; ++s)
    {
      const Vector3 start = around();
      const Vector3 end = around();
      ExpectMatchesVoxelModel(*tracer, model, start, end, 1.0e-13);
    }
  }
}

TEST(SegmentTracerTest, NonConvexRegionIsCrossedSeveralTimes)
{
  const auto model = hull_test::MakeDoglegVoxelModel();
  for (const auto* backend : backends)
  {
    const auto tracer = SegmentTracer::Create(backend, model.Hull(true));

    // Starts in the rest, crosses the first leg of the duct, leaves it through the concave
    // side, enters the second leg and ends there.
    const Vector3 start(-0.6, -0.9, -0.1);
    const Vector3 end(0.4, 0.9, -0.1);
    const auto traced = TraceOne(*tracer, start, end, 1);
    EXPECT_EQ(traced.start_material, 0U);
    ASSERT_EQ(traced.crossings.size(), 3U);
    const std::uint32_t expected_materials[4] = {0, 1, 0, 1};
    for (int c = 0; c < 3; ++c)
    {
      EXPECT_EQ(traced.crossings[c].before, expected_materials[c]);
      EXPECT_EQ(traced.crossings[c].after, expected_materials[c + 1]);
    }
    // The crossings are at y = -0.5, y = 0 and x = 0.
    EXPECT_NEAR(traced.crossings[0].fraction, 0.4 / 1.8, 1.0e-15);
    EXPECT_NEAR(traced.crossings[1].fraction, 0.9 / 1.8, 1.0e-15);
    EXPECT_NEAR(traced.crossings[2].fraction, 0.6, 1.0e-15);
    ExpectMatchesVoxelModel(*tracer, model, start, end, 1.0e-14);
  }
}

TEST(SegmentTracerTest, SegmentsThroughConcaveAndThreeMaterialEdges)
{
  const auto model = hull_test::MakeDoglegVoxelModel();
  for (const bool fan : {false, true})
    for (const auto* backend : backends)
    {
      const auto tracer = SegmentTracer::Create(backend, model.Hull(fan));

      // The concave edge of the duct is the line x = 0, y = 0. A segment through it from one
      // leg to the other stays in the duct.
      Vector3 start(-0.25, -0.25, -0.2);
      Vector3 end(0.25, 0.25, -0.2);
      auto traced = TraceOne(*tracer, start, end, 1);
      EXPECT_EQ(traced.start_material, 1U);
      EXPECT_NEAR(traced.lengths[1], (end - start).Norm(), 1.0e-15);
      EXPECT_NEAR(traced.lengths[0], 0.0, 1.0e-15);

      // Through the same edge from the rest into the duct: half the length in each.
      start = Vector3(-0.25, 0.25, -0.2);
      end = Vector3(0.25, -0.25, -0.2);
      traced = TraceOne(*tracer, start, end, 1);
      EXPECT_EQ(traced.start_material, 0U);
      EXPECT_NEAR(traced.lengths[0], 0.5 * (end - start).Norm(), 1.0e-15);
      EXPECT_NEAR(traced.lengths[1], 0.5 * (end - start).Norm(), 1.0e-15);

      // The rest, the duct and the column meet along the line x = 0.5, y = 0.5.
      start = Vector3(0.75, 0.25, 0.2);
      end = Vector3(0.25, 0.75, 0.2);
      traced = TraceOne(*tracer, start, end, 1);
      EXPECT_EQ(traced.start_material, 0U);
      EXPECT_NEAR(traced.lengths[0], 0.5 * (end - start).Norm(), 1.0e-15);
      EXPECT_NEAR(traced.lengths[1], 0.5 * (end - start).Norm(), 1.0e-15);
      EXPECT_NEAR(traced.lengths[2], 0.0, 1.0e-15);

      start = Vector3(0.25, 0.25, 0.2);
      end = Vector3(0.75, 0.75, 0.2);
      traced = TraceOne(*tracer, start, end, 2);
      EXPECT_EQ(traced.start_material, 1U);
      EXPECT_NEAR(traced.lengths[1], 0.5 * (end - start).Norm(), 1.0e-15);
      EXPECT_NEAR(traced.lengths[2], 0.5 * (end - start).Norm(), 1.0e-15);
      EXPECT_NEAR(traced.lengths[0], 0.0, 1.0e-15);
    }
}

TEST(SegmentTracerTest, StartPointOnAFacet)
{
  const auto model = hull_test::MakeDoglegVoxelModel();
  for (const auto* backend : backends)
  {
    const auto tracer = SegmentTracer::Create(backend, model.Hull(true));

    // A point on the face between the rest (y < -0.5) and the duct (y > -0.5).
    const Vector3 on_facet(-0.8, -0.5, -0.3);
    const Vector3 normal(0.0, 1.0, 0.0);
    const Vector3 in_duct(-0.3, -0.2, -0.2);
    const Vector3 in_rest(-0.7, -0.9, -0.4);

    // Whether or not the facet itself is reported, it weighs nothing: the segment is entirely
    // in the material it runs into.
    for (const auto& [end, material] : {std::pair{in_duct, 1U}, std::pair{in_rest, 0U}})
    {
      const double length = (end - on_facet).Norm();
      const auto traced = TraceOne(*tracer, on_facet, end, material);
      EXPECT_TRUE(traced.start_material == 0U or traced.start_material == 1U);
      EXPECT_NEAR(traced.lengths[material], length, 1.0e-15);
      EXPECT_NEAR(traced.lengths[1 - material], 0.0, 1.0e-15);

      // The lengths are continuous when the start point moves across the facet.
      for (const double offset : {-1.0e-9, 1.0e-9})
      {
        const Vector3 start = on_facet + offset * normal;
        const auto moved = TraceOne(*tracer, start, end, material);
        EXPECT_EQ(moved.start_material, offset > 0.0 ? 1U : 0U);
        EXPECT_NEAR(moved.lengths[material], length, 1.0e-8);
        EXPECT_NEAR(moved.lengths[1 - material], 0.0, 1.0e-8);
        ExpectMatchesVoxelModel(*tracer, model, start, end, 1.0e-14);
      }
    }
  }
}

TEST(SegmentTracerTest, ReversedSegmentGivesSameLengths)
{
  const auto model = hull_test::MakeDoglegVoxelModel();
  const auto tracer = SegmentTracer::Create("bvh", model.Hull(true));
  RandomPoints random(Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0), 7);
  for (int s = 0; s < 500; ++s)
  {
    const Vector3 a = random();
    const Vector3 b = random();
    const auto forward = TraceOne(*tracer, a, b, MaterialIndex(model, b));
    const auto backward = TraceOne(*tracer, b, a, MaterialIndex(model, a));
    EXPECT_EQ(forward.start_material, MaterialIndex(model, a));
    EXPECT_EQ(backward.start_material, MaterialIndex(model, b));
    ASSERT_EQ(forward.crossings.size(), backward.crossings.size());
    for (std::size_t m = 0; m < forward.lengths.size(); ++m)
      EXPECT_NEAR(forward.lengths[m], backward.lengths[m], 1.0e-14);
  }
}

TEST(SegmentTracerTest, SegmentsEndingAtADomainCorner)
{
  // Thin voxels: the segments reach the corner at a shallow angle to the boundary plane z = 0.
  hull_test::VoxelModel model(4, 4, 4, Vector3(0.0, 0.0, 0.0), Vector3(1.0, 1.0, 1.0e-3));
  for (const auto& region : hull_test::DoglegRegions())
    for (int k = region.first[2]; k <= region.last[2]; ++k)
      for (int j = region.first[1]; j <= region.last[1]; ++j)
        for (int i = region.first[0]; i <= region.last[0]; ++i)
          model.At(i, j, k) = region.material;
  const Vector3 corner(0.0, 0.0, 0.0);

  for (const bool fan : {false, true})
    for (const auto* backend : backends)
    {
      const auto hull = model.Hull(fan);
      const auto tracer = SegmentTracer::Create(backend, hull);
      const auto excluded = hull->FindEndPointExclusions(corner);
      ASSERT_FALSE(excluded.empty());

      const auto expect_matches = [&](const Vector3& start)
      {
        // The end point is on the boundary, so its material is that of the corner voxel.
        const auto traced = TraceOne(*tracer, start, corner, 0, excluded);
        const auto expected = model.Lengths(start, corner);
        EXPECT_EQ(traced.start_material, MaterialIndex(model, start)) << start.PrintStr();
        for (std::size_t m = 0; m < expected.size(); ++m)
          EXPECT_NEAR(traced.lengths[m], expected[m], 1.0e-12) << start.PrintStr();
      };

      for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 4; ++j)
          for (int i = 0; i < 4; ++i)
            expect_matches(model.Center(i, j, k));
      RandomPoints random(Vector3(0.0, 0.0, 0.0), Vector3(4.0, 4.0, 4.0e-3), 11);
      for (int s = 0; s < 1000; ++s)
        expect_matches(random());
    }
}

TEST(SegmentTracerTest, ExcludedTrianglesAreIgnored)
{
  const hull_test::VoxelModel model(1, 1, 1, Vector3(0.0, 0.0, 0.0), Vector3(1.0, 1.0, 1.0));
  for (const auto* backend : backends)
  {
    const auto tracer = SegmentTracer::Create(backend, model.Hull(false));
    const Vector3 start(0.3, 0.4, -1.0);
    const Vector3 end(0.3, 0.4, 2.0);
    const auto all = TraceOne(*tracer, start, end, 1);
    ASSERT_EQ(all.crossings.size(), 2U);
    EXPECT_EQ(all.start_material, 1U);

    // Without the triangle through which the segment enters, the remaining crossing reads as a
    // segment that starts inside the cube. A caller that knows the start material detects this.
    const auto without = TraceOne(*tracer, start, end, 1, {all.crossings.front().triangle});
    ASSERT_EQ(without.crossings.size(), 1U);
    EXPECT_EQ(without.crossings.front().triangle, all.crossings.back().triangle);
    EXPECT_EQ(without.start_material, 0U);

    // Without the triangle through which it leaves, no start material is consistent.
    const auto without_exit = TraceOne(*tracer, start, end, 1, {all.crossings.back().triangle});
    EXPECT_EQ(without_exit.start_material, SegmentTracer::INVALID_MATERIAL);
  }
}

TEST(SegmentTracerTest, ComputeMaterialLengthsDetectsInconsistentCrossings)
{
  // 0 -> 1 at one quarter and 1 -> 2 at one half of a segment of length 4 that ends in 2.
  const std::vector<HullCrossing> crossings = {{0.25, 0, 0, 1}, {0.5, 1, 1, 2}};
  std::vector<double> lengths(4);
  EXPECT_EQ(
    ComputeMaterialLengths(crossings.data(), crossings.data() + 2, 4.0, 2, 4, lengths.data()), 0U);
  EXPECT_DOUBLE_EQ(lengths[0], 1.0);
  EXPECT_DOUBLE_EQ(lengths[1], 1.0);
  EXPECT_DOUBLE_EQ(lengths[2], 2.0);
  EXPECT_DOUBLE_EQ(lengths[3], 0.0);

  // The order of the crossings does not matter.
  const std::vector<HullCrossing> reordered = {crossings[1], crossings[0]};
  std::vector<double> reordered_lengths(4);
  EXPECT_EQ(ComputeMaterialLengths(
              reordered.data(), reordered.data() + 2, 4.0, 2, 4, reordered_lengths.data()),
            0U);
  EXPECT_EQ(lengths, reordered_lengths);

  // No crossing: the segment is in its end material.
  EXPECT_EQ(ComputeMaterialLengths(nullptr, nullptr, 4.0, 2, 4, lengths.data()), 2U);
  EXPECT_DOUBLE_EQ(lengths[2], 4.0);

  // A missed crossing, a crossing counted twice, and a wrong end material.
  EXPECT_EQ(
    ComputeMaterialLengths(crossings.data() + 1, crossings.data() + 2, 4.0, 1, 4, lengths.data()),
    SegmentTracer::INVALID_MATERIAL);
  const std::vector<HullCrossing> doubled = {crossings[0], crossings[0], crossings[1]};
  EXPECT_EQ(ComputeMaterialLengths(doubled.data(), doubled.data() + 3, 4.0, 2, 4, lengths.data()),
            SegmentTracer::INVALID_MATERIAL);
  EXPECT_EQ(
    ComputeMaterialLengths(crossings.data(), crossings.data() + 2, 4.0, 3, 4, lengths.data()),
    SegmentTracer::INVALID_MATERIAL);
}

TEST(SegmentTracerTest, CreateAndBatchInterface)
{
  const auto model = hull_test::MakeDoglegVoxelModel();
  EXPECT_THROW(SegmentTracer::Create("embree", model.Hull(true)), std::invalid_argument);
  EXPECT_THROW(SegmentTracer::Create("bvh", nullptr), std::invalid_argument);

  // An empty hull is crossed by nothing.
  const auto empty =
    std::make_shared<MaterialHull>(std::vector<HullTriangle>{}, std::vector<unsigned int>{3});
  for (const auto* backend : backends)
  {
    const auto traced =
      TraceOne(*SegmentTracer::Create(backend, empty), Vector3(0.0), Vector3(1.0), 0);
    EXPECT_TRUE(traced.crossings.empty());
    EXPECT_EQ(traced.start_material, 0U);
    EXPECT_DOUBLE_EQ(traced.lengths[0], 1.0);
  }

  // A batch gives the same crossings as its segments traced one by one, and an empty batch and
  // a segment of zero length are valid.
  const auto tracer = SegmentTracer::Create("bvh", model.Hull(true));
  RandomPoints random(Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0), 3);
  std::vector<HullSegment> segments;
  segments.reserve(51);
  for (int s = 0; s < 50; ++s)
    segments.push_back({random(), random()});
  segments.push_back({Vector3(0.1, 0.2, 0.3), Vector3(0.1, 0.2, 0.3)});

  HullCrossings batch;
  tracer->Trace(segments, {}, batch);
  ASSERT_EQ(batch.GetNumSegments(), segments.size());
  for (std::size_t s = 0; s < segments.size(); ++s)
  {
    HullCrossings single;
    tracer->Trace({segments[s]}, {}, single);
    ExpectSameCrossings(std::vector<HullCrossing>(batch.Begin(s), batch.End(s)), single.crossings);
  }
  EXPECT_EQ(batch.Begin(segments.size() - 1), batch.End(segments.size() - 1));

  tracer->Trace({}, {}, batch);
  EXPECT_EQ(batch.GetNumSegments(), 0U);
  EXPECT_TRUE(batch.crossings.empty());

  // The hierarchy has fewer boxes than twice the number of triangles.
  const BVHSegmentTracer bvh(model.Hull(true));
  EXPECT_GT(bvh.GetNumNodes(), 1U);
  EXPECT_LT(bvh.GetNumNodes(), 2U * model.Hull(true)->GetTriangles().size());
}

TEST(SegmentTracerTest, CountersReportTheWorkDone)
{
  const auto model = hull_test::MakeDoglegVoxelModel();
  const auto hull = model.Hull(true);
  const std::size_t num_triangles = hull->GetTriangles().size();
  RandomPoints random(Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0), 11);
  std::vector<HullSegment> segments;
  segments.reserve(200);
  for (int s = 0; s < 200; ++s)
    segments.push_back({random(), random()});

  HullCrossings crossings;
  HullTraceCounters brute_force;
  SegmentTracer::Create("brute_force", hull)->Trace(segments, {}, crossings, brute_force);
  EXPECT_EQ(brute_force.segments, segments.size());
  EXPECT_EQ(brute_force.box_tests, 0U);
  EXPECT_EQ(brute_force.triangle_tests, segments.size() * num_triangles);
  EXPECT_EQ(brute_force.crossings, crossings.crossings.size());
  EXPECT_GT(brute_force.crossings, 0U);

  // The hierarchy finds the same crossings with fewer triangle tests. The counters add up over
  // the calls.
  HullTraceCounters bvh;
  const auto tracer = SegmentTracer::Create("bvh", hull);
  tracer->Trace(segments, {}, crossings, bvh);
  EXPECT_EQ(bvh.segments, segments.size());
  EXPECT_EQ(bvh.crossings, brute_force.crossings);
  EXPECT_GE(bvh.box_tests, segments.size());
  EXPECT_GE(bvh.triangle_tests, bvh.crossings);
  EXPECT_LT(bvh.triangle_tests, brute_force.triangle_tests);
  const HullTraceCounters once = bvh;
  tracer->Trace(segments, {}, crossings, bvh);
  EXPECT_EQ(bvh.segments, 2 * once.segments);
  EXPECT_EQ(bvh.box_tests, 2 * once.box_tests);
  EXPECT_EQ(bvh.triangle_tests, 2 * once.triangle_tests);
  EXPECT_EQ(bvh.crossings, 2 * once.crossings);
}

namespace
{

/// End points of segments on the dogleg voxel model: inside a voxel, at a corner of the domain,
/// on a boundary face, on a boundary edge, and inside the domain at 2e-7 from its boundary,
/// which is closer than the distance below which a direction grid tests a triangle for every
/// segment.
std::vector<Vector3>
DoglegApexes()
{
  return {Vector3(0.13, -0.42, 0.37),
          Vector3(-1.0, -1.0, -1.0),
          Vector3(-1.0, 0.2, 0.3),
          Vector3(-1.0, -1.0, 0.3),
          Vector3(-1.0 + 2.0e-7, 0.2, 0.3)};
}

/**
 * Start points of segments that end at the apex, chosen to stress a search of the hull: random
 * points, points on the planes of the voxel faces, points whose segment runs within a voxel
 * face or through a vertex of the hull, and points at and around the apex.
 */
std::vector<Vector3>
StartPointsFor(const Vector3& apex, const unsigned int seed)
{
  std::vector<Vector3> starts;
  starts.reserve(3100);
  RandomPoints random(Vector3(-1.2, -1.2, -1.2), Vector3(1.2, 1.2, 1.2), seed);
  for (int n = 0; n < 1500; ++n)
    starts.push_back(random());
  for (int n = 0; n < 600; ++n)
  {
    // On a plane of voxel faces, and in the plane of voxel faces through the apex.
    Vector3 on_plane = random();
    Vector3 with_apex = random();
    const double plane = -1.0 + 0.5 * (n % 5);
    if (n % 3 == 0)
    {
      on_plane.x = plane;
      with_apex.x = apex.x;
    }
    else if (n % 3 == 1)
    {
      on_plane.y = plane;
      with_apex.y = apex.y;
    }
    else
    {
      on_plane.z = plane;
      with_apex.z = apex.z;
    }
    starts.push_back(on_plane);
    starts.push_back(with_apex);
  }
  // Through every vertex of the voxel grid, beyond it and up to it.
  for (int k = 0; k <= 4; ++k)
    for (int j = 0; j <= 4; ++j)
      for (int i = 0; i <= 4; ++i)
      {
        const Vector3 vertex(-1.0 + 0.5 * i, -1.0 + 0.5 * j, -1.0 + 0.5 * k);
        starts.push_back(apex + 1.7 * (vertex - apex));
        starts.push_back(vertex);
      }
  // At the apex and at distances around those at which a direction grid changes its treatment.
  starts.push_back(apex);
  for (const double distance : {1.0e-14, 1.0e-9, 1.0e-6, 3.0e-6, 3.5e-6, 4.0e-6, 1.0e-3})
    for (int n = 0; n < 20; ++n)
    {
      const Vector3 direction = random();
      starts.push_back(apex + (distance / direction.Norm()) * direction);
    }
  return starts;
}

} // namespace

TEST(SegmentTracerTest, CandidatesOfABoxGiveTheCrossingsOfTrace)
{
  const auto model = hull_test::MakeDoglegVoxelModel();
  const auto hull = model.Hull(true);
  const auto num_triangles = hull->GetTriangles().size();
  const auto reference = SegmentTracer::Create("brute_force", hull);

  // Boxes of start points: small and large, flat, around the apex and across the boundary.
  struct StartBox
  {
    Vector3 lower, upper;
  };
  std::vector<StartBox> boxes = {{Vector3(0.6, 0.55, 0.7), Vector3(0.8, 0.9, 0.75)},
                                 {Vector3(-0.9, 0.1, -0.3), Vector3(-0.2, 0.2, 0.6)},
                                 {Vector3(0.5, -1.0, -1.0), Vector3(1.0, -0.5, -0.5)},
                                 {Vector3(-0.25, -0.25, 0.5), Vector3(0.25, 0.25, 0.5)},
                                 {Vector3(0.9, 0.9, 0.9), Vector3(1.3, 1.1, 1.2)},
                                 {Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0)}};
  RandomPoints random_corner(Vector3(-1.0, -1.0, -1.0), Vector3(0.8, 0.8, 0.8), 5);
  for (int n = 0; n < 20; ++n)
  {
    const Vector3 corner = random_corner();
    boxes.push_back({corner, corner + Vector3(0.05 + 0.01 * n, 0.2, 0.02 * n)});
  }

  for (const auto* backend : backends)
  {
    const auto tracer = SegmentTracer::Create(backend, hull);
    std::size_t num_candidates = 0;
    std::size_t num_sets = 0;
    HullTraceCounters counters;
    for (const auto& apex : DoglegApexes())
    {
      const auto excluded = hull->FindTrianglesAt(apex);
      std::vector<StartBox> apex_boxes = boxes;
      apex_boxes.push_back({apex - Vector3(0.1, 0.2, 0.3), apex + Vector3(0.3, 0.1, 0.2)});
      for (const auto& box : apex_boxes)
      {
        std::vector<std::uint32_t> candidates;
        tracer->CollectCandidates(box.lower, box.upper, apex, excluded, candidates, counters);
        num_candidates += candidates.size();
        ++num_sets;
        for (const auto triangle : candidates)
          EXPECT_FALSE(std::binary_search(excluded.begin(), excluded.end(), triangle));

        // Segments from the corners of the box and from points in it.
        std::vector<HullSegment> segments;
        segments.reserve(108);
        for (int corner = 0; corner < 8; ++corner)
          segments.push_back({Vector3((corner & 1) != 0 ? box.upper.x : box.lower.x,
                                      (corner & 2) != 0 ? box.upper.y : box.lower.y,
                                      (corner & 4) != 0 ? box.upper.z : box.lower.z),
                              apex});
        RandomPoints random(box.lower, box.upper, 17);
        for (int n = 0; n < 100; ++n)
          segments.push_back({random(), apex});

        HullCrossings expected;
        HullCrossings found;
        reference->Trace(segments, excluded, expected);
        tracer->TraceCandidates(segments, candidates, found, counters);
        ASSERT_EQ(found.GetNumSegments(), segments.size());
        for (std::size_t n = 0; n < segments.size(); ++n)
          ExpectSameCrossings(std::vector<HullCrossing>(expected.Begin(n), expected.End(n)),
                              std::vector<HullCrossing>(found.Begin(n), found.End(n)));
      }
    }
    EXPECT_EQ(counters.candidate_sets, num_sets);
    EXPECT_EQ(counters.candidate_triangles, num_candidates);
    // The hierarchy leaves out most of the triangles; without it every triangle is a candidate.
    if (std::string(backend) == "bvh")
    {
      EXPECT_LT(num_candidates, num_sets * num_triangles / 2);
      EXPECT_GT(counters.candidate_box_tests, 0U);
    }
    else
      EXPECT_GT(num_candidates, num_sets * (num_triangles - 12));
  }
}

TEST(DirectionGridTracerTest, CrossingsAreThoseOfTheTracer)
{
  const auto model = hull_test::MakeDoglegVoxelModel();
  const auto hull = model.Hull(true);
  const std::shared_ptr<const SegmentTracer> reference = SegmentTracer::Create("brute_force", hull);
  const std::shared_ptr<const SegmentTracer> tracer = SegmentTracer::Create("bvh", hull);

  unsigned int seed = 100;
  for (const auto& apex : DoglegApexes())
  {
    const auto excluded = hull->FindTrianglesAt(apex);
    std::vector<HullSegment> segments;
    for (const auto& start : StartPointsFor(apex, ++seed))
    {
      segments.push_back({start, apex});
      // The end point as the uncollided problem computes it, which is the apex to round-off.
      segments.push_back({start, start + 1.0 * (apex - start)});
    }
    HullCrossings expected;
    HullTraceCounters reference_counters;
    reference->Trace(segments, excluded, expected, reference_counters);

    for (const unsigned int num_bins : {1U, 2U, 7U, 64U, 500U})
    {
      const DirectionGridTracer grid(tracer, apex, excluded, num_bins);
      HullCrossings found;
      HullTraceCounters counters;
      grid.Trace(segments, found, counters);
      ASSERT_EQ(found.GetNumSegments(), segments.size());
      for (std::size_t n = 0; n < segments.size(); ++n)
        ExpectSameCrossings(std::vector<HullCrossing>(expected.Begin(n), expected.End(n)),
                            std::vector<HullCrossing>(found.Begin(n), found.End(n)));
      EXPECT_EQ(counters.segments, segments.size());
      EXPECT_EQ(counters.crossings, reference_counters.crossings);
      // A triangle is tested when its rectangle holds the direction of the segment.
      EXPECT_GE(counters.triangle_tests, counters.crossings);
      EXPECT_GE(grid.GetNumEntries(), hull->GetTriangles().size() - excluded.size());
      // Finer bins list fewer triangles each.
      if (num_bins >= 64)
        EXPECT_LT(counters.triangle_tests, reference_counters.triangle_tests / 20);
    }
  }
}

TEST(DirectionGridTracerTest, OtherSegmentsAreTracedByTheTracer)
{
  const auto model = hull_test::MakeDoglegVoxelModel();
  const auto hull = model.Hull(true);
  const std::shared_ptr<const SegmentTracer> tracer = SegmentTracer::Create("bvh", hull);
  const Vector3 apex(-1.0, -1.0, -1.0);
  const auto excluded = hull->FindTrianglesAt(apex);
  const DirectionGridTracer grid(tracer, apex, excluded, 16);
  EXPECT_EQ(grid.GetApex().x, apex.x);
  EXPECT_GT(grid.GetMaxBinSize(), 0U);

  // Segments that end elsewhere, mixed with segments that end at the apex and with a segment of
  // zero length.
  RandomPoints random(Vector3(-1.1, -1.1, -1.1), Vector3(1.1, 1.1, 1.1), 23);
  std::vector<HullSegment> segments;
  for (int n = 0; n < 300; ++n)
  {
    segments.push_back({random(), random()});
    segments.push_back({random(), apex});
    segments.push_back({random(), apex + Vector3(1.0e-9, 0.0, 0.0)});
  }
  segments.push_back({Vector3(0.1, 0.2, 0.3), Vector3(0.1, 0.2, 0.3)});

  HullCrossings expected;
  HullCrossings found;
  HullTraceCounters counters;
  tracer->Trace(segments, excluded, expected);
  grid.Trace(segments, found, counters);
  ASSERT_EQ(found.GetNumSegments(), segments.size());
  for (std::size_t n = 0; n < segments.size(); ++n)
    ExpectSameCrossings(std::vector<HullCrossing>(expected.Begin(n), expected.End(n)),
                        std::vector<HullCrossing>(found.Begin(n), found.End(n)));
  EXPECT_EQ(counters.segments, segments.size());
  EXPECT_EQ(counters.crossings, expected.crossings.size());
  // The segments that end elsewhere searched the hierarchy, which the others do not: they
  // test fewer rectangles than there are entries in the grid.
  HullTraceCounters apex_counters;
  std::vector<HullSegment> apex_segments;
  for (std::size_t n = 1; n < segments.size(); n += 3)
    apex_segments.push_back(segments[n]);
  grid.Trace(apex_segments, found, apex_counters);
  EXPECT_LE(apex_counters.box_tests, apex_segments.size() * grid.GetMaxBinSize());
  EXPECT_GT(counters.box_tests, apex_counters.box_tests + 600);

  grid.Trace({}, found, counters);
  EXPECT_EQ(found.GetNumSegments(), 0U);

  EXPECT_THROW(DirectionGridTracer(nullptr, apex, excluded, 16), std::invalid_argument);
  EXPECT_THROW(DirectionGridTracer(tracer, apex, excluded, 0), std::invalid_argument);
  EXPECT_THROW(DirectionGridTracer(tracer, apex, excluded, 5000), std::invalid_argument);
}

TEST(SegmentTracerTest, MeshHullMatchesVoxelReference)
{
  const auto grid = hull_test::BuildDoglegMesh();
  const auto model = hull_test::MakeDoglegVoxelModel();
  const auto hull = MaterialHull::Extract(*grid);
  const auto all_faces = MaterialHull::Extract(*grid, MaterialHull::FaceSelection::ALL_FACES);

  for (const auto* backend : backends)
  {
    const auto tracer = SegmentTracer::Create(backend, hull);
    const auto all_faces_tracer = SegmentTracer::Create(backend, all_faces);
    RandomPoints random(Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0), 41);
    for (int s = 0; s < 500; ++s)
    {
      const Vector3 start = random();
      const Vector3 end = random();
      ExpectMatchesVoxelModel(*tracer, model, start, end, 1.0e-13);
      // Tracing against every cell face gives the same lengths: a face between two cells of
      // the same material changes nothing.
      ExpectMatchesVoxelModel(*all_faces_tracer, model, start, end, 1.0e-13);
    }
  }
}

TEST(SegmentTracerTest, MatchesCellWalkOnHexahedra)
{
  if (opensn::mpi_comm.size() != 1)
    GTEST_SKIP() << "The cell walk needs the whole mesh on one rank.";

  const auto grid = hull_test::BuildDoglegMesh();
  const auto hull = MaterialHull::Extract(*grid);
  const auto tracer = SegmentTracer::Create("bvh", hull);
  RandomPoints random(Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0), 97);
  for (int s = 0; s < 300; ++s)
  {
    const Vector3 start = random();
    const Vector3 end = random();
    const auto end_block = FindBlockIDs(*grid, {end}).front();
    const auto traced = TraceOne(*tracer, start, end, hull->GetMaterialIndex(end_block));
    const auto walked = WalkCells(grid, start, end);
    ASSERT_NE(traced.start_material, SegmentTracer::INVALID_MATERIAL);
    for (std::uint32_t m = 0; m < hull->GetNumMaterials(); ++m)
    {
      const auto it = walked.find(hull->GetBlockIDs()[m]);
      EXPECT_NEAR(traced.lengths[m], it == walked.end() ? 0.0 : it->second, 1.0e-12);
    }
    EXPECT_NEAR(traced.lengths[hull->GetExteriorIndex()], 0.0, 1.0e-13);
  }
}

TEST(SegmentTracerTest, TetrahedralHullMatchesAllFacesAndCellWalk)
{
  const auto grid = BuildTwoMaterialTetMesh();
  const auto hull = MaterialHull::Extract(*grid);
  const auto all_faces = MaterialHull::Extract(*grid, MaterialHull::FaceSelection::ALL_FACES);
  ASSERT_LT(hull->GetTriangles().size(), all_faces->GetTriangles().size());

  const auto tracer = SegmentTracer::Create("bvh", hull);
  const auto brute_force = SegmentTracer::Create("brute_force", hull);
  const auto all_faces_tracer = SegmentTracer::Create("bvh", all_faces);

  const double side = 0.032;
  RandomPoints random(Vector3(0.01 * side, 0.01 * side, 0.01 * side),
                      Vector3(0.99 * side, 0.99 * side, 0.99 * side),
                      5);
  std::vector<Vector3> starts;
  std::vector<Vector3> ends;
  for (int s = 0; s < 300; ++s)
  {
    starts.push_back(random());
    ends.push_back(random());
  }
  const auto start_blocks = FindBlockIDs(*grid, starts);
  const auto end_blocks = FindBlockIDs(*grid, ends);

  double length_in_first = 0.0;
  double length_in_second = 0.0;
  for (std::size_t s = 0; s < starts.size(); ++s)
  {
    const auto end_material = hull->GetMaterialIndex(end_blocks[s]);
    const auto traced = TraceOne(*tracer, starts[s], ends[s], end_material);
    EXPECT_EQ(traced.start_material, hull->GetMaterialIndex(start_blocks[s]));
    length_in_first += traced.lengths[0];
    length_in_second += traced.lengths[1];
    EXPECT_NEAR(traced.lengths[2], 0.0, 1.0e-15 * side);
    EXPECT_NEAR(
      traced.lengths[0] + traced.lengths[1], (ends[s] - starts[s]).Norm(), 1.0e-13 * side);

    ExpectSameCrossings(traced.crossings,
                        TraceOne(*brute_force, starts[s], ends[s], end_material).crossings);

    // Every cell face, about ten times as many triangles.
    const auto all = TraceOne(*all_faces_tracer, starts[s], ends[s], end_material);
    EXPECT_EQ(all.start_material, traced.start_material);
    EXPECT_GE(all.crossings.size(), traced.crossings.size());
    for (std::size_t m = 0; m < traced.lengths.size(); ++m)
      EXPECT_NEAR(all.lengths[m], traced.lengths[m], 1.0e-13 * side);

    if (opensn::mpi_comm.size() == 1)
    {
      const auto walked = WalkCells(grid, starts[s], ends[s]);
      for (std::uint32_t m = 0; m < hull->GetNumMaterials(); ++m)
      {
        const auto it = walked.find(hull->GetBlockIDs()[m]);
        EXPECT_NEAR(traced.lengths[m], it == walked.end() ? 0.0 : it->second, 1.0e-11 * side);
      }
    }
  }
  // Both materials are sampled.
  EXPECT_GT(length_in_first, side);
  EXPECT_GT(length_in_second, side);
}

TEST(SegmentTracerTest, QuadraturePointsOfEveryCellMatchVoxelReference)
{
  // The points an uncollided calculation traces from: the volume quadrature points of every
  // cell and the quadrature points of its faces that look away from the source, which lie on a
  // hull facet when the face is a material interface or a boundary. The mesh is regular and the
  // source is at simple coordinates, so some paths pass exactly through an edge of the mesh: in
  // the xy plane the path from (-0.75, 0.625) to the source has a slope of -3/4 and goes
  // through the mesh edge at (-0.25, 0.25).
  struct Box
  {
    std::array<double, 3> lower, upper;
    unsigned int block_id;
  };
  const std::vector<Box> boxes = {{{-2.0, 0.25, -2.0}, {2.0, 0.75, 2.0}, 1},
                                  {{-0.5, 0.25, -0.25}, {0.25, 0.5, 0.5}, 2}};
  const Vector3 source(0.31, -0.17, 0.12);

  std::vector<double> nodes;
  for (int i = 0; i <= 8; ++i)
    nodes.push_back(-1.0 + 0.25 * i);
  const auto grid = BuildOrthogonalMesh({nodes, nodes, nodes});
  grid->SetUniformBlockID(0);
  hull_test::VoxelModel model(8, 8, 8, Vector3(-1.0, -1.0, -1.0), Vector3(0.25, 0.25, 0.25));
  for (const auto& box : boxes)
  {
    auto params = RPPLogicalVolume::GetInputParameters();
    ParameterBlock block;
    const char* lower_names[3] = {"xmin", "ymin", "zmin"};
    const char* upper_names[3] = {"xmax", "ymax", "zmax"};
    for (int axis = 0; axis < 3; ++axis)
    {
      block.AddParameter(lower_names[axis], box.lower[axis]);
      block.AddParameter(upper_names[axis], box.upper[axis]);
    }
    params.AssignParameters(block);
    grid->SetBlockIDFromLogicalVolume(RPPLogicalVolume(params), box.block_id, true);

    for (int k = 0; k < 8; ++k)
      for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i)
        {
          const Vector3 center = model.Center(i, j, k);
          if (center.x > box.lower[0] and center.x < box.upper[0] and center.y > box.lower[1] and
              center.y < box.upper[1] and center.z > box.lower[2] and center.z < box.upper[2])
            model.At(i, j, k) = static_cast<int>(box.block_id);
        }
  }

  const auto hull = MaterialHull::Extract(*grid);
  const auto tracer = SegmentTracer::Create("bvh", hull);
  const std::uint32_t source_material = MaterialIndex(model, source);
  const auto discretization = PieceWiseLinearDiscontinuous::New(grid, QuadratureOrder::FOURTH);

  std::size_t num_points = 0;
  std::size_t num_face_points_on_hull = 0;
  double worst_error = 0.0;
  const auto check = [&](const Cell& cell, const Vector3& point, const int face)
  {
    ++num_points;
    const auto traced = TraceOne(*tracer, point, source, source_material);
    const auto expected = model.Lengths(point, source);
    for (std::size_t m = 0; m < expected.size(); ++m)
      worst_error = std::max(worst_error, std::abs(traced.lengths[m] - expected[m]));

    // A point in the cell starts in the material of the cell; a point on a face may also be
    // found on the far side of the face.
    const std::uint32_t cell_material = hull->GetMaterialIndex(cell.block_id);
    if (face < 0)
    {
      EXPECT_EQ(traced.start_material, cell_material) << point.PrintStr();
      return;
    }
    const auto& cell_face = cell.faces[face];
    const std::uint32_t far_material =
      cell_face.has_neighbor
        ? hull->GetMaterialIndex(grid->GetGlobalCell(cell_face.neighbor_id).block_id)
        : hull->GetExteriorIndex();
    EXPECT_TRUE(traced.start_material == cell_material or traced.start_material == far_material)
      << point.PrintStr();
    if (far_material != cell_material)
      ++num_face_points_on_hull;
  };

  for (const auto& cell : grid->GetLocalCells())
  {
    const auto& mapping = discretization->GetCellMapping(*cell);
    const auto volume_data = mapping.MakeVolumetricFiniteElementData();
    for (const auto qp : volume_data.GetQuadraturePointIndices())
      check(*cell, volume_data.QPointXYZ(qp), -1);
    for (std::size_t f = 0; f < cell->faces.size(); ++f)
    {
      if ((cell->faces[f].centroid - source).Dot(cell->faces[f].normal) <= 0.0)
        continue;
      const auto surface_data = mapping.MakeSurfaceFiniteElementData(f);
      for (const auto qp : surface_data.GetQuadraturePointIndices())
        check(*cell, surface_data.QPointXYZ(qp), static_cast<int>(f));
    }
  }

  // The lengths are sums of at most a few terms of order one.
  EXPECT_LT(worst_error, 1.0e-13);

  std::size_t total_points = 0;
  std::size_t total_face_points_on_hull = 0;
  mpi_comm.all_reduce(num_points, total_points, mpi::op::sum<std::size_t>());
  mpi_comm.all_reduce(
    num_face_points_on_hull, total_face_points_on_hull, mpi::op::sum<std::size_t>());
  // 512 cells with 264 volume points each. Along each axis a cell has one face that looks away
  // from the source, or two when the source lies between its two faces, which is the case of one
  // cell in eight: 512 (3 + 3 / 8) = 1728 faces of 24 points each.
  EXPECT_EQ(total_points, 512U * 264U + 1728U * 24U);
  EXPECT_GT(total_face_points_on_hull, 1000U);
}
