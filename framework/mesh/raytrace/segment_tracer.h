// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/mesh/raytrace/material_hull.h"
#include "framework/data_types/vector3.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace opensn
{

/// A straight segment traced from its start point to its end point.
struct HullSegment
{
  Vector3 start;
  Vector3 end;
};

/// A point where a segment crosses a hull triangle.
struct HullCrossing
{
  /// Position along the segment, 0 at the start point and 1 at the end point.
  double fraction = 0.0;
  /// Index of the triangle in MaterialHull::GetTriangles().
  std::uint32_t triangle = 0;
  /// Material the segment leaves, as a material index of the hull.
  std::uint32_t before = 0;
  /// Material the segment enters.
  std::uint32_t after = 0;
};

/// The crossings of a batch of segments.
struct HullCrossings
{
  /// The crossings of segment i are crossings[offsets[i]] to crossings[offsets[i + 1] - 1],
  /// sorted by fraction and then by triangle.
  std::vector<std::size_t> offsets;
  std::vector<HullCrossing> crossings;

  std::size_t GetNumSegments() const { return offsets.empty() ? 0 : offsets.size() - 1; }
  const HullCrossing* Begin(std::size_t segment) const
  {
    return crossings.data() + offsets[segment];
  }
  const HullCrossing* End(std::size_t segment) const
  {
    return crossings.data() + offsets[segment + 1];
  }
};

/// Prepared form of a segment for the segment-triangle test.
struct HullSegmentFrame
{
  explicit HullSegmentFrame(const HullSegment& segment);

  double origin[3];
  double direction[3];
  /// Axis along which the direction is largest (kz) and the two other axes.
  int kx = 0;
  int ky = 1;
  int kz = 2;
  /// Shear that maps the direction onto the kz axis.
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  /// False for a segment of zero length, which crosses nothing.
  bool valid = false;
};

/**
 * Tests whether a segment crosses a triangle.
 *
 * The test is the edge-function test of Woop, Benthin and Wald (Journal of Computer Graphics
 * Techniques 2(1), 2013) in double precision: the triangle is sheared into a frame in which the
 * segment runs along one axis, and the segment crosses the triangle when it lies on the same
 * side of the three edges. Two properties make each crossing of a closed surface count once:
 *
 * - the edge function of an edge is evaluated from its two end points in a fixed order, so the
 *   triangles that share an edge obtain the same value with opposite signs, whatever the
 *   floating-point contraction of the build;
 * - a zero edge function, which means the segment passes through the edge or a vertex, is
 *   resolved as if the segment were displaced by an infinitesimal (eps, eps^2) in the sheared
 *   frame, the same displacement for every triangle.
 *
 * A segment that passes through an edge or a vertex of a closed surface is therefore reported by
 * the triangles that the displaced segment would cross, and by no others. A triangle seen
 * edge-on is not crossed.
 *
 * On a crossing, fraction is its position along the segment and forward is true when the segment
 * runs from the back side of the triangle to its front side. Crossings with fraction outside
 * [0, 1] are not reported.
 */
bool IntersectSegmentTriangle(const HullSegmentFrame& frame,
                              const std::array<Vector3, 3>& triangle,
                              double& fraction,
                              bool& forward);

/**
 * Finds where segments cross a material hull.
 *
 * A tracer is built once from a hull and then only read, so Trace may be called from several
 * threads at the same time, each with its own output.
 */
class SegmentTracer
{
public:
  explicit SegmentTracer(std::shared_ptr<const MaterialHull> hull);
  virtual ~SegmentTracer() = default;

  const MaterialHull& GetHull() const { return *hull_; }

  /**
   * Finds every crossing of each segment with the hull. Triangles listed in excluded_triangles,
   * which must be sorted, are ignored. The output is overwritten.
   */
  virtual void Trace(const std::vector<HullSegment>& segments,
                     const std::vector<std::uint32_t>& excluded_triangles,
                     HullCrossings& crossings) const = 0;

  /// Marks a segment whose crossings are not consistent in TraceLengths.
  static constexpr std::uint32_t INVALID_MATERIAL = std::numeric_limits<std::uint32_t>::max();

  /**
   * Computes the length of each segment in each material.
   *
   * lengths is resized to segments.size() * (GetNumMaterials() + 1): entry
   * [s * (GetNumMaterials() + 1) + m] is the length of segment s in material m, the last index
   * being the exterior. end_material is the material at the end point of every segment.
   * start_materials receives the material at the start point of each segment, or
   * INVALID_MATERIAL when the crossings of the segment are not consistent; see
   * ComputeMaterialLengths.
   */
  void TraceLengths(const std::vector<HullSegment>& segments,
                    const std::vector<std::uint32_t>& excluded_triangles,
                    std::uint32_t end_material,
                    std::vector<double>& lengths,
                    std::vector<std::uint32_t>& start_materials,
                    HullCrossings& scratch) const;

  /**
   * Creates a tracer. backend is "bvh" (bounding volume hierarchy) or "brute_force" (every
   * triangle is tested, for verification).
   */
  static std::unique_ptr<SegmentTracer> Create(const std::string& backend,
                                               std::shared_ptr<const MaterialHull> hull);

protected:
  /// Tests one triangle and appends the crossing, if any, to crossings.
  void TestTriangle(const HullSegmentFrame& frame,
                    std::uint32_t triangle,
                    const std::vector<std::uint32_t>& excluded_triangles,
                    std::vector<HullCrossing>& crossings) const;

  /// Sorts the crossings from first on by fraction and then by triangle.
  static void SortCrossings(std::vector<HullCrossing>& crossings, std::size_t first);

  const std::shared_ptr<const MaterialHull> hull_;
};

/**
 * Computes the length of a segment in each material from its crossings.
 *
 * The material at the end point is known, so the lengths are anchored there: with L the length
 * of the segment and t_k = fraction_k L the distance of crossing k from the start point,
 *
 *   length_m = L [m = end_material] + sum_k t_k ([before_k = m] - [after_k = m]).
 *
 * This does not depend on the order of the crossings, and a crossing near the start point,
 * such as that of a facet on which the start point lies, carries a weight near zero.
 *
 * lengths must have num_materials entries (the exterior included) and is overwritten. Returns
 * the material at the start point, which is the only material m with
 * [m = end_material] + sum_k ([before_k = m] - [after_k = m]) = 1 while the sum is zero for
 * every other material. Returns SegmentTracer::INVALID_MATERIAL when the crossings do not
 * satisfy this, which means a crossing was missed or counted twice.
 */
std::uint32_t ComputeMaterialLengths(const HullCrossing* begin,
                                     const HullCrossing* end,
                                     double segment_length,
                                     std::uint32_t end_material,
                                     std::uint32_t num_materials,
                                     double* lengths);

} // namespace opensn
