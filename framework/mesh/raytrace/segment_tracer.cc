// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mesh/raytrace/segment_tracer.h"
#include "framework/mesh/raytrace/brute_force_segment_tracer.h"
#include "framework/mesh/raytrace/bvh_segment_tracer.h"
#include "framework/utils/error.h"
#include <algorithm>
#include <cmath>
#include <utility>

// The functions below must each compile to one body: the triangles that share a vertex or an
// edge rely on obtaining bitwise-identical values from them, which inlining into different
// contexts with different floating-point contraction would not guarantee.
#if defined(__GNUC__) || defined(__clang__)
#define OPENSN_HULL_NOINLINE __attribute__((noinline))
#else
#define OPENSN_HULL_NOINLINE
#endif

namespace opensn
{

namespace
{

/// A triangle vertex in the sheared frame of a segment: x and y are transverse to the segment and
/// z is the position along it, as a fraction of its length.
struct ShearedPoint
{
  double x, y, z;
};

OPENSN_HULL_NOINLINE ShearedPoint
ShearVertex(const HullSegmentFrame& frame, const Vector3& vertex)
{
  const double relative[3] = {
    vertex.x - frame.origin[0], vertex.y - frame.origin[1], vertex.z - frame.origin[2]};
  const double along = relative[frame.kz];
  return {
    relative[frame.kx] - frame.sx * along, relative[frame.ky] - frame.sy * along, frame.sz * along};
}

/// Twice the signed area of the triangle (segment, a, b) in the sheared frame.
OPENSN_HULL_NOINLINE double
EdgeFunction(const double ax, const double ay, const double bx, const double by)
{
  return ax * by - ay * bx;
}

/**
 * Evaluates the edge function of the edge from p to q and returns its sign, which is never zero
 * for an edge of nonzero sheared length.
 *
 * The end points are taken in lexicographic order so that both triangles that share the edge
 * evaluate the same expression. A zero value is resolved as for a segment displaced by
 * (eps, eps^2): the edge function of the displaced segment is
 * E + eps (p.y - q.y) + eps^2 (q.x - p.x).
 */
int
EdgeSide(const ShearedPoint& p, const ShearedPoint& q, double& value)
{
  const bool swapped = (q.x < p.x) or (q.x == p.x and q.y < p.y);
  const ShearedPoint& a = swapped ? q : p;
  const ShearedPoint& b = swapped ? p : q;

  value = EdgeFunction(a.x, a.y, b.x, b.y);
  int sign = 0;
  if (value > 0.0)
    sign = 1;
  else if (value < 0.0)
    sign = -1;
  else if (a.y != b.y)
    sign = a.y > b.y ? 1 : -1;
  else if (a.x != b.x)
    sign = b.x > a.x ? 1 : -1;

  if (swapped)
  {
    value = -value;
    sign = -sign;
  }
  return sign;
}

} // namespace

HullSegmentFrame::HullSegmentFrame(const HullSegment& segment)
  : origin{segment.start.x, segment.start.y, segment.start.z},
    direction{segment.end.x - segment.start.x,
              segment.end.y - segment.start.y,
              segment.end.z - segment.start.z}
{
  int largest = 0;
  for (int axis = 1; axis < 3; ++axis)
    if (std::abs(direction[axis]) > std::abs(direction[largest]))
      largest = axis;
  if (direction[largest] == 0.0 or not std::isfinite(direction[largest]))
    return;

  kz = largest;
  kx = (kz + 1) % 3;
  ky = (kx + 1) % 3;
  // Keep the sheared frame right-handed with the segment running towards +z.
  if (direction[kz] < 0.0)
    std::swap(kx, ky);
  sx = direction[kx] / direction[kz];
  sy = direction[ky] / direction[kz];
  sz = 1.0 / direction[kz];
  valid = true;
}

bool
IntersectSegmentTriangle(const HullSegmentFrame& frame,
                         const std::array<Vector3, 3>& triangle,
                         double& fraction,
                         bool& forward)
{
  if (not frame.valid)
    return false;

  const ShearedPoint a = ShearVertex(frame, triangle[0]);
  const ShearedPoint b = ShearVertex(frame, triangle[1]);
  const ShearedPoint c = ShearVertex(frame, triangle[2]);

  double e_ab = 0.0;
  double e_bc = 0.0;
  double e_ca = 0.0;
  const int side = EdgeSide(a, b, e_ab);
  if (side == 0 or EdgeSide(b, c, e_bc) != side or EdgeSide(c, a, e_ca) != side)
    return false;

  // Twice the signed area of the triangle in the sheared frame; zero for a triangle seen edge-on.
  const double area = e_ab + e_bc + e_ca;
  if (area == 0.0)
    return false;

  fraction = (e_bc * a.z + e_ca * b.z + e_ab * c.z) / area;
  if (std::isnan(fraction) or fraction < 0.0 or fraction > 1.0)
    return false;

  // Counter-clockwise in the sheared frame means the normal points along the segment.
  forward = side > 0;
  return true;
}

SegmentTracer::SegmentTracer(std::shared_ptr<const MaterialHull> hull) : hull_(std::move(hull))
{
  OpenSnInvalidArgumentIf(not hull_, "A segment tracer requires a material hull.");
}

void
SegmentTracer::TestTriangle(const HullSegmentFrame& frame,
                            const std::uint32_t triangle,
                            const std::vector<std::uint32_t>& excluded_triangles,
                            std::vector<HullCrossing>& crossings) const
{
  const auto& hull_triangle = hull_->GetTriangles()[triangle];
  double fraction = 0.0;
  bool forward = false;
  if (not IntersectSegmentTriangle(frame, hull_triangle.vertices, fraction, forward))
    return;
  if (not excluded_triangles.empty() and
      std::binary_search(excluded_triangles.begin(), excluded_triangles.end(), triangle))
    return;

  crossings.push_back({fraction,
                       triangle,
                       forward ? hull_triangle.back_material : hull_triangle.front_material,
                       forward ? hull_triangle.front_material : hull_triangle.back_material});
}

void
SegmentTracer::SortCrossings(std::vector<HullCrossing>& crossings, const std::size_t first)
{
  std::sort(
    crossings.begin() + static_cast<std::ptrdiff_t>(first),
    crossings.end(),
    [](const HullCrossing& a, const HullCrossing& b)
    { return a.fraction < b.fraction or (a.fraction == b.fraction and a.triangle < b.triangle); });
}

void
SegmentTracer::TraceLengths(const std::vector<HullSegment>& segments,
                            const std::vector<std::uint32_t>& excluded_triangles,
                            const std::uint32_t end_material,
                            std::vector<double>& lengths,
                            std::vector<std::uint32_t>& start_materials,
                            HullCrossings& scratch) const
{
  const std::uint32_t num_materials = hull_->GetNumMaterials() + 1;
  OpenSnInvalidArgumentIf(end_material >= num_materials,
                          "The end material is not a material of the hull.");

  Trace(segments, excluded_triangles, scratch);
  lengths.resize(segments.size() * num_materials);
  start_materials.resize(segments.size());
  for (std::size_t s = 0; s < segments.size(); ++s)
    start_materials[s] = ComputeMaterialLengths(scratch.Begin(s),
                                                scratch.End(s),
                                                (segments[s].end - segments[s].start).Norm(),
                                                end_material,
                                                num_materials,
                                                &lengths[s * num_materials]);
}

std::unique_ptr<SegmentTracer>
SegmentTracer::Create(const std::string& backend, std::shared_ptr<const MaterialHull> hull)
{
  if (backend == "bvh")
    return std::make_unique<BVHSegmentTracer>(std::move(hull));
  if (backend == "brute_force")
    return std::make_unique<BruteForceSegmentTracer>(std::move(hull));
  OpenSnInvalidArgument("Unknown segment tracer backend \"" + backend +
                        "\". Supported backends are \"bvh\" and \"brute_force\".");
}

std::uint32_t
ComputeMaterialLengths(const HullCrossing* begin,
                       const HullCrossing* end,
                       const double segment_length,
                       const std::uint32_t end_material,
                       const std::uint32_t num_materials,
                       double* lengths)
{
  // Net number of times the segment leaves each material, plus one for the end material.
  thread_local std::vector<int> net;
  net.assign(num_materials, 0);
  std::fill(lengths, lengths + num_materials, 0.0);

  lengths[end_material] = segment_length;
  net[end_material] = 1;
  for (const HullCrossing* crossing = begin; crossing != end; ++crossing)
  {
    const double distance = crossing->fraction * segment_length;
    lengths[crossing->before] += distance;
    lengths[crossing->after] -= distance;
    ++net[crossing->before];
    --net[crossing->after];
  }

  std::uint32_t start_material = SegmentTracer::INVALID_MATERIAL;
  for (std::uint32_t m = 0; m < num_materials; ++m)
  {
    if (net[m] == 0)
      continue;
    if (net[m] != 1 or start_material != SegmentTracer::INVALID_MATERIAL)
      return SegmentTracer::INVALID_MATERIAL;
    start_material = m;
  }
  return start_material;
}

} // namespace opensn
