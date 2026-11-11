// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/mesh/raytrace/segment_tracer.h"

namespace opensn
{

/**
 * Finds where segments that end at one point, the apex, cross a material hull.
 *
 * Seen from the apex, such a segment is one direction, and it can only cross the triangles that
 * cover this direction and are not farther from the apex than its start point. The directions
 * are binned on the six faces of a cube centered at the apex, num_bins x num_bins bins per face.
 * Each bin lists the triangles that cover part of it, the nearest to the apex first, with the
 * bounding rectangle of the outline of each on the face. A segment goes through the list of the
 * bin of its direction, up to the distance of its start point, and tests the triangles whose
 * rectangle and outline hold its direction: there is no search, and the number of triangle
 * tests does not grow with the size of the hull.
 *
 * The lists are conservative. The rectangle and the outline of a triangle are enlarged by more
 * than their round-off, the triangle is listed in every bin that the rectangle touches, and
 * the triangles closer to the apex than a millionth of the size of the hull are tested by
 * every segment. The outline is the projection of the three vertices on the plane of the face;
 * it is used when the three are in front of the apex and project within eight face widths,
 * and the rectangle alone otherwise. The triangle test is that of SegmentTracer, so the crossings
 * are identical to those of the tracer the grid is built from. A segment that does not end at the
 * apex is handed to that tracer.
 *
 * The grid is only read after construction, so several threads may trace with it at the same
 * time.
 */
class DirectionGridTracer
{
public:
  /**
   * Builds the grid of the triangles of the hull of tracer seen from apex. The triangles listed
   * in excluded_triangles, which must be sorted, are ignored by Trace. num_bins is the number of
   * bins along each side of a face of the cube.
   */
  DirectionGridTracer(std::shared_ptr<const SegmentTracer> tracer,
                      const Vector3& apex,
                      std::vector<std::uint32_t> excluded_triangles,
                      unsigned int num_bins);

  /**
   * Finds every crossing of each segment with the hull, as SegmentTracer::Trace does with the
   * excluded triangles of this grid. The output is overwritten and the work done is added to
   * counters.
   */
  void Trace(const std::vector<HullSegment>& segments,
             HullCrossings& crossings,
             HullTraceCounters& counters) const;

  const Vector3& GetApex() const { return apex_; }

  /// Number of entries of the lists of the bins, the triangles tested by every segment included.
  std::size_t GetNumEntries() const { return entries_.size() + near_triangles_.size(); }

  /// Longest list of a bin.
  std::size_t GetMaxBinSize() const;

private:
  /// A triangle in the list of a bin.
  struct Entry
  {
    /// Lower bound of the distance from the apex to the triangle.
    float distance;
    /// Bounding rectangle of the part of the triangle seen through the face, in the
    /// coordinates of the face.
    float lower_u, upper_u, lower_v, upper_v;
    std::uint32_t triangle;
    /// Index of the outline of the triangle in outlines_, or no_outline.
    std::uint32_t outline;
  };

  /// The three vertices of a triangle projected on the plane of a face, counter-clockwise.
  struct Outline
  {
    float u[3];
    float v[3];
  };

  static constexpr std::uint32_t no_outline = std::numeric_limits<std::uint32_t>::max();

  /// True when the point (u, v) of a face is outside the outline, enlarged.
  static bool IsOutside(const Outline& outline, double u, double v);

  const std::shared_ptr<const SegmentTracer> tracer_;
  const Vector3 apex_;
  const std::vector<std::uint32_t> excluded_triangles_;
  const unsigned int num_bins_;
  /// A segment uses the grid when its end point is within this distance of the apex.
  double end_tolerance_ = 0.0;
  /// The triangles within this distance of the apex are tested by every segment.
  double near_distance_ = 0.0;
  std::vector<std::uint32_t> near_triangles_;
  /// The list of bin b is entries_[offsets_[b]] to entries_[offsets_[b + 1] - 1], sorted by
  /// distance.
  std::vector<std::size_t> offsets_;
  std::vector<Entry> entries_;
  std::vector<Outline> outlines_;
};

} // namespace opensn
