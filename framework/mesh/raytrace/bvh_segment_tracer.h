// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/mesh/raytrace/segment_tracer.h"

namespace opensn
{

/**
 * Tracer that finds the triangles near a segment with a bounding volume hierarchy.
 *
 * The hierarchy is a binary tree of axis-aligned boxes built once with the binned surface-area
 * heuristic. A segment visits only the boxes it passes through, so its cost grows with the
 * logarithm of the number of triangles plus the number of triangles it crosses. The boxes are
 * padded so that a triangle the segment-triangle test would report is never skipped: the
 * crossings are identical to those of BruteForceSegmentTracer.
 *
 * CollectCandidates searches the hierarchy for a group of segments at once; see SegmentTracer.
 */
class BVHSegmentTracer : public SegmentTracer
{
public:
  /// Largest number of triangles in a leaf of the hierarchy, by default.
  static constexpr unsigned int default_max_leaf_size = 4;

  /// Builds the hierarchy of the hull, with at most max_leaf_size triangles in a leaf.
  explicit BVHSegmentTracer(std::shared_ptr<const MaterialHull> hull,
                            unsigned int max_leaf_size = default_max_leaf_size);

  void Trace(const std::vector<HullSegment>& segments,
             const std::vector<std::uint32_t>& excluded_triangles,
             HullCrossings& crossings,
             HullTraceCounters& counters) const override;
  using SegmentTracer::Trace;

  /**
   * Searches the hierarchy once with the shaft of the box and the apex, the convex hull of the
   * two: a box of the hierarchy that the shaft misses holds no candidate.
   */
  void CollectCandidates(const Vector3& lower,
                         const Vector3& upper,
                         const Vector3& apex,
                         const std::vector<std::uint32_t>& excluded_triangles,
                         std::vector<std::uint32_t>& candidates,
                         HullTraceCounters& counters) const override;

  /// Number of boxes in the hierarchy.
  std::size_t GetNumNodes() const { return nodes_.size(); }

private:
  struct Node
  {
    double lower[3];
    double upper[3];
    /// For a leaf, the first entry in triangle_ids_; otherwise the first of the two children.
    std::uint32_t first;
    /// Number of triangles of a leaf; zero for an interior node.
    std::uint32_t count;
  };

  void Build();

  const std::uint32_t max_leaf_size_;
  /// Padding of the boxes of the hierarchy.
  double margin_ = 0.0;
  std::vector<Node> nodes_;
  /// Triangle indices, ordered so that each leaf holds a contiguous range.
  std::vector<std::uint32_t> triangle_ids_;
};

} // namespace opensn
