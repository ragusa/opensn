// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mesh/raytrace/bvh_segment_tracer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace opensn
{

namespace
{

/// Largest number of triangles in a leaf.
constexpr std::uint32_t max_leaf_size = 4;
/// Number of bins of the surface-area heuristic.
constexpr int num_bins = 16;

struct Box
{
  std::array<double, 3> lower = {std::numeric_limits<double>::max(),
                                 std::numeric_limits<double>::max(),
                                 std::numeric_limits<double>::max()};
  std::array<double, 3> upper = {std::numeric_limits<double>::lowest(),
                                 std::numeric_limits<double>::lowest(),
                                 std::numeric_limits<double>::lowest()};

  void Include(const Vector3& point)
  {
    const double xyz[3] = {point.x, point.y, point.z};
    for (int d = 0; d < 3; ++d)
    {
      lower[d] = std::min(lower[d], xyz[d]);
      upper[d] = std::max(upper[d], xyz[d]);
    }
  }

  void Include(const Box& box)
  {
    for (int d = 0; d < 3; ++d)
    {
      lower[d] = std::min(lower[d], box.lower[d]);
      upper[d] = std::max(upper[d], box.upper[d]);
    }
  }

  bool Empty() const { return lower[0] > upper[0]; }

  /// Half the surface area; zero for an empty box.
  double HalfArea() const
  {
    if (Empty())
      return 0.0;
    const double dx = upper[0] - lower[0];
    const double dy = upper[1] - lower[1];
    const double dz = upper[2] - lower[2];
    return dx * dy + dy * dz + dz * dx;
  }
};

} // namespace

BVHSegmentTracer::BVHSegmentTracer(std::shared_ptr<const MaterialHull> hull)
  : SegmentTracer(std::move(hull))
{
  Build();
}

void
BVHSegmentTracer::Build()
{
  const auto& triangles = hull_->GetTriangles();
  const auto num_triangles = static_cast<std::uint32_t>(triangles.size());
  nodes_.clear();
  triangle_ids_.resize(num_triangles);
  std::iota(triangle_ids_.begin(), triangle_ids_.end(), 0U);
  if (num_triangles == 0)
    return;

  // Boxes are enlarged by this margin, which exceeds the round-off of the box test below, so a
  // segment that the triangle test places on a triangle always passes through the triangle's box.
  double max_coordinate = 0.0;
  std::vector<Box> triangle_boxes(num_triangles);
  std::vector<std::array<double, 3>> centroids(num_triangles);
  for (std::uint32_t t = 0; t < num_triangles; ++t)
  {
    for (const auto& vertex : triangles[t].vertices)
    {
      triangle_boxes[t].Include(vertex);
      max_coordinate =
        std::max({max_coordinate, std::abs(vertex.x), std::abs(vertex.y), std::abs(vertex.z)});
    }
    for (int d = 0; d < 3; ++d)
      centroids[t][d] = 0.5 * (triangle_boxes[t].lower[d] + triangle_boxes[t].upper[d]);
  }
  const double margin = 1.0e-9 * hull_->GetSize() + 1.0e-12 * max_coordinate;

  // Ranges of triangle_ids_ still to be turned into a node: (node, first, count).
  struct Task
  {
    std::uint32_t node, first, count;
  };
  std::vector<Task> tasks;
  nodes_.reserve(2 * static_cast<std::size_t>(num_triangles));
  nodes_.emplace_back();
  tasks.push_back({0, 0, num_triangles});

  while (not tasks.empty())
  {
    const Task task = tasks.back();
    tasks.pop_back();

    Box bounds;
    Box centroid_bounds;
    for (std::uint32_t i = task.first; i < task.first + task.count; ++i)
    {
      const std::uint32_t t = triangle_ids_[i];
      bounds.Include(triangle_boxes[t]);
      centroid_bounds.Include(Vector3(centroids[t][0], centroids[t][1], centroids[t][2]));
    }
    for (int d = 0; d < 3; ++d)
    {
      nodes_[task.node].lower[d] = bounds.lower[d] - margin;
      nodes_[task.node].upper[d] = bounds.upper[d] + margin;
    }
    nodes_[task.node].first = task.first;
    nodes_[task.node].count = task.count;
    if (task.count <= max_leaf_size)
      continue;

    // Split along the axis on which the triangle centroids are most spread out.
    int axis = 0;
    for (int d = 1; d < 3; ++d)
      if (centroid_bounds.upper[d] - centroid_bounds.lower[d] >
          centroid_bounds.upper[axis] - centroid_bounds.lower[axis])
        axis = d;
    const double axis_lower = centroid_bounds.lower[axis];
    const double extent = centroid_bounds.upper[axis] - axis_lower;

    const auto range_begin = triangle_ids_.begin() + task.first;
    const auto range_end = range_begin + task.count;
    auto middle = range_begin + task.count / 2;
    bool split_found = false;
    if (extent > 0.0)
    {
      // Binned surface-area heuristic: choose the bin boundary that minimizes the expected number
      // of triangle tests, area(left) count(left) + area(right) count(right).
      const auto bin_of = [&centroids, axis, axis_lower, extent](const std::uint32_t t)
      {
        const auto bin = static_cast<int>(num_bins * ((centroids[t][axis] - axis_lower) / extent));
        return std::clamp(bin, 0, num_bins - 1);
      };
      std::array<Box, num_bins> bin_boxes;
      std::array<std::uint32_t, num_bins> bin_counts{};
      for (auto it = range_begin; it != range_end; ++it)
      {
        const int bin = bin_of(*it);
        bin_boxes[bin].Include(triangle_boxes[*it]);
        ++bin_counts[bin];
      }

      std::array<double, num_bins> right_cost{};
      Box right_box;
      std::uint32_t right_count = 0;
      for (int bin = num_bins - 1; bin > 0; --bin)
      {
        right_box.Include(bin_boxes[bin]);
        right_count += bin_counts[bin];
        right_cost[bin] = right_box.HalfArea() * right_count;
      }

      double best_cost = std::numeric_limits<double>::max();
      int best_bin = -1;
      Box left_box;
      std::uint32_t left_count = 0;
      for (int bin = 0; bin + 1 < num_bins; ++bin)
      {
        left_box.Include(bin_boxes[bin]);
        left_count += bin_counts[bin];
        if (left_count == 0 or left_count == task.count)
          continue;
        const double cost = left_box.HalfArea() * left_count + right_cost[bin + 1];
        if (cost < best_cost)
        {
          best_cost = cost;
          best_bin = bin;
        }
      }

      if (best_bin >= 0)
      {
        middle = std::partition(range_begin,
                                range_end,
                                [&bin_of, best_bin](const std::uint32_t t)
                                { return bin_of(t) <= best_bin; });
        split_found = middle != range_begin and middle != range_end;
      }
    }
    if (not split_found)
    {
      // Coincident centroids: split the range in two halves.
      middle = range_begin + task.count / 2;
      std::nth_element(range_begin,
                       middle,
                       range_end,
                       [&centroids, axis](const std::uint32_t a, const std::uint32_t b)
                       { return centroids[a][axis] < centroids[b][axis]; });
    }

    const auto left_count = static_cast<std::uint32_t>(middle - range_begin);
    const auto children = static_cast<std::uint32_t>(nodes_.size());
    nodes_.emplace_back();
    nodes_.emplace_back();
    nodes_[task.node].first = children;
    nodes_[task.node].count = 0;
    tasks.push_back({children, task.first, left_count});
    tasks.push_back({children + 1, task.first + left_count, task.count - left_count});
  }
}

void
BVHSegmentTracer::Trace(const std::vector<HullSegment>& segments,
                        const std::vector<std::uint32_t>& excluded_triangles,
                        HullCrossings& crossings,
                        HullTraceCounters& counters) const
{
  crossings.crossings.clear();
  crossings.offsets.assign(1, 0);
  std::uint64_t box_tests = 0;
  std::uint64_t triangle_tests = 0;

  thread_local std::vector<std::uint32_t> stack;
  for (const auto& segment : segments)
  {
    const HullSegmentFrame frame(segment);
    const std::size_t first = crossings.crossings.size();
    if (frame.valid and not nodes_.empty())
    {
      double inverse[3];
      for (int d = 0; d < 3; ++d)
        inverse[d] = frame.direction[d] != 0.0 ? 1.0 / frame.direction[d] : 0.0;

      stack.assign(1, 0);
      while (not stack.empty())
      {
        const Node& node = nodes_[stack.back()];
        stack.pop_back();
        ++box_tests;

        // Part of the segment, as a fraction of its length, that lies in the box.
        double enter = 0.0;
        double leave = 1.0;
        bool hit = true;
        for (int d = 0; d < 3 and hit; ++d)
        {
          if (frame.direction[d] == 0.0)
          {
            hit = frame.origin[d] >= node.lower[d] and frame.origin[d] <= node.upper[d];
            continue;
          }
          double t0 = (node.lower[d] - frame.origin[d]) * inverse[d];
          double t1 = (node.upper[d] - frame.origin[d]) * inverse[d];
          if (t0 > t1)
            std::swap(t0, t1);
          enter = std::max(enter, t0);
          leave = std::min(leave, t1);
          hit = enter <= leave;
        }
        if (not hit)
          continue;

        if (node.count == 0)
        {
          stack.push_back(node.first);
          stack.push_back(node.first + 1);
        }
        else
        {
          triangle_tests += node.count;
          for (std::uint32_t i = node.first; i < node.first + node.count; ++i)
            TestTriangle(frame, triangle_ids_[i], excluded_triangles, crossings.crossings);
        }
      }
    }
    SortCrossings(crossings.crossings, first);
    crossings.offsets.push_back(crossings.crossings.size());
  }
  counters.segments += segments.size();
  counters.box_tests += box_tests;
  counters.triangle_tests += triangle_tests;
  counters.crossings += crossings.crossings.size();
}

} // namespace opensn
