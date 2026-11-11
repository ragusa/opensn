// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mesh/raytrace/direction_grid_tracer.h"
#include "framework/utils/error.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace opensn
{

namespace
{

/// Enlargement of the outline of a triangle on a face of the cube, in the coordinates of the
/// face, which run from -1 to 1. It exceeds the largest angle, 1e-6, between a segment that
/// DirectionGridTracer accepts and the direction of its start point seen from the apex.
constexpr double outline_padding = 1.0e-5;

/// An outline is kept when its vertices lie within this distance of the center of the face, in
/// the coordinates of the face. Their rounding to single precision is then below 1e-6.
constexpr double outline_reach = 8.0;

/// A point in the frame of a face of the cube: w along the axis of the face, positive towards
/// it, and (u, v) across it. The face is the square |u| <= w, |v| <= w.
struct FacePoint
{
  double u = 0.0;
  double v = 0.0;
  double w = 0.0;
};

/// A convex polygon with at most the seven vertices that clipping a triangle four times gives.
struct FacePolygon
{
  std::array<FacePoint, 8> points{};
  int size = 0;
};

/// Keeps the part of the polygon where scale * w + sign_u * u + sign_v * v >= 0.
FacePolygon
Clip(const FacePolygon& polygon, const double sign_u, const double sign_v, const double scale)
{
  FacePolygon clipped;
  const auto Side = [sign_u, sign_v, scale](const FacePoint& point)
  { return scale * point.w + sign_u * point.u + sign_v * point.v; };
  for (int i = 0; i < polygon.size; ++i)
  {
    const FacePoint& p = polygon.points[i];
    const FacePoint& q = polygon.points[(i + 1) % polygon.size];
    const double side_p = Side(p);
    const double side_q = Side(q);
    if (side_p >= 0.0)
      clipped.points[clipped.size++] = p;
    if ((side_p > 0.0 and side_q < 0.0) or (side_p < 0.0 and side_q > 0.0))
    {
      const double t = side_p / (side_p - side_q);
      clipped.points[clipped.size++] = {
        p.u + t * (q.u - p.u), p.v + t * (q.v - p.v), p.w + t * (q.w - p.w)};
    }
  }
  return clipped;
}

} // namespace

DirectionGridTracer::DirectionGridTracer(std::shared_ptr<const SegmentTracer> tracer,
                                         const Vector3& apex,
                                         std::vector<std::uint32_t> excluded_triangles,
                                         const unsigned int num_bins)
  : tracer_(std::move(tracer)),
    apex_(apex),
    excluded_triangles_(std::move(excluded_triangles)),
    num_bins_(num_bins)
{
  OpenSnInvalidArgumentIf(not tracer_, "A direction grid requires a segment tracer.");
  OpenSnInvalidArgumentIf(num_bins_ == 0 or num_bins_ > 4096,
                          "The number of bins of a direction grid must be between 1 and 4096.");

  const MaterialHull& hull = tracer_->GetHull();
  const auto& triangles = hull.GetTriangles();
  end_tolerance_ = 1.0e-12 * hull.GetSize();
  near_distance_ = 1.0e-6 * hull.GetSize();

  const auto bins = static_cast<double>(num_bins_);
  const auto BinAlong = [bins](const double coordinate)
  {
    const double bin = std::floor(0.5 * (coordinate + 1.0) * bins);
    return static_cast<std::size_t>(std::clamp(bin, 0.0, bins - 1.0));
  };

  // The entries with their bins, to be sorted by bin.
  std::vector<std::pair<std::size_t, Entry>> entries;
  for (std::uint32_t t = 0; t < triangles.size(); ++t)
  {
    if (std::binary_search(excluded_triangles_.begin(), excluded_triangles_.end(), t))
      continue;
    const auto& triangle = triangles[t].vertices;
    const std::array<Vector3, 3> vertices = {
      triangle[0] - apex_, triangle[1] - apex_, triangle[2] - apex_};
    const double distance =
      (ClosestPointOnTriangle(apex_, triangle[0], triangle[1], triangle[2]) - apex_).Norm();
    if (not(distance > near_distance_))
    {
      near_triangles_.push_back(t);
      continue;
    }
    // A lower bound of the distance, in single precision.
    const auto listed_distance =
      static_cast<float>(std::max(0.0, distance - near_distance_) * (1.0 - 1.0e-6));

    for (int face = 0; face < 6; ++face)
    {
      const int axis = face / 2;
      const double sign = (face % 2 == 0) ? 1.0 : -1.0;
      const int axis_u = (axis + 1) % 3;
      const int axis_v = (axis + 2) % 3;
      FacePolygon polygon;
      polygon.size = 3;
      for (int k = 0; k < 3; ++k)
      {
        const double xyz[3] = {vertices[k].x, vertices[k].y, vertices[k].z};
        polygon.points[k] = {xyz[axis_u], xyz[axis_v], sign * xyz[axis]};
      }
      const FacePolygon whole_triangle = polygon;
      // The part of the triangle seen through the face, the face slightly enlarged.
      const double scale = 1.0 + outline_padding;
      polygon = Clip(polygon, -1.0, 0.0, scale);
      polygon = Clip(polygon, 1.0, 0.0, scale);
      polygon = Clip(polygon, 0.0, -1.0, scale);
      polygon = Clip(polygon, 0.0, 1.0, scale);
      if (polygon.size == 0)
        continue;

      // Bounding rectangle of the outline on the face. A point of the clipped triangle has
      // w > 0, because the triangle is not near the apex.
      double lower_u = 2.0;
      double upper_u = -2.0;
      double lower_v = 2.0;
      double upper_v = -2.0;
      bool at_apex = false;
      for (int i = 0; i < polygon.size; ++i)
      {
        const FacePoint& point = polygon.points[i];
        if (not(point.w > 0.0))
        {
          at_apex = true;
          break;
        }
        lower_u = std::min(lower_u, point.u / point.w);
        upper_u = std::max(upper_u, point.u / point.w);
        lower_v = std::min(lower_v, point.v / point.w);
        upper_v = std::max(upper_v, point.v / point.w);
      }
      if (at_apex)
      {
        lower_u = lower_v = -1.0;
        upper_u = upper_v = 1.0;
      }
      // The rectangle is enlarged twice: for the round-off of the outline, and for its own
      // rounding to single precision, which is below 1e-7 here.
      lower_u -= outline_padding;
      upper_u += outline_padding;
      lower_v -= outline_padding;
      upper_v += outline_padding;
      // The outline of the whole triangle on the plane of the face. The projection maps the
      // triangle onto the triangle of its projected vertices when the three are in front of
      // the apex. An outline thinner than its enlargement is of no use.
      std::uint32_t outline_index = no_outline;
      Outline outline{};
      bool has_outline = true;
      for (int k = 0; k < 3 and has_outline; ++k)
      {
        const FacePoint& point = whole_triangle.points[k];
        has_outline = point.w > 0.0 and std::abs(point.u) <= outline_reach * point.w and
                      std::abs(point.v) <= outline_reach * point.w;
        if (has_outline)
        {
          outline.u[k] = static_cast<float>(point.u / point.w);
          outline.v[k] = static_cast<float>(point.v / point.w);
        }
      }
      if (has_outline)
      {
        const double edge_u[2] = {static_cast<double>(outline.u[1]) - outline.u[0],
                                  static_cast<double>(outline.u[2]) - outline.u[0]};
        const double edge_v[2] = {static_cast<double>(outline.v[1]) - outline.v[0],
                                  static_cast<double>(outline.v[2]) - outline.v[0]};
        const double area = edge_u[0] * edge_v[1] - edge_v[0] * edge_u[1];
        const double extent =
          std::abs(edge_u[0]) + std::abs(edge_v[0]) + std::abs(edge_u[1]) + std::abs(edge_v[1]);
        if (std::abs(area) > 8.0 * outline_padding * extent)
        {
          if (area < 0.0)
          {
            std::swap(outline.u[1], outline.u[2]);
            std::swap(outline.v[1], outline.v[2]);
          }
          outline_index = static_cast<std::uint32_t>(outlines_.size());
          outlines_.push_back(outline);
        }
      }
      const Entry entry = {listed_distance,
                           static_cast<float>(lower_u - outline_padding),
                           static_cast<float>(upper_u + outline_padding),
                           static_cast<float>(lower_v - outline_padding),
                           static_cast<float>(upper_v + outline_padding),
                           t,
                           outline_index};
      const std::size_t first_u = BinAlong(lower_u);
      const std::size_t last_u = BinAlong(upper_u);
      const std::size_t first_v = BinAlong(lower_v);
      const std::size_t last_v = BinAlong(upper_v);
      for (std::size_t j = first_v; j <= last_v; ++j)
        for (std::size_t i = first_u; i <= last_u; ++i)
          entries.emplace_back((static_cast<std::size_t>(face) * num_bins_ + j) * num_bins_ + i,
                               entry);
    }
  }

  std::sort(entries.begin(),
            entries.end(),
            [](const auto& a, const auto& b)
            {
              if (a.first != b.first)
                return a.first < b.first;
              if (a.second.distance != b.second.distance)
                return a.second.distance < b.second.distance;
              return a.second.triangle < b.second.triangle;
            });
  const std::size_t total_bins = static_cast<std::size_t>(6) * num_bins_ * num_bins_;
  offsets_.assign(total_bins + 1, 0);
  entries_.reserve(entries.size());
  for (const auto& [bin, entry] : entries)
  {
    ++offsets_[bin + 1];
    entries_.push_back(entry);
  }
  for (std::size_t bin = 0; bin < total_bins; ++bin)
    offsets_[bin + 1] += offsets_[bin];
}

std::size_t
DirectionGridTracer::GetMaxBinSize() const
{
  std::size_t largest = 0;
  for (std::size_t bin = 0; bin + 1 < offsets_.size(); ++bin)
    largest = std::max(largest, offsets_[bin + 1] - offsets_[bin]);
  return largest + near_triangles_.size();
}

bool
DirectionGridTracer::IsOutside(const Outline& outline, const double u, const double v)
{
  // The point is outside when it is farther than the padding on the outer side of an edge. The
  // edge function is the distance to the edge times its length, which is at most the sum of
  // its extents along u and v.
  for (int k = 0; k < 3; ++k)
  {
    const int next = (k + 1) % 3;
    const double edge_u = static_cast<double>(outline.u[next]) - outline.u[k];
    const double edge_v = static_cast<double>(outline.v[next]) - outline.v[k];
    const double side = edge_u * (v - outline.v[k]) - edge_v * (u - outline.u[k]);
    if (side < -outline_padding * (std::abs(edge_u) + std::abs(edge_v)))
      return true;
  }
  return false;
}

void
DirectionGridTracer::Trace(const std::vector<HullSegment>& segments,
                           HullCrossings& crossings,
                           HullTraceCounters& counters) const
{
  static const std::vector<std::uint32_t> no_excluded_triangles;
  thread_local HullCrossings other_crossings;

  crossings.crossings.clear();
  crossings.offsets.assign(1, 0);
  const auto bins = static_cast<double>(num_bins_);
  std::uint64_t rectangle_tests = 0;
  std::uint64_t triangle_tests = 0;
  for (const auto& segment : segments)
  {
    const std::size_t first = crossings.crossings.size();
    const Vector3 from_apex = segment.start - apex_;
    const double length = from_apex.Norm();
    if ((segment.end - apex_).Norm() > end_tolerance_ or not std::isfinite(length))
    {
      // Not a segment of this grid: the tracer finds its crossings, already sorted.
      tracer_->Trace({segment}, excluded_triangles_, other_crossings, counters);
      --counters.segments;
      counters.crossings -= other_crossings.crossings.size();
      crossings.crossings.insert(crossings.crossings.end(),
                                 other_crossings.crossings.begin(),
                                 other_crossings.crossings.end());
      crossings.offsets.push_back(crossings.crossings.size());
      continue;
    }

    const HullSegmentFrame frame(segment);
    triangle_tests += near_triangles_.size();
    for (const auto triangle : near_triangles_)
      tracer_->TestTriangle(frame, triangle, no_excluded_triangles, crossings.crossings);
    // A start point this close to the apex can only reach the triangles near the apex, and
    // its direction seen from the apex is not reliable.
    if (length > near_distance_)
    {
      // The face of the cube that the direction of the start point goes through, and the
      // coordinates (u, v) of the direction on it.
      const double direction[3] = {from_apex.x, from_apex.y, from_apex.z};
      int axis = 0;
      for (int d = 1; d < 3; ++d)
        if (std::abs(direction[d]) > std::abs(direction[axis]))
          axis = d;
      const std::size_t face = static_cast<std::size_t>(2 * axis) + (direction[axis] < 0.0 ? 1 : 0);
      const double u = direction[(axis + 1) % 3] / std::abs(direction[axis]);
      const double v = direction[(axis + 2) % 3] / std::abs(direction[axis]);
      const auto bin_u =
        static_cast<std::size_t>(std::clamp(std::floor(0.5 * (u + 1.0) * bins), 0.0, bins - 1.0));
      const auto bin_v =
        static_cast<std::size_t>(std::clamp(std::floor(0.5 * (v + 1.0) * bins), 0.0, bins - 1.0));
      const std::size_t bin = (face * num_bins_ + bin_v) * num_bins_ + bin_u;

      const auto reach = static_cast<float>(length);
      const Entry* entry = entries_.data() + offsets_[bin];
      const Entry* const end = entries_.data() + offsets_[bin + 1];
      for (; entry != end and entry->distance <= reach; ++entry)
      {
        ++rectangle_tests;
        if (u < entry->lower_u or u > entry->upper_u or v < entry->lower_v or v > entry->upper_v)
          continue;
        if (entry->outline != no_outline and IsOutside(outlines_[entry->outline], u, v))
          continue;
        ++triangle_tests;
        tracer_->TestTriangle(frame, entry->triangle, no_excluded_triangles, crossings.crossings);
      }
    }
    SegmentTracer::SortCrossings(crossings.crossings, first);
    crossings.offsets.push_back(crossings.crossings.size());
  }
  counters.segments += segments.size();
  counters.box_tests += rectangle_tests;
  counters.triangle_tests += triangle_tests;
  counters.crossings += crossings.crossings.size();
}

} // namespace opensn
