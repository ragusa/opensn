// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mesh/raytrace/brute_force_segment_tracer.h"

namespace opensn
{

BruteForceSegmentTracer::BruteForceSegmentTracer(std::shared_ptr<const MaterialHull> hull)
  : SegmentTracer(std::move(hull))
{
}

void
BruteForceSegmentTracer::Trace(const std::vector<HullSegment>& segments,
                               const std::vector<std::uint32_t>& excluded_triangles,
                               HullCrossings& crossings) const
{
  const auto num_triangles = static_cast<std::uint32_t>(hull_->GetTriangles().size());

  crossings.crossings.clear();
  crossings.offsets.assign(1, 0);
  for (const auto& segment : segments)
  {
    const HullSegmentFrame frame(segment);
    const std::size_t first = crossings.crossings.size();
    for (std::uint32_t triangle = 0; triangle < num_triangles; ++triangle)
      TestTriangle(frame, triangle, excluded_triangles, crossings.crossings);
    SortCrossings(crossings.crossings, first);
    crossings.offsets.push_back(crossings.crossings.size());
  }
}

} // namespace opensn
