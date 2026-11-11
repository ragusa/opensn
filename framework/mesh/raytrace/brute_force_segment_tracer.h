// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/mesh/raytrace/segment_tracer.h"

namespace opensn
{

/// Reference tracer that tests every triangle of the hull against every segment.
class BruteForceSegmentTracer : public SegmentTracer
{
public:
  explicit BruteForceSegmentTracer(std::shared_ptr<const MaterialHull> hull);

  void Trace(const std::vector<HullSegment>& segments,
             const std::vector<std::uint32_t>& excluded_triangles,
             HullCrossings& crossings) const override;
};

} // namespace opensn
