// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/object.h"

namespace opensn
{
struct Vector3;

/// Abstract base class for all partitioners
class GraphPartitioner : public Object
{
public:
  /// Given a graph. Returns the partition ids of each row in the graph.
  virtual std::vector<int64_t> Partition(const std::vector<std::vector<uint64_t>>& graph,
                                         const std::vector<Vector3>& centroids,
                                         int number_of_parts) = 0;

  virtual ~GraphPartitioner() = default;

protected:
  static InputParameters GetInputParameters();
  explicit GraphPartitioner(const InputParameters& params);
};

} // namespace opensn
