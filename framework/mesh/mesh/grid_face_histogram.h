// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace opensn
{

/// Utility class for handling face categorizations based on number of vertices.
class GridFaceHistogram
{
private:
  std::vector<std::pair<size_t, size_t>> face_categories_;

public:
  explicit GridFaceHistogram(const std::vector<std::pair<size_t, size_t>>& face_categories)
    : face_categories_(face_categories)
  {
  }

  /**
   * Builds the histogram from the per-face node counts of the local faces (one entry per face).
   * The faces are binned by node count; a binned histogram is produced only when the largest
   * face has more than master_tolerance times the average number of nodes, and a new bin starts
   * when a face has more than slave_tolerance times the running average.
   */
  static std::shared_ptr<GridFaceHistogram> Build(std::vector<size_t> face_sizes,
                                                  double master_tolerance = 100.0,
                                                  double slave_tolerance = 1.1);

  /// Returns the number of bins.
  size_t GetNumberOfFaceHistogramBins() const;

  /// Finds which bin holds the given number of vertices
  size_t MapFaceHistogramBins(size_t num_face_verts) const;

  /// Finds the amount of vertices per face for the given bin.
  size_t GetFaceHistogramBinDOFSize(size_t bin_number) const;
};

} // namespace opensn
