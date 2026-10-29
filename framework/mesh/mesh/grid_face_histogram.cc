// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mesh/mesh/grid_face_histogram.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"
#include <algorithm>
#include <sstream>

namespace opensn
{

std::shared_ptr<GridFaceHistogram>
GridFaceHistogram::Build(std::vector<size_t> face_sizes,
                         double master_tolerance,
                         double slave_tolerance)
{
  std::vector<std::pair<size_t, size_t>> face_categories_list;
  std::vector<size_t>& face_size_histogram = face_sizes;
  std::stable_sort(face_size_histogram.begin(), face_size_histogram.end());

  // Determine total face dofs
  size_t total_face_dofs_count = 0;
  for (auto face_size : face_size_histogram)
    total_face_dofs_count += face_size;

  // Compute average and ratio
  size_t smallest_face = face_size_histogram.front();
  size_t largest_face = face_size_histogram.back();
  size_t total_num_faces = face_size_histogram.size();
  double average_dofs_per_face =
    static_cast<double>(total_face_dofs_count) / static_cast<double>(total_num_faces);

  std::stringstream outstr;
  outstr << "\nSmallest face = " << smallest_face;
  outstr << "\nLargest face = " << largest_face;
  outstr << "\nTotal face dofs = " << total_face_dofs_count;
  outstr << "\nTotal faces = " << face_size_histogram.size();
  outstr << "\nAverage dofs/face = " << average_dofs_per_face;
  outstr << "\nMax to avg ratio = " << static_cast<double>(largest_face) / average_dofs_per_face;
  log.LogAllVerbose2() << outstr.str();

  // Determine number of bins
  size_t last_bin_num_faces = total_num_faces;
  if ((static_cast<double>(largest_face) / average_dofs_per_face) > master_tolerance)
  {
    log.LogAllVerbose2() << "The ratio of max face dofs to average face dofs "
                         << "is larger than " << master_tolerance
                         << ", therefore a binned histogram "
                         << "will be constructed.";

    // Build categories
    size_t running_total_face_dofs = 0;
    size_t running_face_count = 0;
    size_t running_face_size = face_size_histogram[0];

    auto running_average = static_cast<double>(face_size_histogram[0]);

    for (size_t f = 0; f < total_num_faces; ++f)
    {
      if ((static_cast<double>(face_size_histogram[f]) / running_average) > slave_tolerance)
      {
        face_categories_list.emplace_back(running_face_size, running_face_count);
        running_total_face_dofs = 0;
        running_face_count = 0;
      }

      running_face_size = face_size_histogram[f];
      running_total_face_dofs += face_size_histogram[f];
      running_face_count++;
      running_average =
        static_cast<double>(running_total_face_dofs) / static_cast<double>(running_face_count);
      last_bin_num_faces = running_face_count;
    }
  }
  face_categories_list.emplace_back(largest_face, last_bin_num_faces);

  // Verbose print bins
  outstr.str(std::string());
  outstr << "A total of " << face_categories_list.size() << " bins were created:\n";

  int64_t bin_counter = -1;
  for (auto bins : face_categories_list)
  {
    outstr << "Bin " << ++bin_counter << ": " << bins.second << " faces with max face dofs "
           << bins.first << "\n";
  }

  log.LogAllVerbose2() << outstr.str();

  return std::make_shared<GridFaceHistogram>(face_categories_list);
}

size_t
GridFaceHistogram::GetNumberOfFaceHistogramBins() const
{
  return face_categories_.size();
}

size_t
GridFaceHistogram::MapFaceHistogramBins(size_t num_face_verts) const
{
  int64_t category_counter = -1;
  for (auto category : face_categories_)
  {
    category_counter++;
    if (num_face_verts <= category.first)
      return category_counter;
  }

  return 0;
}

size_t
GridFaceHistogram::GetFaceHistogramBinDOFSize(size_t bin_number) const
{
  size_t face_dof_size = 0;

  try
  {
    face_dof_size = face_categories_.at(bin_number).first;
  }
  catch (std::out_of_range& o)
  {
    log.LogAllWarning() << "Fault detected in Mesh::"
                        << "GetFaceHistogramBinDOFSize.";
    return 0;
  }

  return face_dof_size;
}

} // namespace opensn
