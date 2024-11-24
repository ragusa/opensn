// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mesh/mesh_continuum/mesh_continuum.h"

namespace opensn
{

Cell&
LocalCellHandler::operator[](uint64_t cell_local_index)
{
  if (native_cells_.empty())
  {
    throw std::out_of_range("LocalCells attempted to access local cell " +
                            std::to_string(cell_local_index) +
                            ". Local cells are empty. Check the partitioning.");
  }

  if (cell_local_index >= native_cells_.size())
  {
    throw std::out_of_range("Local cell index out of range: " + std::to_string(cell_local_index) +
                            " (max: " + std::to_string(native_cells_.size() - 1) + ").");
  }

  return *native_cells_[cell_local_index];
}

const Cell&
LocalCellHandler::operator[](uint64_t cell_local_index) const
{
  if (native_cells_.empty())
  {
    throw std::out_of_range("LocalCells attempted to access local cell " +
                            std::to_string(cell_local_index) +
                            ". Local cells are empty. Check the partitioning.");
  }

  if (cell_local_index >= native_cells_.size())
  {
    throw std::out_of_range("Local cell index out of range: " + std::to_string(cell_local_index) +
                            " (max: " + std::to_string(native_cells_.size() - 1) + ").");
  }

  return *native_cells_[cell_local_index];
}

} // namespace opensn
