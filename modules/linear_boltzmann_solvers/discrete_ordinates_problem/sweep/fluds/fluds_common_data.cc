// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/math/spatial_discretization/cell_mappings/cell_mapping.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/mesh/cell.h"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace opensn
{

FLUDSCommonData::FLUDSCommonData(const SPDS& spds,
                                 const std::vector<CellFaceNodalMapping>& grid_nodal_mappings)
  : spds_(spds), grid_nodal_mappings_(grid_nodal_mappings)
{
}

const SPDS&
FLUDSCommonData::GetSPDS() const
{
  return spds_;
}

const FaceNodalMapping&
FLUDSCommonData::GetFaceNodalMapping(uint64_t cell_local_id, unsigned int face_id) const
{
  return grid_nodal_mappings_[cell_local_id][face_id];
}

std::vector<CellFaceNodalMapping>
MakeGridNodalMappings(const Mesh& grid, const SpatialDiscretization& sdm)
{
  std::vector<CellFaceNodalMapping> grid_nodal_mappings;
  grid_nodal_mappings.reserve(grid.GetLocalCellCount());

  for (const auto& cell : grid.GetLocalCells())
  {
    const auto& cell_mapping = sdm.GetCellMapping(*cell);
    const auto& node_locations = cell_mapping.GetNodeLocations();

    CellFaceNodalMapping cell_nodal_mapping;
    cell_nodal_mapping.reserve(cell->faces.size());

    for (size_t f = 0; f < cell->faces.size(); ++f)
    {
      const auto& face = cell->faces[f];
      std::vector<short> face_node_mapping;
      std::vector<short> cell_node_mapping;
      int adj_face_idx = -1;

      if (face.has_neighbor)
      {
        const auto& adj_cell = grid.GetGlobalCell(face.neighbor_id);
        const auto& adj_cell_mapping = sdm.GetCellMapping(adj_cell);
        const auto& adj_node_locations = adj_cell_mapping.GetNodeLocations();
        adj_face_idx = face.GetNeighborAdjacentFaceIndex(&grid);

        const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
        const size_t adj_num_face_nodes = adj_cell_mapping.GetNumFaceNodes(adj_face_idx);
        if (adj_num_face_nodes != num_face_nodes)
          throw std::logic_error("MakeGridNodalMappings: face node count mismatch between cell " +
                                 std::to_string(cell->global_id) + " and cell " +
                                 std::to_string(adj_cell.global_id));

        // Relative tolerance from the face size
        double size2 = 0.0;
        for (size_t fi = 0; fi < num_face_nodes; ++fi)
          size2 = std::max(
            size2, (node_locations[cell_mapping.MapFaceNode(f, fi)] - face.centroid).NormSquare());
        const double tol2 = (size2 > 0.0) ? 1.0e-16 * size2 : 1.0e-24;

        face_node_mapping.reserve(num_face_nodes);
        cell_node_mapping.reserve(num_face_nodes);
        for (size_t fi = 0; fi < num_face_nodes; ++fi)
        {
          const auto& xi = node_locations[cell_mapping.MapFaceNode(f, fi)];
          bool found = false;
          for (size_t afi = 0; afi < adj_num_face_nodes; ++afi)
          {
            const int ai = adj_cell_mapping.MapFaceNode(adj_face_idx, afi);
            if ((xi - adj_node_locations[ai]).NormSquare() <= tol2)
            {
              face_node_mapping.push_back(static_cast<short>(afi));
              cell_node_mapping.push_back(static_cast<short>(ai));
              found = true;
              break;
            }
          }
          if (not found)
            throw std::logic_error(
              "MakeGridNodalMappings: face node mapping failed. Could not find a matching node "
              "on the neighbour of cell " +
              std::to_string(cell->global_id) + ", face " + std::to_string(f) + " (neighbour " +
              std::to_string(face.neighbor_id) + ", centroid " + face.centroid.PrintStr() + ")");
        }
      }

      cell_nodal_mapping.emplace_back(adj_face_idx, face_node_mapping, cell_node_mapping);
    }

    grid_nodal_mappings.push_back(std::move(cell_nodal_mapping));
  }

  return grid_nodal_mappings;
}

} // namespace opensn
