// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/spatial_arbitrary_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_aggregation/angle_aggregation.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/math/spatial_discretization/cell_mappings/cell_mapping.h"
#include "framework/math/quadratures/angular/angular_quadrature.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/mesh/cell.h"
#include <limits>
#include <stdexcept>
#include <utility>

namespace opensn
{

SpatialArbitraryBoundary::SpatialArbitraryBoundary(
  BoundaryBank& bank,
  std::uint64_t bid,
  const std::shared_ptr<Mesh>& grid,
  const SpatialDiscretization& sdm,
  const std::vector<LBSGroupset>& groupsets,
  std::shared_ptr<VectorSpatialAngularFunction> function)
  : SweepBoundary(bank, LBSBoundaryType::ARBITRARY), function_(std::move(function))
{
  if (not function_)
    throw std::invalid_argument("SpatialArbitraryBoundary: a function is required.");

  // Boundary face nodes and their locations
  std::uint64_t face_node_counter = 0;
  for (const auto& cell : grid->GetLocalCells())
  {
    const auto& cell_mapping = sdm.GetCellMapping(*cell);
    const auto& node_locations = cell_mapping.GetNodeLocations();
    for (unsigned int f = 0; f < cell->faces.size(); ++f)
    {
      const auto& face = cell->faces[f];
      if (face.has_neighbor or face.neighbor_id != bid)
        continue;
      const auto num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      for (unsigned int fnode = 0; fnode < num_face_nodes; ++fnode)
      {
        facenode_to_index_[FaceNode(cell->local_id, f, fnode)] = face_node_counter++;
        facenode_locations_.push_back(node_locations[cell_mapping.MapFaceNode(f, fnode)]);
      }
    }
  }

  // Bank storage: [face node][direction][group] per groupset
  extra_data_.resize(groupsets.size());
  for (const auto& groupset : groupsets)
  {
    num_groups_ = std::max(num_groups_, groupset.last_group + 1);
    const auto num_angles = groupset.quadrature->GetNumAngles();
    auto& extra_data = extra_data_[groupset.id];
    extra_data.node_stride = num_angles;
    extra_data.map_dirnum.assign(num_angles, std::numeric_limits<std::uint64_t>::max());

    const auto num_slots = face_node_counter * num_angles;
    bank_.ExtendBoundaryFlux(groupset.id, num_slots * groupset.GetNumGroups());
    auto& counter = bank_[groupset.id].counter;
    offset_[groupset.id] = counter;
    counter += num_slots;
  }
}

void
SpatialArbitraryBoundary::InitializeAngleDependent(const std::vector<LBSGroupset>& groupsets)
{
  for (const auto& groupset : groupsets)
  {
    auto& map_dirnum = extra_data_[groupset.id].map_dirnum;
    std::uint64_t i = 0;
    for (const auto& angleset : *groupset.angle_agg)
      for (const auto angle_index : angleset->GetAngleIndices())
        map_dirnum[angle_index] = i++;
  }
  UpdateBoundaryFlux(groupsets);
}

void
SpatialArbitraryBoundary::UpdateBoundaryFlux(const std::vector<LBSGroupset>& groupsets)
{
  // The incident flux does not depend on time: evaluate once.
  if (evaluated_)
    return;

  for (const auto& groupset : groupsets)
  {
    const auto& extra_data = extra_data_[groupset.id];
    const auto& omegas = groupset.quadrature->GetOmegas();
    const auto num_gs_groups = groupset.GetNumGroups();

    for (std::uint64_t idx = 0; idx < facenode_locations_.size(); ++idx)
    {
      // All directions of this node in one call, direction-major over all groups
      const auto values =
        function_->EvaluateDirections(facenode_locations_[idx], omegas, num_groups_);
      for (std::size_t n = 0; n < omegas.size(); ++n)
      {
        double* flux =
          GetBoundaryFlux(groupset.id, idx * extra_data.node_stride + extra_data.map_dirnum[n]);
        const double* values_n = values.data() + n * num_groups_ + groupset.first_group;
        for (unsigned int g = 0; g < num_gs_groups; ++g)
          flux[g] = values_n[g];
      }
    }
  }
  evaluated_ = true;
}

double*
SpatialArbitraryBoundary::PsiIncoming(std::uint32_t cell_local_id,
                                      unsigned int face_num,
                                      unsigned int fi,
                                      unsigned int angle_num,
                                      int groupset_id,
                                      unsigned int group_idx)
{
  const auto idx = facenode_to_index_.at(FaceNode(cell_local_id, face_num, fi));
  const auto& extra_data = extra_data_[groupset_id];
  return GetBoundaryFlux(groupset_id,
                         idx * extra_data.node_stride + extra_data.map_dirnum[angle_num]) +
         group_idx;
}

std::uint64_t
SpatialArbitraryBoundary::GetOffsetToAngleset(const FaceNode& face_node,
                                              AngleSet& anglset,
                                              bool is_outgoing)
{
  if (is_outgoing)
    throw std::logic_error("SpatialArbitraryBoundary does not support outgoing flux.");
  const auto idx = facenode_to_index_.at(face_node);
  const int groupset_id = anglset.GetGroupsetID();
  const auto& extra_data = extra_data_[groupset_id];
  const auto slot =
    idx * extra_data.node_stride + extra_data.map_dirnum[anglset.GetAngleIndices()[0]];
  return (offset_[groupset_id] + slot) * bank_[groupset_id].groupset_size;
}

} // namespace opensn
