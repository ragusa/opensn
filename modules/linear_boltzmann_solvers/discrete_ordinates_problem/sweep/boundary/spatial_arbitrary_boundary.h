// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/sweep_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include "framework/math/functions/function.h"
#include "framework/data_types/vector3.h"
#include <map>
#include <memory>
#include <vector>

namespace opensn
{

class Mesh;
class SpatialDiscretization;

/**
 * Incident angular flux given by a user function of the position and the direction,
 * psi_inc(x, omega) per group, evaluated at every boundary face node and every quadrature
 * direction. This is the boundary condition of a manufactured solution: the incoming trace of
 * the manufactured angular flux. The function is evaluated once, after the sweep data structures
 * exist (InitializeAngleDependent), and the values are constant in time.
 *
 * Storage in the boundary bank, per groupset: [face node][direction][group], with the direction
 * numbered in angle-set order as the ArbitraryBoundary does.
 */
class SpatialArbitraryBoundary : public SweepBoundary
{
public:
  SpatialArbitraryBoundary(BoundaryBank& bank,
                           std::uint64_t bid,
                           const std::shared_ptr<Mesh>& grid,
                           const SpatialDiscretization& sdm,
                           const std::vector<LBSGroupset>& groupsets,
                           std::shared_ptr<VectorSpatialAngularFunction> function);

  double* PsiIncoming(std::uint32_t cell_local_id,
                      unsigned int face_num,
                      unsigned int fi,
                      unsigned int angle_num,
                      int groupset_id,
                      unsigned int group_idx) override;

  void UpdateBoundaryFlux(const std::vector<LBSGroupset>& groupsets) override;

  void InitializeAngleDependent(const std::vector<LBSGroupset>& groupsets) override;

  std::uint64_t
  GetOffsetToAngleset(const FaceNode& face_node, AngleSet& anglset, bool is_outgoing) override;

private:
  /// Function of the position, the direction and the number of groups: psi_inc per group.
  std::shared_ptr<VectorSpatialAngularFunction> function_;
  /// Boundary face nodes of this boundary, numbered consecutively.
  std::map<FaceNode, std::uint64_t> facenode_to_index_;
  /// Location of every boundary face node, by face-node index.
  std::vector<Vector3> facenode_locations_;
  /// Total number of groups of the problem (the function returns one value per group).
  unsigned int num_groups_ = 0;
  bool evaluated_ = false;

  struct ExtraData
  {
    /// Map from angle number in the quadrature to the internal angle index (angle-set order).
    std::vector<std::uint64_t> map_dirnum;
    /// Bank slots per face node: the number of quadrature directions.
    std::uint64_t node_stride = 0;
  };
  std::vector<ExtraData> extra_data_;
};

} // namespace opensn
