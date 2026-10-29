// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/parameters/input_parameters.h"
#include <vector>
#include <iostream>
#include <limits>

namespace opensn
{
struct Vector3;
class Cell;
class LogicalVolume;
class VectorSpatialFunction;
class VectorSpatialAngularFunction;
class GroupTimeFunction;
class LBSProblem;

/**
 * A class for multi-group volumetric sources.
 *
 * This class differs from the standard material property sources in that it can be specified
 * for an arbitrary logical volume that may span multiple material regions and with spatial and
 * group-wise behaviors via a VectorSpatialFunction. The source is isotropic unless an angular
 * function is given, in which case the source depends on the direction as well (see
 * EvaluateAngular).
 *
 * The flexibility of this object allows for its use as a standard volumetric source or as
 * a volumetric response function for adjoint calculations.
 */
class VolumetricSource
{
public:
  explicit VolumetricSource(const InputParameters& params);

  /**
   * Recomputes the local subscriber cells for `lbs_problem`.
   *
   * This replaces data from any previous mesh. Share a VolumetricSource only between problems
   * with the same mesh and compatible spatial layouts. Collective.
   */
  void Initialize(const LBSProblem& lbs_problem);

  /**
   * Evaluate the distributed source for a given cell node for all groups
   *
   * If the cell does not belong to the logical volume tied to this source,
   * a vector of zeros are returned.
   */
  std::vector<double>
  operator()(const Cell& cell, const Vector3& xyz, unsigned int num_groups) const;
  std::vector<double>
  Evaluate(const Cell& cell, const Vector3& xyz, unsigned int num_groups, double time) const;

  /// True when the source was defined with an angular function (direction-dependent source).
  bool IsAngular() const { return static_cast<bool>(angular_function_); }

  /**
   * Evaluate an angular source for a given cell node and direction for all groups. The source
   * is the emission density per unit solid angle, normalised like the angular flux: the
   * direction-average of the returned values is what an isotropic source of the same strength
   * would be.
   *
   * If the cell does not belong to the logical volume tied to this source, a vector of zeros is
   * returned. Throws when the source is not angular.
   */
  std::vector<double> EvaluateAngular(const Cell& cell,
                                      const Vector3& xyz,
                                      const Vector3& omega,
                                      unsigned int num_groups,
                                      double time) const;

  size_t GetNumLocalSubscribers() const { return num_local_subsribers_; }
  size_t GetNumGlobalSubsribers() const { return num_global_subscribers_; }

  const std::vector<std::uint32_t>& GetSubscribers() const { return subscribers_; }
  std::shared_ptr<LogicalVolume> GetLogicalVolume() const { return logvol_; }
  const std::vector<unsigned int>& GetBlockIDs() const { return block_ids_; }
  bool IsActive(double time) const;

private:
  int id_;

  std::vector<unsigned int> block_ids_;
  const std::shared_ptr<LogicalVolume> logvol_;

  std::vector<double> strength_;
  const std::shared_ptr<VectorSpatialFunction> function_;
  const std::shared_ptr<VectorSpatialAngularFunction> angular_function_;
  const std::shared_ptr<GroupTimeFunction> strength_function_;

  double start_time_ = -std::numeric_limits<double>::infinity();
  double end_time_ = std::numeric_limits<double>::infinity();

  size_t num_local_subsribers_ = 0;
  size_t num_global_subscribers_ = 0;

  std::vector<std::uint32_t> subscribers_;

public:
  static InputParameters GetInputParameters();
  static std::shared_ptr<VolumetricSource> Create(const ParameterBlock& params);

private:
  static int next_id_;
};

} // namespace opensn
