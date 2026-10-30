// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/vector3.h"
#include <functional>
#include <vector>

namespace opensn
{

using ScalarSpatialFunction = std::function<double(const Vector3&)>;
using ScalarSpatialMaterialFunction = std::function<double(int, const Vector3&)>;
using ScalarMaterialFunction = std::function<double(double, unsigned int)>;
/// A function of a field value, the block ID and the position at which the value was sampled.
using ScalarMaterialSpatialFunction = std::function<double(double, unsigned int, const Vector3&)>;

/// Base class for evaluating spatial material functions given a coordinate.
class VectorSpatialFunction : public std::function<std::vector<double>(const Vector3&, std::size_t)>
{
public:
  VectorSpatialFunction() = default;
  VectorSpatialFunction(const std::function<std::vector<double>(const Vector3&, std::size_t)>& src)
    : std::function<std::vector<double>(const Vector3&, std::size_t)>(src)
  {
  }
};

/**
 * Base class for evaluating angle-dependent spatial functions: given a position, a direction
 * and the number of groups, return one value per group. Used for angular volumetric sources
 * (method of manufactured solutions).
 */
class VectorSpatialAngularFunction
  : public std::function<std::vector<double>(const Vector3&, const Vector3&, std::size_t)>
{
public:
  using PerDirection =
    std::function<std::vector<double>(const Vector3&, const Vector3&, std::size_t)>;
  /**
   * Batched form: given a position, a list of directions and the number of groups, return
   * the values for every direction and group, direction-major (all groups of the first
   * direction first).
   */
  using Batched =
    std::function<std::vector<double>(const Vector3&, const std::vector<Vector3>&, std::size_t)>;

  VectorSpatialAngularFunction() = default;
  VectorSpatialAngularFunction(const PerDirection& src) : PerDirection(src) {}

  /// Wraps a batched callable; the per-direction call evaluates it with a single direction.
  static VectorSpatialAngularFunction FromBatched(const Batched& batched);

  bool IsBatched() const { return static_cast<bool>(batched_); }

  /**
   * Evaluates the function for all `omegas` at once: one call of the batched callable when
   * there is one, otherwise one per-direction call per entry. The result has
   * `omegas.size() * num_groups` entries, direction-major.
   */
  std::vector<double> EvaluateDirections(const Vector3& xyz,
                                         const std::vector<Vector3>& omegas,
                                         std::size_t num_groups) const;

private:
  Batched batched_;
};

/// Base class for evaluating incoming angular fluxes given group and direction indices.
class AngularFluxFunction : public std::function<double(int, int)>
{
public:
  AngularFluxFunction() = default;
  AngularFluxFunction(const std::function<double(int, int)>& src)
    : std::function<double(int, int)>(src)
  {
  }
};

/// Base class for evaluating incoming angular fluxes given group, direction, and time.
class AngularFluxTimeFunction : public std::function<double(int, int, double)>
{
public:
  AngularFluxTimeFunction() = default;
  AngularFluxTimeFunction(const std::function<double(int, int, double)>& src)
    : std::function<double(int, int, double)>(src)
  {
  }
};

/// Base class for evaluating group-wise strengths given group index and time.
class GroupTimeFunction : public std::function<double(unsigned int, double)>
{
public:
  GroupTimeFunction() = default;
  GroupTimeFunction(const std::function<double(unsigned int, double)>& src)
    : std::function<double(unsigned int, double)>(src)
  {
  }
};

} // namespace opensn
