// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/functions/function.h"
#include "framework/data_types/vector3.h"
#include <stdexcept>

namespace opensn
{

VectorSpatialAngularFunction
VectorSpatialAngularFunction::FromBatched(const Batched& batched)
{
  VectorSpatialAngularFunction f(
    [batched](const Vector3& xyz, const Vector3& omega, std::size_t num_groups)
    {
      auto values = batched(xyz, std::vector<Vector3>{omega}, num_groups);
      if (values.size() != num_groups)
        throw std::runtime_error("VectorSpatialAngularFunction: the batched function returned " +
                                 std::to_string(values.size()) + " values for one direction and " +
                                 std::to_string(num_groups) + " groups.");
      return values;
    });
  f.batched_ = batched;
  return f;
}

std::vector<double>
VectorSpatialAngularFunction::EvaluateDirections(const Vector3& xyz,
                                                 const std::vector<Vector3>& omegas,
                                                 std::size_t num_groups) const
{
  const std::size_t expected = omegas.size() * num_groups;
  if (batched_)
  {
    auto values = batched_(xyz, omegas, num_groups);
    if (values.size() != expected)
      throw std::runtime_error("VectorSpatialAngularFunction: the batched function returned " +
                               std::to_string(values.size()) + " values for " +
                               std::to_string(omegas.size()) + " directions and " +
                               std::to_string(num_groups) + " groups; expected " +
                               std::to_string(expected) + ".");
    return values;
  }

  std::vector<double> values;
  values.reserve(expected);
  for (const auto& omega : omegas)
  {
    const auto v = (*this)(xyz, omega, num_groups);
    if (v.size() != num_groups)
      throw std::runtime_error("VectorSpatialAngularFunction: the function returned " +
                               std::to_string(v.size()) + " values for " +
                               std::to_string(num_groups) + " groups.");
    values.insert(values.end(), v.begin(), v.end());
  }
  return values;
}

} // namespace opensn
