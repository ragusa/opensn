// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/lbs_problem/source_functions/source_flags.h"
#include "framework/materials/multi_group_xs/multi_group_xs.h"
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace opensn
{

class LBSProblem;
class LBSGroupset;

/**
 * Implements a customizable source function using virtual methods.
 * This base class will function well for steady simulations and kEigenvalue
 * simulations. It needs some customization for adjoint and transient.
 */
class SourceFunction
{
public:
  /// Constructor.
  explicit SourceFunction(const LBSProblem& lbs_problem);
  virtual ~SourceFunction() = default;

  /**
   * Sets the source moments for the groups in the current group set.
   *
   * \param groupset The groupset the under consideration.
   * \param q A vector to contribute the source to.
   * \param phi The primary STL vector to operate off.
   * \param source_flags Flags for adding specific terms into the
   *        destination vector. Available flags are for applying
   *        the material source, across/within-group scattering,
   *        and across/within-groups fission.
   */
  virtual void operator()(const LBSGroupset& groupset,
                          std::vector<double>& q,
                          const std::vector<double>& phi,
                          SourceFlags source_flags);

  virtual double FixedSourceMoments() const;

  using PrecursorList = std::vector<MultiGroupXS::Precursor>;
  /// Compute delayed particle precursor sources.
  virtual double DelayedFission(const PrecursorList& precursors,
                                const std::vector<double>& nu_delayed_sigma_f,
                                const double* phi,
                                std::size_t node_id) const;

  virtual void AddAdditionalSources(const LBSGroupset& groupset,
                                    std::vector<double>& q,
                                    const std::vector<double>& phi,
                                    SourceFlags source_flags)
  {
    AddFixedExternalSources(groupset, q, source_flags);
  }

  /**
   * Adds the point and volumetric sources of the groupset's groups to the source moments when
   * `source_flags` requests the fixed sources. The sources are evaluated once into a cache
   * (one moment vector, built per groupset on first use) and the cache is reused until the
   * problem's sources or its time change, so function-based sources, whose evaluation goes
   * through a user callback at every node (and direction), are not re-evaluated for every
   * right-hand-side assembly, Richardson iteration or final sweep.
   */
  void AddFixedExternalSources(const LBSGroupset& groupset,
                               std::vector<double>& q,
                               SourceFlags source_flags);

  /// Adds point sources to the source moments.
  void AddPointSources(const LBSGroupset& groupset,
                       std::vector<double>& q,
                       const std::vector<double>& phi,
                       SourceFlags source_flags);

  /// Adds volumetric sources to the source moments.
  void AddVolumetricSources(const LBSGroupset& groupset,
                            std::vector<double>& q,
                            const std::vector<double>& phi,
                            SourceFlags source_flags);

protected:
  /**
   * Returns the cached fixed-source moments with the groupset's groups up to date, rebuilding
   * the cache when the sources or the time changed since it was filled.
   */
  const std::vector<double>& FixedExternalSourceCache(const LBSGroupset& groupset);

  const LBSProblem& lbs_problem_;

  /// Point and volumetric sources evaluated into moments, laid out like the phi vector.
  std::vector<double> fixed_ext_src_cache_;
  /// Per groupset: whether its groups of the cache are up to date.
  std::vector<bool> fixed_ext_src_cache_valid_;
  std::uint64_t fixed_ext_src_cache_version_ = 0;
  double fixed_ext_src_cache_time_ = 0.0;

  bool apply_fixed_src_ = false;
  bool apply_wgs_scatter_src_ = false;
  bool apply_ags_scatter_src_ = false;
  bool apply_wgs_fission_src_ = false;
  bool apply_ags_fission_src_ = false;
  bool apply_previous_precursor_src_ = false;
  bool suppress_wg_scatter_src_ = false;

  unsigned int gs_i_ = 0;
  unsigned int gs_f_ = 0;
  unsigned int first_grp_ = 0;
  unsigned int last_grp_ = 0;

  double cell_volume_ = 0.0;
  unsigned int g_ = 0;
  const double* fixed_src_moments_ = nullptr;
  std::vector<double> default_zero_src_;
};

} // namespace opensn
