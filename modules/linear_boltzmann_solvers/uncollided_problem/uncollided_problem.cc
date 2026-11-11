// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/uncollided_problem/uncollided_problem.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/point_source/point_source.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_discontinuous.h"
#include "framework/math/spatial_discretization/cell_mappings/cell_mapping.h"
#include "framework/math/spatial_weight_function.h"
#include "framework/math/quadratures/quadrature_order.h"
#include "framework/math/quadratures/gausslegendre_quadrature.h"
#include "framework/math/quadratures/angular/legendre_poly/legendrepoly.h"
#include "framework/data_types/allowable_range.h"
#include "framework/logging/log.h"
#include "framework/utils/error.h"
#include "framework/utils/timer.h"
#include "framework/utils/utils.h"
#include "framework/utils/hdf_utils.h"
#include "framework/parameters/input_parameters.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <boost/graph/topological_sort.hpp>
#include <iomanip>
#include <numeric>
#include <queue>
#include <utility>
#include <unordered_map>
#include <cmath>
#include <exception>
#include <algorithm>
#include <array>
#include <limits>

namespace opensn
{
namespace
{

std::string
FormatDuration(const double seconds)
{
  const auto total_seconds = static_cast<uint64_t>(std::max(0.0, seconds));
  const auto hours = total_seconds / 3600;
  const auto minutes = (total_seconds % 3600) / 60;
  const auto remaining_seconds = total_seconds % 60;

  std::ostringstream output;
  output << std::setfill('0') << std::setw(2) << hours << ':' << std::setw(2) << minutes << ':'
         << std::setw(2) << remaining_seconds;
  return output.str();
}

/// Concatenates the values of every rank, in rank order, on rank 0. The other ranks get an empty
/// vector. Collective.
template <typename T>
std::vector<T>
GatherOnRoot(const std::vector<T>& local_values)
{
  if (opensn::mpi_comm.size() == 1)
    return local_values;

  constexpr auto max_count = static_cast<size_t>(std::numeric_limits<int>::max());
  const int num_ranks = opensn::mpi_comm.size();
  // A count that does not fit is reported as -1 so that every rank learns of it.
  const int local_size =
    local_values.size() > max_count ? -1 : static_cast<int>(local_values.size());
  std::vector<int> sizes(num_ranks, 0);
  opensn::mpi_comm.all_gather(local_size, sizes);

  std::vector<int> counts(num_ranks, 0);
  std::vector<int> offsets(num_ranks, 0);
  size_t total_size = 0;
  bool too_large = false;
  for (int rank = 0; rank < num_ranks; ++rank)
  {
    too_large = too_large or sizes[rank] < 0 or total_size > max_count;
    if (too_large)
      break;
    offsets[rank] = static_cast<int>(total_size);
    total_size += static_cast<size_t>(sizes[rank]);
  }
  OpenSnLogicalErrorIf(too_large or total_size > max_count,
                       "The uncollided data of all ranks is too large to gather on one rank.");
  if (opensn::mpi_comm.rank() == 0)
    counts = sizes;

  std::vector<T> global_values;
  opensn::mpi_comm.gather(local_values, global_values, counts, offsets, 0);
  return global_values;
}

/// Runs work on rank 0. If it throws, every rank throws, so that no rank waits for a rank that
/// has stopped. Collective.
template <typename Work>
void
RunOnRoot(const Work& work)
{
  int failed = 0;
  std::exception_ptr failure;
  if (opensn::mpi_comm.rank() == 0)
  {
    try
    {
      work();
    }
    catch (...)
    {
      failed = 1;
      failure = std::current_exception();
    }
  }
  opensn::mpi_comm.broadcast(failed, 0);
  if (failure)
    std::rethrow_exception(failure);
  OpenSnLogicalErrorIf(failed != 0, "Uncollided generation failed on rank 0.");
}

} // namespace

InputParameters
UncollidedProblem::GetInputParameters()
{
  InputParameters params = LBSProblem::GetInputParameters();

  params.ChangeExistingParamToOptional("name", "UncollidedProblem");

  params.AddOptionalParameterArray(
    "boundary_conditions",
    {},
    "Vacuum or reflecting boundary conditions. Reflecting boundaries must be planar, mutually "
    "orthogonal symmetry planes without an opposing reflecting plane.");
  // Each "boundary_conditions" entry follows the "BoundaryOptionsBlock" parameter-block schema.

  params.AddOptionalParameterArray(
    "near_source", {}, "List of near-source logical volumes, one for each point source.");

  params.AddOptionalParameter(
    "scattering_order", 0, "The scattering order of the collided flux problem.");

  params.AddOptionalParameter(
    "exact_near_source_leakage",
    false,
    "Integrate the face leakages and volume integrals of the near-source cells that share a "
    "vertex with a point source's cell exactly over direction, instead of with spatial "
    "quadrature. This improves accuracy when a source lies close to an edge or vertex of its "
    "cell, at the cost of additional ray tracing.");

  params.AddOptionalParameter(
    "ray_tracer",
    "cell_walk",
    "How the optical depth between a point and a source is computed. \"cell_walk\" follows the "
    "path from cell to cell. \"material_hull\" intersects the path only with the faces across "
    "which the material changes and with the domain boundary, so its cost does not grow with "
    "the number of cells along the path. Both give the same optical depth to round-off.");
  params.ConstrainParameterRange("ray_tracer",
                                 AllowableRangeList::New({"cell_walk", "material_hull"}));

  params.AddOptionalParameter(
    "method",
    "sweep",
    "How the cells outside the near-source regions are treated. \"sweep\" solves the "
    "advection-removal equation of the uncollided flux in them with a sweep from the source "
    "outward. \"ray_trace\" has no such cells: every cell is ray traced and corrected for "
    "conservation, as the near-source cells are, and \"near_source\" is not used.");
  params.ConstrainParameterRange("method", AllowableRangeList::New({"sweep", "ray_trace"}));

  params.AddOptionalParameter(
    "hull_tracer_backend",
    "bvh",
    "Implementation of the material-hull ray tracer. \"bvh\" uses a bounding volume hierarchy. "
    "\"brute_force\" tests every triangle of the hull and is meant for verification.");
  params.ConstrainParameterRange("hull_tracer_backend",
                                 AllowableRangeList::New({"bvh", "brute_force"}));

  return params;
}

std::shared_ptr<UncollidedProblem>
UncollidedProblem::Create(const ParameterBlock& params)
{
  return Build(std::shared_ptr<UncollidedProblem>(new UncollidedProblem(
    MakeInputParameters<UncollidedProblem>("lbs::UncollidedProblem", params))));
}

void
UncollidedProblem::CheckConfigurationErrors(std::vector<std::string>& errors) const
{
  LBSProblem::CheckConfigurationErrors(errors);

  if (opensn::mpi_comm.size() != 1 and
      not(ray_trace_all_cells_ and ray_tracer_type_ == RayTracerType::MATERIAL_HULL))
    errors.emplace_back("uncollided flux generation must run with exactly one MPI rank, unless "
                        "every cell is ray traced with the material-hull ray tracer (method = "
                        "\"ray_trace\" and ray_tracer = \"material_hull\").");
  if (grid_->GetDimension() != 3)
    errors.emplace_back("only three-dimensional meshes are supported.");
  if (grid_->GetCoordinateSystem() != CoordinateSystemType::CARTESIAN)
    errors.emplace_back("only Cartesian meshes are supported.");
  if (not GetVolumetricSources().empty())
    errors.emplace_back("only point sources are supported.");
  if (GetPointSources().empty())
    errors.emplace_back("at least one point source is required.");
  // The generator computes the forward uncollided flux of its point sources.
  if (options_.adjoint)
    errors.emplace_back("uncollided flux generation does not support adjoint mode.");
  // The near-source list is indexed one-to-one with the point-source list.
  const auto planes = ComputeReflectionPlanes(errors);
  // A source on a reflecting plane coincides with its own image.
  for (size_t i = 0; i < GetPointSources().size(); ++i)
    for (const auto& plane : planes)
    {
      const Vector3& location = GetPointSources()[i]->GetLocation();
      const double tolerance = 1.0e-12 * std::max(1.0, std::abs(plane.offset));
      if (std::abs(plane.normal.Dot(location) - plane.offset) <= tolerance)
        errors.emplace_back("point source " + std::to_string(i) +
                            " lies on a reflecting boundary, which is not supported for "
                            "uncollided generation.");
    }
  if (ray_trace_all_cells_)
  {
    // Every cell is ray traced, so no region is needed.
    if (not near_source_logvols_.empty())
      errors.emplace_back("near-source logical volumes are not used with method = "
                          "\"ray_trace\", which ray traces every cell.");
  }
  else if (near_source_logvols_.size() != GetPointSources().size())
    errors.emplace_back(
      "the number of near-source logical volumes must match the number of point sources.");
  else
    for (size_t i = 0; i < near_source_logvols_.size(); ++i)
      if (near_source_logvols_[i] and
          not near_source_logvols_[i]->Inside(GetPointSources()[i]->GetLocation()))
        errors.emplace_back("point source " + std::to_string(i) +
                            " lies outside its near-source logical volume.");

  // Point sources are located in the mesh when the problem is built. A source that belongs to
  // several cells is supported on the domain boundary, where each cell receives the part of
  // the source emitted into the solid angle it subtends and the rest leaves the domain.
  if (discretization_)
    for (const auto& point_source : GetPointSources())
      if (point_source->GetNumGlobalSubscribers() != 1 and not IsOnDomainBoundary(*point_source))
      {
        errors.emplace_back("point sources on a face, edge, or vertex shared by more than one "
                            "cell are not currently supported for uncollided generation, "
                            "except on the domain boundary.");
        break;
      }
}

bool
UncollidedProblem::IsOnDomainBoundary(const PointSource& point_source) const
{
  // The source is on the boundary when it lies on a side triangle of a boundary face of one of
  // its cells, which can be on any rank.
  constexpr double relative_tolerance = 1.0e-9;
  const Vector3& location = point_source.GetLocation();
  int on_boundary = 0;
  for (const auto& subscriber : point_source.GetSubscribers())
  {
    const Cell& cell = grid_->GetLocalCell(subscriber.cell_local_id);
    for (const auto& face : cell.faces)
    {
      if (face.has_neighbor)
        continue;
      const size_t num_sides = face.vertex_ids.size();
      for (size_t side = 0; side < num_sides; ++side)
      {
        const Vector3& a = grid_->GlobalVertex(face.vertex_ids[side]);
        const Vector3& b = grid_->GlobalVertex(face.vertex_ids[(side + 1) % num_sides]);
        const double distance =
          (ClosestPointOnTriangle(location, a, b, face.centroid) - location).Norm();
        if (distance <= relative_tolerance * (b - a).Norm())
          on_boundary = 1;
      }
    }
  }
  opensn::mpi_comm.all_reduce(on_boundary, mpi::op::max<int>());
  return on_boundary != 0;
}

UncollidedProblem::UncollidedProblem(const InputParameters& params)
  : LBSProblem(params),
    ray_tracer_type_(params.GetParamValue<std::string>("ray_tracer") == "material_hull"
                       ? RayTracerType::MATERIAL_HULL
                       : RayTracerType::CELL_WALK),
    hull_tracer_backend_(params.GetParamValue<std::string>("hull_tracer_backend")),
    ray_trace_all_cells_(params.GetParamValue<std::string>("method") == "ray_trace"),
    ell_max_(params.GetParamValue<size_t>("scattering_order")),
    exact_near_source_leakage_(params.GetParamValue<bool>("exact_near_source_leakage"))
{
  num_moments_ = 1;

  const auto& near_source_param = params.GetParam("near_source");
  near_source_param.RequireBlockTypeIs(ParameterBlockType::ARRAY);
  for (const auto& log_vol : near_source_param)
    near_source_logvols_.push_back(log_vol.GetValue<std::shared_ptr<LogicalVolume>>());

  // ComputeReflectionPlanes() interprets the boundary conditions.
  const auto& boundary_conditions = params.GetParam("boundary_conditions");
  boundary_conditions.RequireBlockTypeIs(ParameterBlockType::ARRAY);
  for (const auto& boundary : boundary_conditions)
    boundary_conditions_.emplace_back(boundary.GetParamValue<std::string>("name"),
                                      boundary.GetParamValue<std::string>("type"));
}

void
UncollidedProblem::InitializeSpatialDiscretization()
{
  CALI_CXX_MARK_SCOPE("InitializeSpatialDiscretization");

  log.Log() << "Initializing spatial discretization.\n";
  discretization_ = PieceWiseLinearDiscontinuous::New(grid_, QuadratureOrder::FOURTH);

  ComputeUnitIntegrals();

  // Pre-compute per-cell bounding-box diagonals for RayTracer tolerance scaling.
  // Without this, RayTracer::TraceRay() recomputes it on every call by iterating
  // over all cell vertices.
  cell_sizes_.resize(grid_->GetLocalCellCount());
  for (const auto& cell : grid_->GetLocalCells())
  {
    const auto& v0 = grid_->GlobalVertex(cell->vertex_ids.front());
    double xmin = v0.x, xmax = v0.x;
    double ymin = v0.y, ymax = v0.y;
    double zmin = v0.z, zmax = v0.z;
    for (const auto vid : cell->vertex_ids)
    {
      const auto& v = grid_->GlobalVertex(vid);
      xmin = std::min(xmin, v.x);
      xmax = std::max(xmax, v.x);
      ymin = std::min(ymin, v.y);
      ymax = std::max(ymax, v.y);
      zmin = std::min(zmin, v.z);
      zmax = std::max(zmax, v.z);
    }
    cell_sizes_[cell->local_id] =
      std::max((Vector3(xmax, ymax, zmax) - Vector3(xmin, ymin, zmin)).Norm(), 1.0);
  }

  // Detect non-convex cells so the correct TraceRay path is selected.
  // For each face (outward normal n), every off-face vertex v of a convex cell satisfies
  // dot(n, v - v0) <= 0. Standard hex/tet meshes are fully convex.
  const auto& local_cells = grid_->GetLocalCells();
  all_cells_convex_ = std::all_of(
    local_cells.begin(), local_cells.end(), [this](const auto& cell) { return IsConvex(*cell); });
  if (not all_cells_convex_)
    log.Log() << "Non-convex cells detected; concavity checks enabled globally.";

  // Precompute flat face vertex data when all cells are convex and all faces fit in FaceVertData.
  // This bypasses the std::map-backed VertexHandler and the two O(log N) GlobalCellHandler lookups
  // per while-loop iteration in RaytraceLineInto, replacing them with flat-array and unordered_map
  // accesses that hit the L2 cache instead of chasing red-black tree nodes.
  if (all_cells_convex_)
  {
    size_t total_faces = 0;
    bool can_fast = true;
    for (const auto& cell : grid_->GetLocalCells())
    {
      for (const auto& face : cell->faces)
      {
        if (face.vertex_ids.size() > FaceVertData::max_sides)
        {
          can_fast = false;
          break;
        }
      }
      if (not can_fast)
        break;
      total_faces += cell->faces.size();
    }

    if (can_fast)
    {
      all_face_verts_.resize(total_faces);
      cell_face_offsets_.resize(grid_->GetLocalCellCount());
      cell_num_faces_.resize(grid_->GetLocalCellCount());
      global_to_local_id_.reserve(grid_->GetLocalCellCount());
      size_t offset = 0;
      for (const auto& cell : grid_->GetLocalCells())
      {
        cell_face_offsets_[cell->local_id] = static_cast<uint32_t>(offset);
        cell_num_faces_[cell->local_id] = static_cast<uint32_t>(cell->faces.size());
        global_to_local_id_[cell->global_id] = static_cast<uint32_t>(cell->local_id);
        for (const auto& face : cell->faces)
        {
          auto& fv = all_face_verts_[offset++];
          fv.num_sides = static_cast<uint32_t>(face.vertex_ids.size());
          for (size_t s = 0; s < fv.num_sides; ++s)
            fv.verts[s] = grid_->GlobalVertex(face.vertex_ids[s]);
          fv.centroid = face.centroid;
          fv.neighbor_id = face.neighbor_id;
          fv.pad = 0;
        }
      }
      use_fast_trace_ = true;
    }
  }
}

std::vector<UncollidedProblem::ReflectionPlane>
UncollidedProblem::ComputeReflectionPlanes(std::vector<std::string>& errors) const
{
  constexpr double tolerance = 1.0e-12;
  std::vector<ReflectionPlane> planes;
  for (const auto& [name, type] : boundary_conditions_)
  {
    if (type != "vacuum" and type != "reflecting")
    {
      errors.emplace_back("uncollided transport supports only vacuum and reflecting boundary "
                          "conditions.");
      continue;
    }
    if (type == "vacuum")
      continue;

    const auto boundary_it = grid_->GetBoundaryNameMap().find(name);
    if (boundary_it == grid_->GetBoundaryNameMap().end())
    {
      errors.emplace_back("boundary name \"" + name + "\" was not found.");
      continue;
    }
    const auto boundary_id = boundary_it->second;

    bool found_face = false;
    bool planar = true;
    Vector3 normal;
    double offset = 0.0;
    for (const auto& cell : grid_->GetLocalCells())
      for (const auto& face : cell->faces)
        if (not face.has_neighbor and face.neighbor_id == boundary_id)
        {
          if (not found_face)
          {
            normal = face.normal.Normalized();
            offset = normal.Dot(face.centroid);
            found_face = true;
          }
          else if (std::abs(normal.Dot(face.normal.Normalized()) - 1.0) > tolerance or
                   std::abs(normal.Dot(face.centroid) - offset) > tolerance)
            planar = false;
        }

    if (opensn::mpi_comm.size() > 1)
    {
      // The faces of a boundary can be on any rank. The first rank that has some sets the
      // plane, and the planes of the other ranks are compared with it.
      const int num_ranks = opensn::mpi_comm.size();
      constexpr int num_values = 6;
      const std::vector<double> local_plane = {
        found_face ? 1.0 : 0.0, planar ? 1.0 : 0.0, normal.x, normal.y, normal.z, offset};
      std::vector<int> counts(num_ranks, num_values);
      std::vector<int> offsets(num_ranks, 0);
      for (int rank = 0; rank < num_ranks; ++rank)
        offsets[rank] = rank * num_values;
      std::vector<double> planes_of_ranks(static_cast<size_t>(num_ranks) * num_values);
      opensn::mpi_comm.all_gather(local_plane, planes_of_ranks, counts, offsets);

      found_face = false;
      planar = true;
      for (int rank = 0; rank < num_ranks; ++rank)
      {
        const double* plane = &planes_of_ranks[static_cast<size_t>(rank) * num_values];
        if (plane[0] == 0.0)
          continue;
        const Vector3 rank_normal(plane[2], plane[3], plane[4]);
        if (not found_face)
        {
          normal = rank_normal;
          offset = plane[5];
          found_face = true;
        }
        else if (std::abs(normal.Dot(rank_normal) - 1.0) > tolerance or
                 std::abs(plane[5] - offset) > tolerance)
          planar = false;
        if (plane[1] == 0.0)
          planar = false;
      }
    }

    if (not found_face)
      errors.emplace_back("reflecting boundary \"" + name + "\" has no faces.");
    else if (not planar)
      errors.emplace_back("reflecting boundary \"" + name + "\" is not planar.");
    else
      planes.push_back({boundary_id, normal, offset});
  }

  for (size_t i = 0; i < planes.size(); ++i)
    for (size_t j = i + 1; j < planes.size(); ++j)
      if (std::abs(planes[i].normal.Dot(planes[j].normal)) > tolerance)
        errors.emplace_back(
          "reflecting boundaries must be mutually orthogonal symmetry planes. Opposing or oblique "
          "reflecting planes require an unbounded image-source series and are not supported.");
  return planes;
}

void
UncollidedProblem::BuildRuntimeData()
{
  LBSProblem::BuildRuntimeData();
  std::vector<std::string> errors;
  reflection_planes_ = ComputeReflectionPlanes(errors);
  for (const auto& plane : reflection_planes_)
    reflecting_boundary_ids_.insert(plane.boundary_id);
}

void
UncollidedProblem::BuildSourcePoints()
{
  source_points_.clear();
  reflected_source_points_.clear();

  for (size_t source_index = 0; source_index < point_sources_.size(); ++source_index)
  {
    const auto& point_source = point_sources_[source_index];
    SourcePoint source_point{point_source->GetLocation(),
                             point_source->GetStrength(),
                             {},
                             ray_trace_all_cells_ ? nullptr : near_source_logvols_[source_index],
                             {}};
    for (const auto& subscriber : point_source->GetSubscribers())
      source_point.subscribers.push_back({subscriber.cell_local_id, subscriber.volume_weight});
    source_points_.push_back(std::move(source_point));

    // A source near a cell face can be poorly resolved by spatial quadrature.
    const auto& subscribers = source_points_.back().subscribers;
    if (subscribers.size() == 1)
    {
      const Cell& source_cell = grid_->GetLocalCell(subscribers.front().cell_local_id);
      const Vector3& loc = source_points_.back().location;

      double min_face_dist = std::numeric_limits<double>::max();
      double centroid_clearance = std::numeric_limits<double>::max();
      for (const auto& face : source_cell.faces)
      {
        min_face_dist = std::min(min_face_dist, std::abs(face.normal.Dot(loc - face.centroid)));
        centroid_clearance = std::min(
          centroid_clearance, std::abs(face.normal.Dot(source_cell.centroid - face.centroid)));
      }

      constexpr double min_relative_face_distance = 0.1;
      if (centroid_clearance > 0.0 and
          min_face_dist < min_relative_face_distance * centroid_clearance)
        log.Log0Warning()
          << GetName() << ": point source " << source_index << " at " << loc.PrintStr()
          << " lies only " << min_face_dist << " from a face of its containing cell (global id "
          << source_cell.global_id << ", centroid " << source_cell.centroid.PrintStr()
          << ", centroid clearance " << centroid_clearance
          << "). The near-source ray-traced flux evaluation is likely to be poorly resolved "
             "there; consider refining the mesh near this source, or moving it closer to its "
             "containing cell's centroid, so it sits more centrally within its cell.";
    }
  }
}

void
UncollidedProblem::AddReflectedSourcePoints()
{
  if (reflection_planes_.empty())
    return;

  const size_t physical_source_count = source_points_.size();
  const size_t num_image_combinations = (size_t{1} << reflection_planes_.size()) - 1;
  reflected_source_points_.reserve(physical_source_count * num_image_combinations);

  for (size_t source_index = 0; source_index < physical_source_count; ++source_index)
  {
    const auto& physical_source = source_points_[source_index];
    for (size_t mask = 1; mask <= num_image_combinations; ++mask)
    {
      auto image_location = physical_source.location;
      std::vector<ReflectionPlane> image_planes;
      for (size_t p = 0; p < reflection_planes_.size(); ++p)
        if ((mask & (size_t{1} << p)) != 0)
        {
          const auto& plane = reflection_planes_[p];
          image_location -= 2.0 * (plane.normal.Dot(image_location) - plane.offset) * plane.normal;
          image_planes.push_back(plane);
        }

      reflected_source_points_.push_back(SourcePoint{
        image_location, physical_source.strength, {}, nullptr, std::move(image_planes)});
    }
  }

  log.Log() << "Added " << reflected_source_points_.size() << " reflected image sources for "
            << physical_source_count << " physical sources "
            << "and " << reflection_planes_.size() << " symmetry planes.";
}

UncollidedProblem::~UncollidedProblem() = default;

void
UncollidedProblem::PrintSimHeader()
{
  if (opensn::mpi_comm.rank() == 0)
  {
    std::stringstream outstr;
    outstr << "\nInitializing " << GetName() << "\n";
    log.Log() << outstr.str() << '\n';
  }
}

void
UncollidedProblem::BuildSweepOrdering(const SourcePoint& source_point)
{
  CALI_CXX_MARK_SCOPE("BuildSweepOrdering");

  constexpr double tolerance = 1.0e-16;

  constexpr auto FOPARALLEL = FaceOrientation::PARALLEL;
  constexpr auto FOINCOMING = FaceOrientation::INCOMING;
  constexpr auto FOOUTGOING = FaceOrientation::OUTGOING;

  const size_t num_local_cells = grid_->GetLocalCellCount();
  cell_face_orientations_.assign(num_local_cells, {});
  for (const auto& cell : grid_->GetLocalCells())
    cell_face_orientations_[cell->local_id].assign(cell->faces.size(), FOPARALLEL);

  for (const auto& cell : grid_->GetLocalCells())
  {
    size_t f = 0;
    for (auto& face : cell->faces)
    {
      // Determine if the face is incident. A face whose plane contains the source carries no
      // current. It is recognized by its distance to the source and not by the sign of mu, which
      // is then round-off: the centroid of a face in the plane z = 0.7 is not exactly in it.
      const auto ContainsSource = [&source_point](const CellFace& candidate)
      {
        const Vector3 to_centroid = candidate.centroid - source_point.location;
        return std::abs(candidate.normal.Dot(to_centroid)) <=
               1.0e-12 * std::max(to_centroid.Norm(), std::sqrt(candidate.area));
      };
      FaceOrientation orientation = FOPARALLEL;
      Vector3 omega = ComputeOmega(source_point.location, face.centroid);
      const double mu = omega.Dot(face.normal);

      bool owns_face = true;
      if (face.has_neighbor and cell->global_id > face.neighbor_id)
        owns_face = false;

      if (owns_face)
      {
        if (ContainsSource(face))
          orientation = FOPARALLEL;
        else if (mu > tolerance)
          orientation = FOOUTGOING;
        else if (mu < -tolerance)
          orientation = FOINCOMING;

        cell_face_orientations_[cell->local_id][f] = orientation;

        if (face.has_neighbor and face.IsNeighborLocal(grid_.get()))
        {
          const auto& adj_cell = grid_->GetGlobalCell(face.neighbor_id);
          const auto adj_face_idx = face.GetNeighborAdjacentFaceIndex(grid_.get());
          auto& adj_face_ori = cell_face_orientations_[adj_cell.local_id][adj_face_idx];

          switch (orientation)
          {
            case FOPARALLEL:
              adj_face_ori = FOPARALLEL;
              break;
            case FOINCOMING:
              adj_face_ori = FOOUTGOING;
              break;
            case FOOUTGOING:
              adj_face_ori = FOINCOMING;
              break;
          }
        }
      }
      else if (not face.IsNeighborLocal(grid_.get()))
      {
        // The cell that owns the face is on another rank. Decide as that rank does, from its
        // copy of the face, which this rank holds in a ghost cell.
        const auto& adj_cell = grid_->GetGlobalCell(face.neighbor_id);
        const auto& adj_face = adj_cell.faces[face.GetNeighborAdjacentFaceIndex(grid_.get())];
        const double adj_mu =
          ComputeOmega(source_point.location, adj_face.centroid).Dot(adj_face.normal);
        if (ContainsSource(adj_face))
          orientation = FOPARALLEL;
        else if (adj_mu > tolerance)
          orientation = FOINCOMING;
        else if (adj_mu < -tolerance)
          orientation = FOOUTGOING;
        cell_face_orientations_[cell->local_id][f] = orientation;
      }

      ++f;
    } // for face
  }

  Graph local_cell_graph(num_local_cells);
  for (const auto& cell : grid_->GetLocalCells())
  {
    for (size_t f = 0; f < cell->faces.size(); ++f)
      if (cell_face_orientations_[cell->local_id][f] == FOOUTGOING and
          cell->faces[f].has_neighbor and cell->faces[f].IsNeighborLocal(grid_.get()))
        boost::add_edge(
          cell->local_id, cell->faces[f].GetNeighborLocalID(grid_.get()), 0.0, local_cell_graph);
  }

  std::vector<size_t> sweep_order;
  try
  {
    boost::topological_sort(local_cell_graph, std::back_inserter(sweep_order));
  }
  catch (const boost::not_a_dag&)
  {
    OpenSnLogicalError(GetName() +
                       ": the cells cannot be ordered outward from the point source "
                       "at " +
                       source_point.location.PrintStr() +
                       " because their face orientations form a cycle.");
  }
  std::reverse(sweep_order.begin(), sweep_order.end());

  // Near-source region cells plus their incoming dependencies.
  //
  // A raytraced cell can depend on incoming leakage from an upwind cell.
  // Ensure the near-source region includes all incoming dependencies so
  // it is self-contained.
  std::vector<bool> is_near_source(num_local_cells, false);
  for (const size_t cell_id : sweep_order)
  {
    const auto& cell = grid_->GetLocalCell(cell_id);
    if (ray_trace_all_cells_ or (source_point.near_source_logvol and
                                 source_point.near_source_logvol->Inside(cell.centroid)))
      is_near_source[cell_id] = true;
  }

  // A source may lie on a face or vertex and have multiple subscriber cells.
  // Subscriber cells must always be raytraced, even when their centroids are
  // outside the near-source region, because the source contribution is defined
  // directly in those cells.
  for (const auto& subscriber : source_point.subscribers)
    is_near_source[subscriber.cell_local_id] = true;

  bool added_dependency = true;
  while (added_dependency)
  {
    added_dependency = false;
    for (const size_t cell_id : sweep_order)
    {
      if (not is_near_source[cell_id])
        continue;

      const auto& cell = grid_->GetLocalCell(cell_id);
      const size_t cell_num_faces = cell.faces.size();

      for (size_t f = 0; f < cell_num_faces; ++f)
      {
        const auto& face = cell.faces[f];
        if (not face.has_neighbor or not face.IsNeighborLocal(grid_.get()) or
            cell_face_orientations_[cell_id][f] != FaceOrientation::INCOMING)
          continue;

        const size_t neigh_id = face.GetNeighborLocalID(grid_.get());
        if (not is_near_source[neigh_id])
        {
          is_near_source[neigh_id] = true;
          added_dependency = true;
        }
      }
    }
  }

  near_spls_.clear();
  bulk_spls_.clear();
  cell_regions_.assign(num_local_cells, CellRegion::BULK);
  for (const size_t cell_id : sweep_order)
  {
    if (is_near_source[cell_id])
    {
      near_spls_.push_back(cell_id);
      cell_regions_[cell_id] = CellRegion::NEAR_SOURCE;
    }
    else
      bulk_spls_.push_back(cell_id);
  }
}

// NOLINTBEGIN(clang-analyzer-security.ArrayBound)
// False positive: the analyzer traces Execute() -> BuildSweepOrdering() ->
// boost::topological_sort() into Boost's shared_array_property_map DFS color
// map and cannot prove the vertex index is non-negative through the BGL visitor
// chain. The suppression must cover Execute() because the analyzer anchors the
// diagnostic to the first note it emits in user code, which is here.
void
UncollidedProblem::Execute(const std::string& file_name,
                           const unsigned int progress_interval,
                           const std::string& diagnostics_file_name)
{
  CALI_CXX_MARK_SCOPE("GenerateUncollidedFlux");

  std::fill(phi_new_local_.begin(), phi_new_local_.end(), 0.0);
  production_ = 0.0;
  physical_removal_ = 0.0;
  reflected_removal_ = 0.0;
  out_flow_ = 0.0;

  cell_walk_nudged_paths_ = 0;
  cell_walk_edge_paths_ = 0;
  timings_ = PhaseTimings();
  write_diagnostics_ = not diagnostics_file_name.empty();
  source_diagnostics_.clear();
  Timer phase_timer;
  if (ray_tracer_type_ == RayTracerType::MATERIAL_HULL)
    BuildMaterialHull();
  timings_.material_hull = phase_timer.GetTime() / 1000.0;

  // Create h5 file. Rank 0 writes it, with the data of the other ranks gathered there.
  std::optional<H5FileHandle> file_handle;
  RunOnRoot(
    [this, &file_name, &file_handle]()
    {
      const auto raw_file = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      OpenSnInvalidArgumentIf(
        raw_file < 0, GetName() + ": failed to create uncollided flux file \"" + file_name + "\".");
      file_handle.emplace(raw_file);
    });
  const hid_t file = file_handle ? file_handle->Id() : H5I_INVALID_HID;

  const auto& sdm = *discretization_;

  const size_t num_loc_cells = grid_->GetLocalCellCount();
  const size_t num_loc_nodes = sdm.GetNumLocalNodes();
  OpenSnLogicalErrorIf(
    opensn::mpi_comm.size() == 1 and (num_loc_cells == 0 or num_loc_nodes == 0),
    GetName() + ": uncollided flux generation requires a nonempty local mesh on the serial rank.");
  const size_t num_loc_unknowns = num_loc_nodes * num_groups_;
  moments_.clear();
  moment_names_.clear();
  accumulated_moments_.clear();
  for (unsigned int ell = 1; ell <= ell_max_; ++ell)
  {
    const auto signed_ell = static_cast<int>(ell);
    for (int m = -signed_ell; m <= signed_ell; ++m)
    {
      moments_.push_back({ell, m});
      moment_names_.push_back(std::to_string(ell) + "," + std::to_string(m));
      accumulated_moments_.emplace_back(num_loc_unknowns, 0.0);
    }
  }

  // The serial file stores one contiguous nodal segment per cell. Persist the
  // cell sizes so a later parallel solve can reconstruct offsets without
  // requiring access to nonlocal cells.
  std::vector<uint64_t> global_ids(num_loc_cells);
  std::vector<uint64_t> cell_node_counts(num_loc_cells);
  std::vector<double> nodes_x;
  std::vector<double> nodes_y;
  std::vector<double> nodes_z;
  std::vector<double> cell_sigma_t(num_loc_cells * num_groups_);
  nodes_x.reserve(num_loc_nodes);
  nodes_y.reserve(num_loc_nodes);
  nodes_z.reserve(num_loc_nodes);
  for (const auto& cell : grid_->GetLocalCells())
  {
    global_ids[cell->local_id] = cell->global_id;
    cell_node_counts[cell->local_id] = sdm.GetCellNumNodes(*cell);
    for (const auto vertex_id : cell->vertex_ids)
    {
      const auto& vertex = grid_->GlobalVertex(vertex_id);
      nodes_x.push_back(vertex.x);
      nodes_y.push_back(vertex.y);
      nodes_z.push_back(vertex.z);
    }

    const auto& sigma_t = cell_transport_views_[cell->local_id].GetXS().GetSigmaTotal();
    for (size_t g = 0; g < num_groups_; ++g)
      cell_sigma_t[static_cast<size_t>(cell->local_id) * num_groups_ + g] = sigma_t[g];
  }
  // The file stores the cells of all ranks, rank after rank.
  const auto all_global_ids = GatherOnRoot(global_ids);
  const auto all_cell_node_counts = GatherOnRoot(cell_node_counts);
  const auto all_nodes_x = GatherOnRoot(nodes_x);
  const auto all_nodes_y = GatherOnRoot(nodes_y);
  const auto all_nodes_z = GatherOnRoot(nodes_z);
  const auto all_cell_sigma_t = GatherOnRoot(cell_sigma_t);
  const uint64_t global_cell_count = grid_->GetGlobalNumberOfCells();
  RunOnRoot(
    [&]()
    {
      OpenSnLogicalErrorIf(not H5WriteDataset1D<uint64_t>(file, "cell ids", all_global_ids),
                           GetName() + ": failed to write cell ids.");
      OpenSnLogicalErrorIf(
        not H5WriteDataset1D<uint64_t>(file, "cell node counts", all_cell_node_counts),
        GetName() + ": failed to write cell node counts.");
      OpenSnLogicalErrorIf(not H5WriteDataset1D<double>(file, "nodes x", all_nodes_x),
                           GetName() + ": failed to write node x coordinates.");
      OpenSnLogicalErrorIf(not H5WriteDataset1D<double>(file, "nodes y", all_nodes_y),
                           GetName() + ": failed to write node y coordinates.");
      OpenSnLogicalErrorIf(not H5WriteDataset1D<double>(file, "nodes z", all_nodes_z),
                           GetName() + ": failed to write node z coordinates.");
      OpenSnLogicalErrorIf(not H5WriteDataset1D<double>(file, "cell sigma_t", all_cell_sigma_t),
                           GetName() + ": failed to write total cross sections.");
      OpenSnLogicalErrorIf(not H5CreateAttribute<unsigned int>(file, "num groups", num_groups_),
                           GetName() + ": failed to write group count.");
      OpenSnLogicalErrorIf(not H5CreateAttribute<unsigned int>(file, "max moment order", ell_max_),
                           GetName() + ": failed to write maximum moment order.");
      OpenSnLogicalErrorIf(
        not H5CreateAttribute<uint64_t>(file, "global cell count", global_cell_count),
        GetName() + ": failed to write global cell count.");
      OpenSnLogicalErrorIf(not H5CreateAttribute<uint64_t>(
                             file, "reflecting boundary count", reflecting_boundary_ids_.size()),
                           GetName() + ": failed to write reflecting boundary count.");
      if (not reflecting_boundary_ids_.empty())
      {
        const std::vector<uint64_t> reflecting_boundary_ids(reflecting_boundary_ids_.begin(),
                                                            reflecting_boundary_ids_.end());
        OpenSnLogicalErrorIf(
          not H5WriteDataset1D<uint64_t>(file, "reflecting boundary ids", reflecting_boundary_ids),
          GetName() + ": failed to write reflecting boundary ids.");
      }
    });

  const size_t num_source_points = source_points_.size();
  Timer progress_timer;
  unsigned int next_progress_percent = progress_interval;
  if (progress_interval > 0)
    log.Log() << "Starting uncollided transport for " << num_source_points
              << " physical source points. Progress will be reported every " << progress_interval
              << "%.";

  // Physical sources use the near-source ray trace and bulk sweep.
  for (size_t source_index = 0; source_index < num_source_points; ++source_index)
  {
    const auto& source_point = source_points_[source_index];
    if (progress_interval > 0 and source_index == 0 and num_source_points > 1)
      log.Log() << "Uncollided progress: processing source point 1 / " << num_source_points << ".";

    const auto& pt_loc = source_point.location;

    // Initialize uncollided flux and moment vector
    destination_phi_.assign(num_loc_unknowns, 0.);
    phase_timer.Reset();
    BuildSweepOrdering(source_point);
    timings_.sweep_ordering += phase_timer.GetTime() / 1000.0;

    RaytraceNearSourceRegion(source_point);
    phase_timer.Reset();
    if (not bulk_spls_.empty())
      SweepBulkRegion(source_point);
    timings_.bulk_sweep += phase_timer.GetTime() / 1000.0;

    // Update phi_new_local_
    for (const auto& cell : grid_->GetLocalCells())
    {
      const auto& cell_mapping = sdm.GetCellMapping(*cell);
      const size_t cell_num_nodes = cell_mapping.GetNumNodes();

      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        const auto ir = sdm.MapDOFLocal(*cell, i);

        for (size_t g = 0; g < num_groups_; ++g)
        {
          phi_new_local_[ir * num_groups_ + g] += destination_phi_[ir * num_groups_ + g];
        }
      }
    }
    phase_timer.Reset();
    UpdateBalance(source_point);

    AccumulateMoments(pt_loc);
    timings_.moments += phase_timer.GetTime() / 1000.0;

    if (progress_interval > 0)
    {
      const size_t completed = source_index + 1;
      const double completed_percent =
        100.0 * static_cast<double>(completed) / static_cast<double>(num_source_points);
      if (completed == num_source_points or completed_percent >= next_progress_percent)
      {
        const double elapsed_seconds = progress_timer.GetTime() / 1000.0;
        const double remaining_seconds = elapsed_seconds *
                                         static_cast<double>(num_source_points - completed) /
                                         static_cast<double>(completed);
        log.Log() << std::fixed << std::setprecision(1) << "Uncollided progress: " << completed
                  << " / " << num_source_points << " source points (" << completed_percent
                  << "%), elapsed " << FormatDuration(elapsed_seconds) << ", ETA "
                  << FormatDuration(remaining_seconds) << ".";

        while (next_progress_percent <= completed_percent and next_progress_percent < 100)
          next_progress_percent = std::min(100U, next_progress_percent + progress_interval);
      }
    }
  }

  phase_timer.Reset();
  if (not reflected_source_points_.empty())
    ProjectReflectedImageSources(progress_interval);
  timings_.image_sources = phase_timer.GetTime() / 1000.0;

  if (ray_tracer_type_ == RayTracerType::MATERIAL_HULL)
    log.Log() << "Material-hull ray tracing: " << hull_retraced_segments_.load()
              << " paths were traced again from a displaced start point.";
  else if (cell_walk_edge_paths_ > 0 or cell_walk_nudged_paths_ > 0)
  {
    log.Log() << "Cell-walk ray tracing: " << cell_walk_edge_paths_.load()
              << " paths passed through an edge or a vertex of the mesh and were walked again "
                 "with the edge-consistent triangle test.";
    if (cell_walk_nudged_paths_ > 0)
      log.Log0Warning() << GetName() << ": " << cell_walk_nudged_paths_.load()
                        << " ray-traced paths found no exit from a cell and were continued "
                           "from a displaced point. Their optical depth is approximate.";
  }

  phase_timer.Reset();
  WriteFluxMoments(file);

  // Finalize balance calculation
  FinalizeBalance(file);
  if (write_diagnostics_)
    WriteDiagnostics(diagnostics_file_name);
  timings_.output = phase_timer.GetTime() / 1000.0;

  // Wall time of rank 0 in each phase. The phases are collective, so the other ranks spend
  // about the same time in them, part of it waiting.
  log.Log() << std::fixed << std::setprecision(3) << "Uncollided generation time by phase (s):\n"
            << " Material hull            = " << timings_.material_hull << "\n"
            << " Cell ordering            = " << timings_.sweep_ordering << "\n"
            << " Ray tracing              = " << timings_.ray_tracing << "\n"
            << " Conservation correction  = " << timings_.conservation << "\n"
            << " Bulk sweep               = " << timings_.bulk_sweep << "\n"
            << " Image sources            = " << timings_.image_sources << "\n"
            << " Balance and moments      = " << timings_.moments << "\n"
            << " Output                   = " << timings_.output << "\n";
}
// NOLINTEND(clang-analyzer-security.ArrayBound)

void
UncollidedProblem::RaytraceLineInto(RayTracer& ray_tracer,
                                    const Cell& cell,
                                    const Vector3& qp_xyz,
                                    const SourcePoint& source_point,
                                    std::vector<double>& phi_out,
                                    std::vector<std::pair<size_t, double>>& scratch_segs,
                                    std::vector<double>& scratch_bp,
                                    std::vector<double>& scratch_mfp,
                                    const int face_index,
                                    const double tolerance)
{
  const auto& pt_loc = source_point.location;
  const auto& strength = source_point.strength;

  const double total_length = (pt_loc - qp_xyz).Norm();
  OpenSnLogicalErrorIf(total_length < 1.0e-15,
                       GetName() + ": point source lies at cell quadrature point.");

  scratch_bp.clear();
  scratch_bp.push_back(0.0);
  scratch_bp.push_back(1.0);
  const auto unfolded_direction = pt_loc - qp_xyz;
  for (const auto& plane : source_point.reflection_planes)
  {
    const double denominator = plane.normal.Dot(unfolded_direction);
    OpenSnLogicalErrorIf(std::abs(denominator) <= tolerance,
                         GetName() + ": image-source ray is parallel to its reflecting plane.");
    const double t = (plane.offset - plane.normal.Dot(qp_xyz)) / denominator;
    OpenSnLogicalErrorIf(t < -tolerance or t > 1.0 + tolerance,
                         GetName() + ": image-source ray does not intersect its reflecting plane.");
    if (t > tolerance and t < 1.0 - tolerance)
      scratch_bp.push_back(t);
  }
  std::sort(scratch_bp.begin(), scratch_bp.end());

  if (ray_tracer_type_ == RayTracerType::MATERIAL_HULL)
    ComputeHullOpticalDepth(
      cell, qp_xyz, source_point, scratch_bp, face_index, scratch_mfp, tolerance);
  else
    ComputeCellWalkOpticalDepth(
      ray_tracer, cell, qp_xyz, source_point, scratch_segs, scratch_bp, scratch_mfp, tolerance);

  for (size_t g = 0; g < num_groups_; ++g)
    phi_out[g] =
      strength[g] / (4.0 * M_PI * total_length * total_length) * std::exp(-scratch_mfp[g]);
}

std::pair<Vector3, Vector3>
UncollidedProblem::FoldSegment(const Vector3& point,
                               const Vector3& unfolded_direction,
                               const double begin,
                               const double end,
                               const std::vector<ReflectionPlane>& planes)
{
  auto ReflectPoint = [](const Vector3& point, const ReflectionPlane& plane)
  { return point - 2.0 * (plane.normal.Dot(point) - plane.offset) * plane.normal; };

  Vector3 segment_start = point + begin * unfolded_direction;
  Vector3 segment_end = point + end * unfolded_direction;
  const Vector3 midpoint = 0.5 * (segment_start + segment_end);
  for (const auto& plane : planes)
    if (plane.normal.Dot(midpoint) > plane.offset)
    {
      segment_start = ReflectPoint(segment_start, plane);
      segment_end = ReflectPoint(segment_end, plane);
    }
  return {segment_start, segment_end};
}

void
UncollidedProblem::ComputeCellWalkOpticalDepth(RayTracer& ray_tracer,
                                               const Cell& cell,
                                               const Vector3& qp_xyz,
                                               const SourcePoint& source_point,
                                               std::vector<std::pair<size_t, double>>& scratch_segs,
                                               const std::vector<double>& breakpoints,
                                               std::vector<double>& optical_depth,
                                               const double tolerance)
{
  const auto& pt_loc = source_point.location;
  const auto unfolded_direction = pt_loc - qp_xyz;

  scratch_segs.clear();
  size_t cell_id = cell.local_id;
  for (size_t interval = 0; interval + 1 < breakpoints.size(); ++interval)
  {
    const auto [segment_start, segment_end] = FoldSegment(qp_xyz,
                                                          unfolded_direction,
                                                          breakpoints[interval],
                                                          breakpoints[interval + 1],
                                                          source_point.reflection_planes);

    double remaining_distance = (segment_end - segment_start).Norm();
    if (remaining_distance <= tolerance)
      continue;

    Vector3 segment_omega = (segment_end - segment_start).Normalized();
    if (interval > 0 and all_cells_convex_)
    {
      // A fold can land where a reflecting boundary meets an internal face or edge. The reflected
      // segment may then enter a different incident cell than the segment that reached the fold.
      const double probe_distance =
        std::min(0.5 * remaining_distance, cell_sizes_[cell_id] * 1.0e-8);
      const Vector3 probe = segment_start + probe_distance * segment_omega;
      const auto contains_probe = [this, &probe](const Cell& candidate)
      {
        const double point_tolerance = cell_sizes_[candidate.local_id] * 1.0e-10;
        return std::all_of(candidate.faces.begin(),
                           candidate.faces.end(),
                           [&probe, point_tolerance](const CellFace& face)
                           { return face.normal.Dot(probe - face.centroid) <= point_tolerance; });
      };
      if (not contains_probe(grid_->GetLocalCell(cell_id)))
      {
        const auto containing_cell = std::find_if(grid_->GetLocalCells().begin(),
                                                  grid_->GetLocalCells().end(),
                                                  [&contains_probe](const auto& candidate)
                                                  { return contains_probe(*candidate); });
        OpenSnLogicalErrorIf(containing_cell == grid_->GetLocalCells().end(),
                             GetName() +
                               ": reflected image-source ray could not locate its re-entry cell.");
        cell_id = (*containing_cell)->local_id;
      }
    }
    const size_t piece_start_cell = cell_id;
    const size_t piece_first_segment = scratch_segs.size();
    Vector3 line_point = segment_start;
    // The last piece ends at the source.
    const bool ends_at_source = interval + 2 == breakpoints.size();
    while (remaining_distance > tolerance)
    {
      // The rest of the path lies in a convex cell that contains the source. Stopping here also
      // covers a source on a vertex of the cell, where no exit face can be found.
      if (ends_at_source and all_cells_convex_ and
          std::any_of(source_point.subscribers.begin(),
                      source_point.subscribers.end(),
                      [cell_id](const auto& subscriber)
                      { return subscriber.cell_local_id == cell_id; }))
      {
        scratch_segs.emplace_back(cell_id, remaining_distance);
        break;
      }

      double dist_in_cell = NAN;
      Vector3 exit_point;
      uint64_t dest_neighbor = 0;
      bool destination_has_neighbor = false;
      uint32_t destination_face_index = 0;

      if (use_fast_trace_)
      {
        // Fast path: precomputed flat face vertex data -- no std::map traversal.
        const double cell_size = cell_sizes_[cell_id];
        const double back_tol = cell_size * 1.0e-10;
        const uint32_t face_off = cell_face_offsets_[cell_id];
        const uint32_t n_faces = cell_num_faces_[cell_id];

        bool found = false;
        dist_in_cell = std::numeric_limits<double>::max();
        for (uint32_t f = 0; f < n_faces; ++f)
        {
          const auto& fv = all_face_verts_[face_off + f];
          const auto& v2 = fv.centroid;
          const uint32_t ns = fv.num_sides;
          for (uint32_t s = 0; s < ns; ++s)
          {
            const auto& v0 = fv.verts[s];
            const auto& v1 = fv.verts[s + 1 < ns ? s + 1 : 0];
            if ((v1 - v0).Cross(v2 - v0).Dot(segment_omega) < 0.0)
              continue;
            Vector3 candidate;
            if (CheckLineIntersectTriangle2(v0, v1, v2, line_point, segment_omega, candidate))
            {
              const double d = (candidate - line_point).Norm();
              if (d > back_tol and d < dist_in_cell)
              {
                exit_point = candidate;
                dist_in_cell = d;
                dest_neighbor = fv.neighbor_id;
                destination_has_neighbor = grid_->GetLocalCell(cell_id).faces[f].has_neighbor;
                destination_face_index = f;
                found = true;
              }
            }
          }
        }

        if (not found)
        {
          // No exit was found: the path passes through an edge or a vertex of the mesh, where
          // the triangle test above can reject every triangle around the edge, or accept one
          // that leads to a cell the path only touches. Walk this piece of the path again from
          // its start with a triangle test that resolves edges and vertices consistently.
          thread_local std::vector<std::pair<size_t, double>> piece_segs;
          size_t piece_end_cell = piece_start_cell;
          if (WalkSegmentWatertight(piece_start_cell,
                                    segment_start,
                                    segment_end,
                                    piece_segs,
                                    piece_end_cell,
                                    tolerance))
          {
            ++cell_walk_edge_paths_;
            scratch_segs.resize(piece_first_segment);
            scratch_segs.insert(scratch_segs.end(), piece_segs.begin(), piece_segs.end());
            cell_id = piece_end_cell;
            break;
          }

          ++cell_walk_nudged_paths_;
          // Rare fallback for degenerate cases (point on a face/vertex): let TraceRay nudge.
          auto trace_point = line_point;
          auto trace_direction = segment_omega;
          const auto oi =
            ray_tracer.TraceRay(grid_->GetLocalCell(cell_id), trace_point, trace_direction);
          OpenSnLogicalErrorIf(oi.particle_lost,
                               GetName() +
                                 ": reflected image-source ray lost in fast-trace fallback.");
          dist_in_cell = (oi.pos_f - line_point).Dot(segment_omega);
          exit_point = oi.pos_f;
          dest_neighbor = oi.destination_face_neighbor;
          destination_has_neighbor =
            grid_->GetLocalCell(cell_id).faces[oi.destination_face_index].has_neighbor;
          destination_face_index = oi.destination_face_index;
        }
      }
      else
      {
        // Original path: used for non-convex meshes or faces with >max_sides vertices.
        auto trace_point = line_point;
        auto trace_direction = segment_omega;
        const auto oi =
          ray_tracer.TraceRay(grid_->GetLocalCell(cell_id), trace_point, trace_direction);
        OpenSnLogicalErrorIf(oi.particle_lost,
                             GetName() + ": ray lost in mesh during segment traversal.");
        dist_in_cell = (oi.pos_f - line_point).Dot(segment_omega);
        exit_point = oi.pos_f;
        dest_neighbor = oi.destination_face_neighbor;
        destination_has_neighbor =
          grid_->GetLocalCell(cell_id).faces[oi.destination_face_index].has_neighbor;
        destination_face_index = oi.destination_face_index;
      }

      const double endpoint_tolerance = std::max(tolerance, cell_sizes_[cell_id] * 1.0e-10);
      const auto& destination_face = grid_->GetLocalCell(cell_id).faces[destination_face_index];
      if (not destination_has_neighbor)
      {
        // TraceRay can nudge its working origin when a ray lies on a face or edge. Recompute a
        // boundary hit from the unchanged analytic ray and the boundary-face plane so the nudge
        // cannot shorten a segment ending on a reflecting plane.
        const double denominator = destination_face.normal.Dot(segment_omega);
        if (std::abs(denominator) > endpoint_tolerance)
        {
          const double boundary_distance =
            destination_face.normal.Dot(destination_face.centroid - line_point) / denominator;
          if (boundary_distance > endpoint_tolerance)
            dist_in_cell = boundary_distance;
        }
      }

      // Keep the traversal on the analytic segment. Reusing an intersection point computed from
      // a face triangulation can accumulate transverse roundoff over many polyhedral cells.
      exit_point = line_point + dist_in_cell * segment_omega;
      const double endpoint_error = (exit_point - segment_end).Norm();
      const bool endpoint_on_destination_face =
        not destination_has_neighbor and
        std::abs(destination_face.normal.Dot(segment_end - destination_face.centroid)) <=
          endpoint_tolerance;
      const bool reaches_segment_end = remaining_distance <= dist_in_cell + endpoint_tolerance or
                                       endpoint_error <= endpoint_tolerance or
                                       endpoint_on_destination_face;
      OpenSnLogicalErrorIf(
        not destination_has_neighbor and not reaches_segment_end,
        GetName() + ": reflected image-source ray left the mesh " + std::to_string(endpoint_error) +
          " from its segment endpoint. target=" + qp_xyz.PrintStr() +
          ", image=" + pt_loc.PrintStr() + ", segment_start=" + segment_start.PrintStr() +
          ", segment_end=" + segment_end.PrintStr() + ", hit=" + exit_point.PrintStr() +
          ", cell=" + std::to_string(cell_id) + ".");
      const double distance_in_cell = reaches_segment_end ? remaining_distance : dist_in_cell;
      OpenSnLogicalErrorIf(distance_in_cell <= tolerance,
                           GetName() + ": reflected image-source ray failed to advance.");
      scratch_segs.emplace_back(cell_id, distance_in_cell);
      remaining_distance -= distance_in_cell;
      if (reaches_segment_end or remaining_distance <= tolerance)
        break;

      if (use_fast_trace_)
      {
        // O(1) unordered_map lookup replaces two O(log N) std::map calls.
        auto it = global_to_local_id_.find(dest_neighbor);
        OpenSnLogicalErrorIf(
          it == global_to_local_id_.end(),
          GetName() + ": reflected image-source ray left the mesh before reaching the source.");
        cell_id = it->second;
      }
      else
      {
        OpenSnLogicalErrorIf(
          not grid_->IsCellLocal(dest_neighbor),
          GetName() + ": reflected image-source ray left the mesh before reaching the source.");
        cell_id = grid_->GetGlobalCell(dest_neighbor).local_id;
      }
      line_point = exit_point;
    }
  }

  optical_depth.assign(num_groups_, 0.0);
  for (const auto& segment : scratch_segs)
  {
    const auto& sigma_t = cell_transport_views_[segment.first].GetXS().GetSigmaTotal();
    for (size_t g = 0; g < num_groups_; ++g)
      optical_depth[g] += sigma_t[g] * segment.second;
  }
}
bool
UncollidedProblem::WalkSegmentWatertight(const size_t start_cell,
                                         const Vector3& segment_start,
                                         const Vector3& segment_end,
                                         std::vector<std::pair<size_t, double>>& segments,
                                         size_t& end_cell,
                                         const double tolerance) const
{
  segments.clear();
  const HullSegmentFrame frame({segment_start, segment_end});
  if (not frame.valid)
    return false;
  const double length = (segment_end - segment_start).Norm();

  size_t cell_id = start_cell;
  // Fraction of the segment at which it entered the current cell.
  double entered = 0.0;
  // A segment visits a convex cell once, and the cells it only touches for a zero length.
  const size_t max_steps = grid_->GetLocalCellCount() + 1;
  for (size_t step = 0; step < max_steps; ++step)
  {
    // The segment leaves a convex cell through the one triangle it crosses outward. The side
    // triangles (v_s, v_s+1, face centroid) have the outward normal of the cell.
    const uint32_t face_offset = cell_face_offsets_[cell_id];
    const uint32_t num_faces = cell_num_faces_[cell_id];
    bool leaves = false;
    double exit_fraction = std::numeric_limits<double>::max();
    uint32_t exit_face = 0;
    for (uint32_t f = 0; f < num_faces; ++f)
    {
      const auto& face = all_face_verts_[face_offset + f];
      for (uint32_t side = 0; side < face.num_sides; ++side)
      {
        const std::array<Vector3, 3> triangle = {
          face.verts[side], face.verts[side + 1 < face.num_sides ? side + 1 : 0], face.centroid};
        double fraction = 0.0;
        bool outward = false;
        if (IntersectSegmentTriangle(frame, triangle, fraction, outward) and outward and
            fraction < exit_fraction)
        {
          leaves = true;
          exit_fraction = fraction;
          exit_face = f;
        }
      }
    }

    const Cell& cell = grid_->GetLocalCell(cell_id);
    const double end_tolerance = std::max(tolerance, cell_sizes_[cell_id] * 1.0e-10);
    if (not leaves)
    {
      // The segment ends in this cell. Check it, so that a wrong turn at an edge cannot assign
      // the rest of the segment to a cell that does not contain its end.
      const double point_tolerance = cell_sizes_[cell_id] * 1.0e-8;
      for (const auto& face : cell.faces)
        if (face.normal.Dot(segment_end - face.centroid) > point_tolerance)
          return false;
      segments.emplace_back(cell_id, (1.0 - entered) * length);
      end_cell = cell_id;
      return true;
    }

    segments.emplace_back(cell_id, std::max(0.0, exit_fraction - entered) * length);
    entered = std::max(entered, exit_fraction);
    if (not cell.faces[exit_face].has_neighbor)
    {
      // The segment reaches the boundary: this is its end, on a boundary face, or it leaves the
      // mesh.
      end_cell = cell_id;
      return (1.0 - entered) * length <= end_tolerance;
    }
    const auto neighbor =
      global_to_local_id_.find(all_face_verts_[face_offset + exit_face].neighbor_id);
    if (neighbor == global_to_local_id_.end())
      return false;
    cell_id = neighbor->second;
  }
  return false;
}

void
UncollidedProblem::BuildMaterialHull()
{
  CALI_CXX_MARK_SCOPE("BuildMaterialHull");

  Timer timer;
  hull_ = MaterialHull::Extract(*grid_);
  hull_tracer_ = SegmentTracer::Create(hull_tracer_backend_, hull_);

  // Total cross section of each hull material; the exterior, the last material, is a void.
  const std::uint32_t num_materials = hull_->GetNumMaterials();
  hull_sigma_t_.assign(static_cast<size_t>(num_materials) + 1, std::vector<double>(num_groups_));
  for (std::uint32_t m = 0; m < num_materials; ++m)
    hull_sigma_t_[m] = GetBlockID2XSMap().at(hull_->GetBlockIDs()[m])->GetSigmaTotal();

  cell_hull_materials_.resize(grid_->GetLocalCellCount());
  for (const auto& cell : grid_->GetLocalCells())
    cell_hull_materials_[cell->local_id] = hull_->GetMaterialIndex(cell->block_id);

  // A path that is folded at a reflecting boundary touches that boundary but does not cross it.
  std::vector<std::uint32_t> reflecting_triangles;
  const auto& triangles = hull_->GetTriangles();
  for (size_t t = 0; t < triangles.size(); ++t)
    if (triangles[t].front_material == hull_->GetExteriorIndex() and
        IsReflectingBoundary(triangles[t].boundary_id))
      reflecting_triangles.push_back(static_cast<std::uint32_t>(t));

  for (size_t i = 0; i < source_points_.size(); ++i)
  {
    auto& source_point = source_points_[i];
    try
    {
      source_point.hull_excluded_triangles = hull_->FindEndPointExclusions(source_point.location);
    }
    catch (const std::invalid_argument&)
    {
      OpenSnInvalidArgument(GetName() + ": point source " + std::to_string(i) + " at " +
                            source_point.location.PrintStr() +
                            " lies on an interface between two materials, which the "
                            "material-hull ray tracer does not support.");
    }

    // The cells of a source can be on any rank. They must all be of one material.
    unsigned int lowest_material = std::numeric_limits<unsigned int>::max();
    unsigned int highest_material = 0;
    for (const auto& subscriber : source_point.subscribers)
    {
      lowest_material = std::min(lowest_material, cell_hull_materials_[subscriber.cell_local_id]);
      highest_material = std::max(highest_material, cell_hull_materials_[subscriber.cell_local_id]);
    }
    if (opensn::mpi_comm.size() > 1)
    {
      opensn::mpi_comm.all_reduce(lowest_material, mpi::op::min<unsigned int>());
      opensn::mpi_comm.all_reduce(highest_material, mpi::op::max<unsigned int>());
    }
    OpenSnLogicalErrorIf(lowest_material > highest_material,
                         GetName() + ": point source " + std::to_string(i) + " has no cell.");
    OpenSnInvalidArgumentIf(lowest_material != highest_material,
                            GetName() + ": point source " + std::to_string(i) + " at " +
                              source_point.location.PrintStr() +
                              " lies on an interface between two materials, which the "
                              "material-hull ray tracer does not support.");
    source_point.hull_material = lowest_material;
  }

  // The folded path of an image source ends at its physical source. The images of a physical
  // source follow one another in reflected_source_points_.
  if (not source_points_.empty())
  {
    const size_t num_images_per_source = reflected_source_points_.size() / source_points_.size();
    for (size_t s = 0; s < source_points_.size(); ++s)
      for (size_t k = 0; k < num_images_per_source; ++k)
      {
        const auto& physical_source = source_points_[s];
        auto& image_source = reflected_source_points_[s * num_images_per_source + k];
        image_source.hull_material = physical_source.hull_material;
        image_source.hull_excluded_triangles.clear();
        std::set_union(physical_source.hull_excluded_triangles.begin(),
                       physical_source.hull_excluded_triangles.end(),
                       reflecting_triangles.begin(),
                       reflecting_triangles.end(),
                       std::back_inserter(image_source.hull_excluded_triangles));
      }
  }

  hull_retraced_segments_ = 0;
  log.Log() << "Material hull: " << triangles.size() << " triangles between " << num_materials
            << " materials and the exterior, " << hull_tracer_backend_ << " tracer, built in "
            << std::fixed << std::setprecision(3) << timer.GetTime() / 1000.0 << " s.";
}

void
UncollidedProblem::ComputeHullOpticalDepth(const Cell& cell,
                                           const Vector3& qp_xyz,
                                           const SourcePoint& source_point,
                                           const std::vector<double>& breakpoints,
                                           const int face_index,
                                           std::vector<double>& optical_depth,
                                           const double tolerance)
{
  // Displacements of the start point, relative to its distance to the cell centroid, tried in
  // turn when the crossings of a segment are inconsistent. The first entry is the point itself.
  // The others move it into the cell, towards the centroid and slightly sideways so that it also
  // leaves any plane of symmetry it shares with the centroid.
  static const std::array<Vector3, 5> displacements = {Vector3(0.0, 0.0, 0.0),
                                                       Vector3(0.618, 0.382, 0.687),
                                                       Vector3(-0.451, 0.774, 0.444),
                                                       Vector3(0.577, -0.613, 0.540),
                                                       Vector3(-0.734, -0.291, 0.613)};

  thread_local std::vector<HullSegment> pieces;
  thread_local HullCrossings crossings;
  thread_local std::vector<double> lengths;
  thread_local std::vector<double> piece_lengths;

  const std::uint32_t exterior = hull_->GetExteriorIndex();
  const std::uint32_t num_materials = exterior + 1;
  const std::uint32_t cell_material = cell_hull_materials_[cell.local_id];
  const Vector3 to_centroid = cell.centroid - qp_xyz;
  const Vector3 unfolded_direction = source_point.location - qp_xyz;

  bool consistent = false;
  double total_length = 0.0;
  for (size_t attempt = 0; attempt < displacements.size() and not consistent; ++attempt)
  {
    // The pieces of the path between the reflecting planes, each folded back into the domain.
    // A displaced start point changes the first piece only.
    pieces.clear();
    for (size_t interval = 0; interval + 1 < breakpoints.size(); ++interval)
    {
      auto [piece_start, piece_end] = FoldSegment(qp_xyz,
                                                  unfolded_direction,
                                                  breakpoints[interval],
                                                  breakpoints[interval + 1],
                                                  source_point.reflection_planes);
      if ((piece_end - piece_start).Norm() <= tolerance)
        continue;
      if (pieces.empty() and attempt > 0)
        piece_start = piece_start + static_cast<double>(attempt) * 1.0e-10 * to_centroid +
                      1.0e-11 * to_centroid.Norm() * displacements[attempt];
      pieces.push_back({piece_start, piece_end});
    }
    if (attempt > 0)
      ++hull_retraced_segments_;

    hull_tracer_->Trace(pieces, source_point.hull_excluded_triangles, crossings);

    // The material at the end of the last piece is that of the source. The material at the
    // start of a piece is the material at the end of the piece before it.
    lengths.assign(num_materials, 0.0);
    piece_lengths.resize(num_materials);
    std::uint32_t material = source_point.hull_material;
    total_length = 0.0;
    for (size_t p = pieces.size(); p-- > 0 and material != SegmentTracer::INVALID_MATERIAL;)
    {
      const double length = (pieces[p].end - pieces[p].start).Norm();
      material = ComputeMaterialLengths(crossings.Begin(p),
                                        crossings.End(p),
                                        length,
                                        material,
                                        num_materials,
                                        piece_lengths.data());
      for (std::uint32_t m = 0; m < num_materials; ++m)
        lengths[m] += piece_lengths[m];
      total_length += length;
    }

    // The path must start in the material of the cell. A point on a face of the cell may also
    // be found on the far side of that face.
    consistent = material == cell_material;
    if (not consistent and material != SegmentTracer::INVALID_MATERIAL and face_index >= 0)
    {
      const auto& face = cell.faces[face_index];
      const std::uint32_t far_material =
        face.has_neighbor ? hull_->GetMaterialIndex(grid_->GetGlobalCell(face.neighbor_id).block_id)
                          : exterior;
      consistent = material == far_material;
    }
  }
  OpenSnLogicalErrorIf(not consistent,
                       GetName() +
                         ": the material-hull ray tracer found inconsistent material "
                         "crossings between the point " +
                         qp_xyz.PrintStr() + " of cell " + std::to_string(cell.global_id) +
                         " and the source at " + source_point.location.PrintStr() + ".");
  OpenSnLogicalErrorIf(lengths[exterior] > 1.0e-9 * total_length,
                       GetName() + ": the path between the point " + qp_xyz.PrintStr() +
                         " of cell " + std::to_string(cell.global_id) + " and the source at " +
                         source_point.location.PrintStr() +
                         " leaves the mesh. Uncollided generation requires that a point "
                         "source sees every cell through the mesh.");

  optical_depth.assign(num_groups_, 0.0);
  for (std::uint32_t m = 0; m < exterior; ++m)
    if (lengths[m] != 0.0)
      for (size_t g = 0; g < num_groups_; ++g)
        optical_depth[g] += hull_sigma_t_[m][g] * lengths[m];
}

void
UncollidedProblem::ProjectReflectedImageSources(const unsigned int progress_interval)
{
  CALI_CXX_MARK_SCOPE("ProjectReflectedImageSources");

  const auto& sdm = *discretization_;
  const size_t num_cells = grid_->GetLocalCellCount();
  const size_t num_threads =
    std::min<size_t>(std::max(1U, opensn_num_threads), std::max<size_t>(1, num_cells));
  const size_t num_moments = accumulated_moments_.size();
  std::vector<double> thread_removals(num_threads, 0.0);
  std::vector<double> thread_outflows(num_threads, 0.0);
  std::atomic<size_t> completed_cells = 0;
  std::atomic<unsigned int> next_progress_percent = progress_interval;
  std::exception_ptr worker_exception;
  std::mutex exception_mutex;
  Timer progress_timer;

  log.Log() << "Directly projecting " << reflected_source_points_.size()
            << " reflected image sources over " << num_cells << " cells with " << num_threads
            << " threads.";

  auto ProjectCells = [&](const size_t thread_id)
  {
    try
    {
      RayTracer ray_tracer(grid_, &cell_sizes_, not all_cells_convex_);
      double local_removal = 0.0;
      double local_outflow = 0.0;
      // Pre-allocate scratch buffers once per thread to avoid heap allocation in the hot loop.
      std::vector<double> phi_qp(num_groups_, 0.0);
      std::vector<std::pair<size_t, double>> scratch_segs;
      std::vector<double> scratch_bp;
      std::vector<double> scratch_mfp(num_groups_, 0.0);
      scratch_segs.reserve(64);
      scratch_bp.reserve(8);
      for (size_t cell_index = thread_id; cell_index < num_cells; cell_index += num_threads)
      {
        const auto& cell = grid_->GetLocalCell(cell_index);
        const auto& cell_mapping = sdm.GetCellMapping(cell);
        const size_t cell_num_nodes = cell_mapping.GetNumNodes();
        const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();
        const auto& unit_matrices = unit_cell_matrices_[cell.local_id];
        const auto& intV_shapeI = unit_matrices.intV_shapeI;
        const auto& sigma_t = cell_transport_views_[cell.local_id].GetXS().GetSigmaTotal();
        std::vector<Vector<double>> cell_phi(num_groups_, Vector<double>(cell_num_nodes, 0.0));
        std::vector<std::vector<Vector<double>>> moment_rhs(
          num_moments,
          std::vector<Vector<double>>(num_groups_, Vector<double>(cell_num_nodes, 0.0)));

        for (const auto& source_point : reflected_source_points_)
        {
          std::vector<Vector<double>> image_phi(num_groups_, Vector<double>(cell_num_nodes, 0.0));
          for (const auto& qp : fe_vol_data.GetQuadraturePointIndices())
          {
            const auto& qp_xyz = fe_vol_data.QPointXYZ(qp);
            RaytraceLineInto(ray_tracer,
                             cell,
                             qp_xyz,
                             source_point,
                             phi_qp,
                             scratch_segs,
                             scratch_bp,
                             scratch_mfp);
            for (size_t i = 0; i < cell_num_nodes; ++i)
            {
              const double integrand = fe_vol_data.ShapeValue(i, qp) * fe_vol_data.JxW(qp);
              for (size_t g = 0; g < num_groups_; ++g)
                image_phi[g](i) += phi_qp[g] * integrand;
            }
          }

          for (size_t g = 0; g < num_groups_; ++g)
          {
            auto mass_matrix = unit_matrices.intV_shapeI_shapeJ;
            GaussElimination(mass_matrix, image_phi[g], static_cast<int>(cell_num_nodes));

            double projected_integral = 0.0;
            for (size_t i = 0; i < cell_num_nodes; ++i)
              projected_integral += intV_shapeI(i) * image_phi[g](i);
            local_removal += sigma_t[g] * projected_integral;

            for (size_t i = 0; i < cell_num_nodes; ++i)
              cell_phi[g](i) += image_phi[g](i);
          }

          for (const auto& qp : fe_vol_data.GetQuadraturePointIndices())
          {
            const auto& qp_xyz = fe_vol_data.QPointXYZ(qp);
            const auto omega = ComputeOmega(source_point.location, qp_xyz);
            OpenSnLogicalErrorIf(
              omega.Norm() == 0.0,
              GetName() + ": reflected image source coincides with a volume quadrature point.");
            const double theta = std::acos(std::clamp(omega.z, -1.0, 1.0));
            const double varphi = std::atan2(omega.y, omega.x);
            std::fill(phi_qp.begin(), phi_qp.end(), 0.0);
            for (size_t j = 0; j < cell_num_nodes; ++j)
            {
              const double shape = fe_vol_data.ShapeValue(j, qp);
              for (size_t g = 0; g < num_groups_; ++g)
                phi_qp[g] += shape * image_phi[g](j);
            }

            for (size_t moment_index = 0; moment_index < moments_.size(); ++moment_index)
            {
              const auto& moment = moments_[moment_index];
              const double harmonic = Ylm(moment.ell, moment.m, varphi, theta);
              for (size_t i = 0; i < cell_num_nodes; ++i)
              {
                const double weight =
                  fe_vol_data.ShapeValue(i, qp) * harmonic * fe_vol_data.JxW(qp);
                for (size_t g = 0; g < num_groups_; ++g)
                  moment_rhs[moment_index][g](i) += phi_qp[g] * weight;
              }
            }
          }

          for (size_t f = 0; f < cell.faces.size(); ++f)
          {
            const auto& face = cell.faces[f];
            if (face.has_neighbor or IsReflectingBoundary(face.neighbor_id))
              continue;

            const auto face_omega = ComputeOmega(source_point.location, face.centroid);
            if (face_omega.Dot(face.normal) <= 1.0e-16)
              continue;

            const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);
            for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
            {
              const auto& qp_xyz = fe_srf_data.QPointXYZ(qp);
              const auto omega = ComputeOmega(source_point.location, qp_xyz);
              const double integrand = omega.Dot(face.normal) * fe_srf_data.JxW(qp);
              RaytraceLineInto(ray_tracer,
                               cell,
                               qp_xyz,
                               source_point,
                               phi_qp,
                               scratch_segs,
                               scratch_bp,
                               scratch_mfp,
                               static_cast<int>(f));
              for (size_t g = 0; g < num_groups_; ++g)
                local_outflow += phi_qp[g] * integrand;
            }
          }
        }

        for (size_t i = 0; i < cell_num_nodes; ++i)
        {
          const auto ir = sdm.MapDOFLocal(cell, i);
          for (size_t g = 0; g < num_groups_; ++g)
            phi_new_local_[ir * num_groups_ + g] += cell_phi[g](i);
        }

        for (size_t moment_index = 0; moment_index < num_moments; ++moment_index)
          for (size_t g = 0; g < num_groups_; ++g)
          {
            auto mass_matrix = unit_matrices.intV_shapeI_shapeJ;
            GaussElimination(
              mass_matrix, moment_rhs[moment_index][g], static_cast<int>(cell_num_nodes));
            for (size_t i = 0; i < cell_num_nodes; ++i)
            {
              const auto ir = sdm.MapDOFLocal(cell, i);
              accumulated_moments_[moment_index][ir * num_groups_ + g] +=
                moment_rhs[moment_index][g](i);
            }
          }

        const size_t completed = completed_cells.fetch_add(1, std::memory_order_relaxed) + 1;
        if (progress_interval > 0)
        {
          const auto completed_percent =
            static_cast<unsigned int>(100 * completed / std::max<size_t>(1, num_cells));
          auto next_percent = next_progress_percent.load(std::memory_order_relaxed);
          while (completed_percent >= next_percent and next_percent <= 100)
            if (next_progress_percent.compare_exchange_weak(
                  next_percent,
                  std::min(101U, next_percent + progress_interval),
                  std::memory_order_relaxed))
            {
              const double elapsed_seconds = progress_timer.GetTime() / 1000.0;
              const double remaining_seconds = elapsed_seconds *
                                               static_cast<double>(num_cells - completed) /
                                               static_cast<double>(completed);
              log.Log() << std::fixed << std::setprecision(1)
                        << "Reflected projection progress: " << completed << " / " << num_cells
                        << " cells (" << completed_percent << "%), elapsed "
                        << FormatDuration(elapsed_seconds) << ", ETA "
                        << FormatDuration(remaining_seconds) << ".";
              break;
            }
        }
      }
      thread_removals[thread_id] = local_removal;
      thread_outflows[thread_id] = local_outflow;
    }
    catch (...)
    {
      std::scoped_lock lock(exception_mutex);
      if (not worker_exception)
        worker_exception = std::current_exception();
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(num_threads > 0 ? num_threads - 1 : 0);
  for (size_t thread_id = 1; thread_id < num_threads; ++thread_id)
    workers.emplace_back(ProjectCells, thread_id);
  ProjectCells(0);
  for (auto& worker : workers)
    worker.join();
  if (worker_exception)
    std::rethrow_exception(worker_exception);

  reflected_removal_ += std::accumulate(thread_removals.begin(), thread_removals.end(), 0.0);
  out_flow_ += std::accumulate(thread_outflows.begin(), thread_outflows.end(), 0.0);
}

void
UncollidedProblem::CorrectCellBalance(const std::uint64_t cell_global_id,
                                      const std::vector<FaceOrientation>& orientations,
                                      const std::vector<bool>& vacuum_boundary_faces,
                                      const std::vector<double>& sigma_t,
                                      const std::vector<double>& projected_integral,
                                      std::vector<double>& source,
                                      std::vector<std::vector<double>>& leakage,
                                      std::vector<double>& alpha,
                                      std::vector<char>& leakage_fallback,
                                      ConservationSummary& summary,
                                      double& out_flow) const
{
  constexpr auto FOINCOMING = FaceOrientation::INCOMING;
  constexpr auto FOOUTGOING = FaceOrientation::OUTGOING;
  const size_t cell_num_faces = orientations.size();
  bool cell_current_mismatched = false;

  // Incoming leakage contributes to the available cell source.
  for (size_t f = 0; f < cell_num_faces; ++f)
    if (orientations[f] == FOINCOMING)
      for (size_t g = 0; g < num_groups_; ++g)
        source[g] += leakage[f][g];

  alpha.assign(num_groups_, 1.0);
  leakage_fallback.assign(num_groups_, 0);
  for (size_t g = 0; g < num_groups_; ++g)
  {
    double outgoing_leakage = 0.0;
    for (size_t f = 0; f < cell_num_faces; ++f)
      if (orientations[f] == FOOUTGOING)
        outgoing_leakage += leakage[f][g];

    const double raw_removal = outgoing_leakage + sigma_t[g] * projected_integral[g];
    const double removal_tolerance = 1.0e-12 * std::abs(source[g]);
    OpenSnLogicalErrorIf(not std::isfinite(raw_removal) or raw_removal < -removal_tolerance,
                         GetName() +
                           ": near-source ray tracing produced an invalid removal for "
                           "cell " +
                           std::to_string(cell_global_id) + ", group " + std::to_string(g) + ".");

    const bool use_leakage_fallback = raw_removal <= removal_tolerance and source[g] > 0.0;
    alpha[g] = use_leakage_fallback ? 1.0 : (raw_removal > 0.0 ? source[g] / raw_removal : 1.0);
    leakage_fallback[g] = use_leakage_fallback ? 1 : 0;
    OpenSnLogicalErrorIf(not std::isfinite(alpha[g]) or alpha[g] < 0.0,
                         GetName() +
                           ": near-source conservation produced an invalid scale "
                           "factor for cell " +
                           std::to_string(cell_global_id) + ", group " + std::to_string(g) + ".");

    const double relative_change = std::abs(alpha[g] - 1.0);
    constexpr double significant_relative_change = 0.01;
    if (relative_change > significant_relative_change)
    {
      cell_current_mismatched = true;
      ++summary.mismatched_cell_group_count;
      summary.original_outgoing_sum += raw_removal;
      summary.outgoing_change_sum += relative_change * raw_removal;
      summary.max_relative_outgoing_change =
        std::max(summary.max_relative_outgoing_change, relative_change);
    }

    if (use_leakage_fallback)
    {
      // The removal vanishes although the cell has a source: the outgoing faces share the
      // source equally.
      size_t outgoing_face_count = 0;
      for (size_t f = 0; f < cell_num_faces; ++f)
        if (orientations[f] == FOOUTGOING)
          ++outgoing_face_count;
      OpenSnLogicalErrorIf(outgoing_face_count == 0,
                           GetName() + ": near-source cell " + std::to_string(cell_global_id) +
                             " has a positive source but no outgoing face.");

      const double face_share = source[g] / static_cast<double>(outgoing_face_count);
      for (size_t f = 0; f < cell_num_faces; ++f)
        if (orientations[f] == FOOUTGOING)
          leakage[f][g] = face_share;
    }
    else
    {
      // Apply the shared correction to the leakage.
      for (size_t f = 0; f < cell_num_faces; ++f)
        if (orientations[f] == FOOUTGOING)
          leakage[f][g] *= alpha[g];
    }

    for (size_t f = 0; f < cell_num_faces; ++f)
      if (vacuum_boundary_faces[f] and orientations[f] == FOOUTGOING)
        out_flow += leakage[f][g];

    // What is left of the cell balance after the correction, relative to what enters the cell.
    double corrected_outflow = 0.0;
    for (size_t f = 0; f < cell_num_faces; ++f)
      if (orientations[f] == FOOUTGOING)
        corrected_outflow += leakage[f][g];
    const double residual =
      source[g] - corrected_outflow - alpha[g] * sigma_t[g] * projected_integral[g];
    const double relative_residual = source[g] > 0.0 ? std::abs(residual) / source[g] : 0.0;

    summary.min_alpha = std::min(summary.min_alpha, alpha[g]);
    summary.max_alpha = std::max(summary.max_alpha, alpha[g]);
    size_t band = 0;
    while (band < ConservationSummary::alpha_bands.size() and
           relative_change >= ConservationSummary::alpha_bands[band])
      ++band;
    ++summary.alpha_histogram[band];
    if (use_leakage_fallback)
      ++summary.leakage_fallback_count;
    summary.max_balance_residual = std::max(summary.max_balance_residual, relative_residual);
    if (relative_change > summary.largest_change)
    {
      summary.largest_change = relative_change;
      summary.largest_change_cell = cell_global_id;
      summary.largest_change_group = g;
      summary.largest_change_alpha = alpha[g];
    }
    if (summary.keep_cells)
    {
      summary.cell_alphas.push_back(alpha[g]);
      summary.cell_residuals.push_back(relative_residual);
    }
  }
  ++summary.num_cells;
  if (summary.keep_cells)
    summary.cell_ids.push_back(cell_global_id);

  if (cell_current_mismatched)
    ++summary.mismatched_cell_count;
}

void
UncollidedProblem::CorrectRayTracedCells(const SourcePoint& source_point,
                                         std::vector<RawNearSourceCell>& raw_cells,
                                         ConservationSummary& summary,
                                         double& source_angle_fraction,
                                         size_t& exact_neighbor_count,
                                         size_t& exact_neighbor_fallback_count,
                                         size_t& num_cells)
{
  CALI_CXX_MARK_SCOPE("CorrectRayTracedCells");

  constexpr auto FOINCOMING = FaceOrientation::INCOMING;
  constexpr auto FOOUTGOING = FaceOrientation::OUTGOING;
  constexpr auto FOPARALLEL = FaceOrientation::PARALLEL;
  constexpr std::uint64_t no_neighbor = std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t is_source_flag = 1;
  constexpr std::uint64_t is_exact_neighbor_flag = 2;
  constexpr std::uint64_t integrated_exactly_flag = 4;

  const size_t num_local_cells = near_spls_.size();
  const std::vector<double>& strength = source_point.strength;

  // What the correction needs from each cell, in the order of near_spls_. Integers: global id,
  // number of faces, flags, and for each face its orientation, whether it is a vacuum boundary
  // face, the global id of the cell beyond it and the index of the face in that cell. Reals: the
  // ray-traced leakage of each face and group, then the integral of the projected flux, the total
  // cross section and the point-source rate for each group, then the fraction of the source
  // emitted into the cell.
  std::vector<std::uint64_t> local_keys;
  std::vector<double> local_values;
  for (size_t n = 0; n < num_local_cells; ++n)
  {
    const size_t c = near_spls_[n];
    const Cell& cell = grid_->GetLocalCell(c);
    const auto& raw = raw_cells[n];
    const size_t cell_num_faces = cell.faces.size();

    local_keys.push_back(cell.global_id);
    local_keys.push_back(cell_num_faces);
    local_keys.push_back((raw.is_source_cell ? is_source_flag : 0) |
                         (raw.is_exact_neighbor ? is_exact_neighbor_flag : 0) |
                         (raw.integrated_exactly ? integrated_exactly_flag : 0));
    for (size_t f = 0; f < cell_num_faces; ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = cell_face_orientations_[c][f];
      local_keys.push_back(orientation == FOOUTGOING ? 2 : (orientation == FOINCOMING ? 1 : 0));
      local_keys.push_back(
        not face.has_neighbor and not IsReflectingBoundary(face.neighbor_id) ? 1 : 0);
      local_keys.push_back(face.has_neighbor ? face.neighbor_id : no_neighbor);
      local_keys.push_back(face.has_neighbor ? static_cast<std::uint64_t>(
                                                 face.GetNeighborAdjacentFaceIndex(grid_.get()))
                                             : 0);
      for (size_t g = 0; g < num_groups_; ++g)
        local_values.push_back(raw.leakage[f][g]);
    }

    const auto& IntV_shapeI = unit_cell_matrices_[c].intV_shapeI;
    for (size_t g = 0; g < num_groups_; ++g)
    {
      double projected_integral = 0.0;
      for (size_t i = 0; i < raw.phi[g].Rows(); ++i)
        projected_integral += IntV_shapeI(i) * raw.phi[g](i);
      local_values.push_back(projected_integral);
    }
    const auto& sigma_t = cell_transport_views_[c].GetXS().GetSigmaTotal();
    for (size_t g = 0; g < num_groups_; ++g)
      local_values.push_back(sigma_t[g]);
    // Each cell of the source receives the fraction of the source emitted into the solid angle
    // it subtends.
    const double angle_fraction =
      raw.is_source_cell ? SubtendedAngleFraction(cell, source_point.location) : 0.0;
    for (size_t g = 0; g < num_groups_; ++g)
      local_values.push_back(strength[g] * angle_fraction);
    local_values.push_back(angle_fraction);
  }

  const auto keys = GatherOnRoot(local_keys);
  const auto values = GatherOnRoot(local_values);
  const int num_ranks = opensn::mpi_comm.size();
  OpenSnLogicalErrorIf(num_local_cells * num_groups_ >
                         static_cast<size_t>(std::numeric_limits<int>::max()),
                       GetName() + ": too many cells and groups on one rank.");
  std::vector<int> cells_per_rank(num_ranks, 0);
  opensn::mpi_comm.all_gather(static_cast<int>(num_local_cells), cells_per_rank);

  // Scale factor of each cell and group, in the gathered order.
  std::vector<double> all_alpha;
  RunOnRoot(
    [&]()
    {
      struct TracedCell
      {
        std::uint64_t global_id = 0;
        std::uint64_t flags = 0;
        std::vector<FaceOrientation> orientations;
        std::vector<bool> vacuum_boundary_faces;
        std::vector<std::uint64_t> neighbors;
        std::vector<size_t> neighbor_faces;
        std::vector<std::vector<double>> leakage;
        std::vector<double> projected_integral;
        std::vector<double> sigma_t;
        std::vector<double> source;
        double angle_fraction = 0.0;
        /// Number of incoming faces whose upwind cell has not been corrected yet.
        size_t pending_inflows = 0;
      };

      std::vector<TracedCell> cells;
      std::unordered_map<std::uint64_t, size_t> cell_index;
      size_t key = 0;
      size_t value = 0;
      while (key < keys.size())
      {
        TracedCell cell;
        cell.global_id = keys[key++];
        const size_t cell_num_faces = keys[key++];
        cell.flags = keys[key++];
        cell.leakage.assign(cell_num_faces, std::vector<double>(num_groups_));
        for (size_t f = 0; f < cell_num_faces; ++f)
        {
          const auto orientation = keys[key++];
          cell.orientations.push_back(
            orientation == 2 ? FOOUTGOING : (orientation == 1 ? FOINCOMING : FOPARALLEL));
          cell.vacuum_boundary_faces.push_back(keys[key++] != 0);
          cell.neighbors.push_back(keys[key++]);
          cell.neighbor_faces.push_back(keys[key++]);
          if (cell.orientations.back() == FOINCOMING and cell.neighbors.back() != no_neighbor)
            ++cell.pending_inflows;
          for (size_t g = 0; g < num_groups_; ++g)
            cell.leakage[f][g] = values[value++];
        }
        for (auto* per_group : {&cell.projected_integral, &cell.sigma_t, &cell.source})
          for (size_t g = 0; g < num_groups_; ++g)
            per_group->push_back(values[value++]);
        cell.angle_fraction = values[value++];
        cell_index[cell.global_id] = cells.size();
        cells.push_back(std::move(cell));
      }
      OpenSnLogicalErrorIf(key != keys.size() or value != values.size(),
                           GetName() + ": the gathered ray-traced cells are inconsistent.");
      num_cells = cells.size();

      // Correct the cells from the source outward: a cell is ready once the cells upwind of it
      // are corrected. Among the ready cells the lowest global id goes first, so that the order,
      // and with it the round-off of the totals, does not depend on the partitioning.
      using ReadyCell = std::pair<std::uint64_t, size_t>;
      std::priority_queue<ReadyCell, std::vector<ReadyCell>, std::greater<>> ready;
      for (size_t idx = 0; idx < cells.size(); ++idx)
        if (cells[idx].pending_inflows == 0)
          ready.emplace(cells[idx].global_id, idx);

      all_alpha.assign(cells.size() * num_groups_, 1.0);
      std::vector<double> alpha;
      std::vector<char> leakage_fallback;
      size_t num_corrected = 0;
      while (not ready.empty())
      {
        const size_t idx = ready.top().second;
        ready.pop();
        auto& cell = cells[idx];
        const size_t cell_num_faces = cell.orientations.size();

        // Retrieve leakage in from incoming faces. Nothing enters through a boundary face.
        for (size_t f = 0; f < cell_num_faces; ++f)
          if (cell.orientations[f] == FOINCOMING and cell.neighbors[f] != no_neighbor)
          {
            const auto& upwind = cells[cell_index.at(cell.neighbors[f])];
            OpenSnLogicalErrorIf(upwind.orientations.at(cell.neighbor_faces[f]) != FOOUTGOING,
                                 GetName() + ": cells " + std::to_string(cell.global_id) + " and " +
                                   std::to_string(upwind.global_id) +
                                   " disagree on the orientation of their common face.");
            cell.leakage[f] = upwind.leakage[cell.neighbor_faces[f]];
          }

        if ((cell.flags & is_exact_neighbor_flag) != 0)
          ++((cell.flags & integrated_exactly_flag) != 0 ? exact_neighbor_count
                                                         : exact_neighbor_fallback_count);
        if ((cell.flags & is_source_flag) != 0)
        {
          source_angle_fraction += cell.angle_fraction;
          if ((cell.flags & integrated_exactly_flag) == 0)
            log.Log0Warning() << GetName() << ": cell " << cell.global_id
                              << " containing the point source at "
                              << source_point.location.PrintStr()
                              << " has a face that is not directly visible from the source. Its "
                                 "flux is projected from quadrature points, which is inaccurate "
                                 "near the source.";
        }

        CorrectCellBalance(cell.global_id,
                           cell.orientations,
                           cell.vacuum_boundary_faces,
                           cell.sigma_t,
                           cell.projected_integral,
                           cell.source,
                           cell.leakage,
                           alpha,
                           leakage_fallback,
                           summary,
                           out_flow_);
        std::copy(alpha.begin(), alpha.end(), all_alpha.begin() + idx * num_groups_);
        ++num_corrected;

        for (size_t f = 0; f < cell_num_faces; ++f)
          if (cell.orientations[f] == FOOUTGOING and cell.neighbors[f] != no_neighbor)
          {
            const size_t downwind = cell_index.at(cell.neighbors[f]);
            if (--cells[downwind].pending_inflows == 0)
              ready.emplace(cells[downwind].global_id, downwind);
          }
      }
      OpenSnLogicalErrorIf(num_corrected != cells.size(),
                           GetName() +
                             ": the cells cannot be ordered outward from the point "
                             "source at " +
                             source_point.location.PrintStr() +
                             " because their face orientations form a cycle.");
    });

  // Send each rank the scale factors of its cells.
  std::vector<double> local_alpha(num_local_cells * num_groups_, 1.0);
  constexpr int alpha_tag = 7651;
  if (opensn::mpi_comm.rank() == 0)
  {
    size_t offset = 0;
    for (int rank = 0; rank < num_ranks; ++rank)
    {
      const size_t count = static_cast<size_t>(cells_per_rank[rank]) * num_groups_;
      if (rank == 0)
        std::copy_n(all_alpha.begin(), count, local_alpha.begin());
      else
        opensn::mpi_comm.send(rank, alpha_tag, all_alpha.data() + offset, static_cast<int>(count));
      offset += count;
    }
  }
  else
    opensn::mpi_comm.recv(0, alpha_tag, local_alpha.data(), static_cast<int>(local_alpha.size()));

  // Apply the correction to the flux. Where the outgoing faces share the source equally the
  // factor is one and the flux is kept.
  const auto& sdm = *discretization_;
  for (size_t n = 0; n < num_local_cells; ++n)
  {
    const Cell& cell = grid_->GetLocalCell(near_spls_[n]);
    auto& phi = raw_cells[n].phi;
    for (size_t g = 0; g < num_groups_; ++g)
    {
      phi[g].Scale(local_alpha[n * num_groups_ + g]);
      for (size_t i = 0; i < phi[g].Rows(); ++i)
        destination_phi_[sdm.MapDOFLocal(cell, i) * num_groups_ + g] = phi[g](i);
    }
  }
}

void
UncollidedProblem::TraceNearSourceCell(RayTracer& ray_tracer,
                                       const SourcePoint& source_point,
                                       const size_t c,
                                       const bool is_exact_neighbor,
                                       RawNearSourceCell& raw)
{
  const auto& sdm = *discretization_;
  const Vector3& pt_loc = source_point.location;
  constexpr auto FOOUTGOING = FaceOrientation::OUTGOING;

  const Cell& cell = grid_->GetLocalCell(c);

  // Cell mapping
  auto coord_sys = grid_->GetCoordinateSystem();
  auto swf = SpatialWeightFunction::FromCoordinateType(coord_sys);
  const auto& cell_mapping = sdm.GetCellMapping(cell);
  const size_t cell_num_faces = cell.faces.size();
  const size_t cell_num_nodes = cell_mapping.GetNumNodes();
  const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();

  // RHS contributions for bulk region cells adjacent to a raytraced cell.
  auto& cell_bulk_rhs = raw.bulk_rhs;
  auto& bulk_face_rhs_weights = raw.bulk_face_rhs_weights;
  cell_bulk_rhs.clear();
  bulk_face_rhs_weights.assign(cell_num_faces, {});

  // The flux is singular at a point source, so neither the least-squares projection nor the
  // face quadrature resolves it in the cell that contains the source. That cell instead gets
  // its exact cell-average flux and exact face leakages. Optionally, so do its vertex
  // neighbors.
  std::optional<ExactCellIntegrals> exact_integrals;
  const bool is_source_cell =
    std::any_of(source_point.subscribers.begin(),
                source_point.subscribers.end(),
                [c](const auto& subscriber) { return subscriber.cell_local_id == c; });
  if (is_source_cell or is_exact_neighbor)
    exact_integrals = IntegrateCellExactly(ray_tracer, cell, source_point);
  raw.is_source_cell = is_source_cell;
  raw.is_exact_neighbor = is_exact_neighbor;
  raw.integrated_exactly = exact_integrals.has_value();

  // Compute leakages
  auto& cell_leakage = raw.leakage;
  cell_leakage.resize(cell_num_faces);
  std::vector<double> face_leakage;
  for (size_t f = 0; f < cell_num_faces; ++f)
  {
    const auto orientation = cell_face_orientations_[c][f];
    face_leakage.assign(num_groups_, 0.);

    // Compute leakage out of outgoing face
    if (orientation == FOOUTGOING)
    {
      // Face data
      const auto& face = cell.faces[f];

      const Vector3& normal = face.normal;
      const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);
      std::vector<std::vector<double>> face_fluxes;
      std::vector<double> face_omega_n_jxw;
      face_fluxes.reserve(fe_srf_data.GetQuadraturePointIndices().size());
      face_omega_n_jxw.reserve(fe_srf_data.GetQuadraturePointIndices().size());

      for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
      {
        const auto& qp_xyz = fe_srf_data.QPointXYZ(qp);
        const auto omega = ComputeOmega(pt_loc, qp_xyz);
        const auto phi_qp =
          RaytraceLine(ray_tracer, cell, qp_xyz, source_point, static_cast<int>(f));
        const double integrand = (*swf)(qp_xyz)*omega.Dot(normal) * fe_srf_data.JxW(qp);

        for (size_t g = 0; g < num_groups_; ++g)
          face_leakage[g] += phi_qp[g] * integrand;
        face_fluxes.push_back(phi_qp);
        face_omega_n_jxw.push_back(integrand);
      }

      // Scale the traced face fluxes so that the leakage through the face equals the exact
      // value. The traced values still set its distribution over face nodes.
      if (exact_integrals)
        for (size_t g = 0; g < num_groups_; ++g)
          if (face_leakage[g] > 0.0)
          {
            const double scale = exact_integrals->face_leakage[f][g] / face_leakage[g];
            for (auto& face_flux : face_fluxes)
              face_flux[g] *= scale;
            face_leakage[g] = exact_integrals->face_leakage[f][g];
          }

      if (face.has_neighbor and not ray_trace_all_cells_)
      {
        size_t neighbor_id = face.GetNeighborLocalID(grid_.get());

        // Near-source/bulk region interface
        if (cell_regions_[neighbor_id] == CellRegion::BULK)
        {
          // Face data
          const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
          const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);

          // Neighbor data
          const Cell& neighbor = grid_->GetLocalCell(neighbor_id);
          const auto& neighbor_mapping = sdm.GetCellMapping(neighbor);

          size_t f_ = face.GetNeighborAdjacentFaceIndex(grid_.get());
          const size_t neighbor_num_face_nodes = neighbor_mapping.GetNumFaceNodes(f_);
          double face_measure = 0.0;
          for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
            face_measure += (*swf)(fe_srf_data.QPointXYZ(qp)) * fe_srf_data.JxW(qp);
          OpenSnLogicalErrorIf(face_measure <= 0.0,
                               "Invalid surface measure on near-source cell " +
                                 std::to_string(cell.global_id) + " face " + std::to_string(f) +
                                 ".");

          for (size_t fi = 0; fi < num_face_nodes; ++fi)
          {
            const int i = cell_mapping.MapFaceNode(f, fi);

            int j = -1;
            for (size_t fj = 0; fj < neighbor_num_face_nodes; ++fj)
            {
              const int neighbor_node = neighbor_mapping.MapFaceNode(f_, fj);
              if (neighbor.vertex_ids[neighbor_node] == cell.vertex_ids[i])
              {
                j = neighbor_node;
                break;
              }
            }
            OpenSnLogicalErrorIf(j < 0,
                                 "Failed to map outgoing uncollided sweep face node from cell " +
                                   std::to_string(cell.global_id) + " face " + std::to_string(f) +
                                   " to neighbor cell " + std::to_string(neighbor.global_id) +
                                   " face " + std::to_string(f_) + ".");

            // Compute rhs for bulk region sweep
            const auto jr = sdm.MapDOFLocal(neighbor, j);
            double fallback_rhs_weight = 0.0;
            size_t qp_index = 0;
            for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
            {
              // Compute the upwind face contribution to the neighboring bulk
              // cell's RHS.  The bulk sweep later computes the cell RHS from
              // destination_phi_, and storing by the neighbor's local DOF keeps
              // this interface term independent of the current cell's local
              // face-node ordering.
              const double integrand = face_omega_n_jxw[qp_index] * fe_srf_data.ShapeValue(i, qp);

              auto& rhs_g = cell_bulk_rhs[jr];
              if (rhs_g.empty())
                rhs_g.assign(num_groups_, 0.0);

              for (size_t g = 0; g < num_groups_; ++g)
                rhs_g[g] += face_fluxes[qp_index][g] * integrand;
              fallback_rhs_weight += (*swf)(fe_srf_data.QPointXYZ(qp)) *
                                     fe_srf_data.ShapeValue(i, qp) * fe_srf_data.JxW(qp);
              ++qp_index;
            } // for qp
            bulk_face_rhs_weights[f].emplace_back(jr, fallback_rhs_weight / face_measure);
          } // for fi
        } // if neighbor_id in bulk_spls_
      } // if face.has_neighbor
    } // if outgoing

    // Save leakage through face
    cell_leakage[f] = face_leakage;
  }

  // Ray-traced solution in cell
  auto& phi = raw.phi;
  phi.assign(num_groups_, Vector<double>(cell_num_nodes, 0.));

  if (exact_integrals and is_source_cell)
  {
    // Represent the source cell by its exact cell-average flux.
    const auto& intV_shapeI = unit_cell_matrices_[c].intV_shapeI;
    const double cell_volume = std::accumulate(intV_shapeI.begin(), intV_shapeI.end(), 0.0);
    for (size_t g = 0; g < num_groups_; ++g)
      phi[g].Set(exact_integrals->volume_integral[g] / cell_volume);
  }
  else
  {
    // Mass matrix times least-squares flux vector
    for (const auto& qp : fe_vol_data.GetQuadraturePointIndices())
    {
      // Raytrace to point
      Vector3 qp_xyz = fe_vol_data.QPointXYZ(qp);

      std::vector<double> phi_qp = RaytraceLine(ray_tracer, cell, qp_xyz, source_point);

      for (unsigned int i = 0; i < cell_num_nodes; ++i)
      {
        // Integrand value at quadrature point
        double integrand =
          (*swf)(fe_vol_data.QPointXYZ(qp)) * fe_vol_data.ShapeValue(i, qp) * fe_vol_data.JxW(qp);

        // Compute group-wise least-squares fluxes
        for (size_t g = 0; g < num_groups_; ++g)
          phi[g](i) += phi_qp[g] * integrand;
      }
    }

    // Invert mass matrix
    for (size_t g = 0; g < num_groups_; ++g)
    {
      auto mass_matrix = unit_cell_matrices_[c].intV_shapeI_shapeJ;
      GaussElimination(mass_matrix, phi[g], static_cast<int>(cell_num_nodes));
    }

    // Keep the projected shape but match the exact volume integral, or use the exact average
    // when the projection has no positive integral to scale.
    if (exact_integrals)
    {
      const auto& intV_shapeI = unit_cell_matrices_[c].intV_shapeI;
      const double cell_volume = std::accumulate(intV_shapeI.begin(), intV_shapeI.end(), 0.0);
      for (size_t g = 0; g < num_groups_; ++g)
      {
        double projected_integral = 0.0;
        for (size_t i = 0; i < cell_num_nodes; ++i)
          projected_integral += intV_shapeI(i) * phi[g](i);
        if (projected_integral > 0.0)
          phi[g].Scale(exact_integrals->volume_integral[g] / projected_integral);
        else
          phi[g].Set(exact_integrals->volume_integral[g] / cell_volume);
      }
    }
  }
}

void
UncollidedProblem::RaytraceNearSourceRegion(const SourcePoint& source_point)
{
  CALI_CXX_MARK_SCOPE("RaytraceNearSourceRegion");
  log.Log() << (ray_trace_all_cells_ ? "\nRay-tracing all cells.\n"
                                     : "\nRay-tracing near-source region.\n");

  const auto& sdm = *discretization_;

  // Point source data
  const Vector3& pt_loc = source_point.location;
  const std::vector<double>& strength = source_point.strength;

  // Face orientations
  constexpr auto FOINCOMING = FaceOrientation::INCOMING;
  constexpr auto FOOUTGOING = FaceOrientation::OUTGOING;

  ConservationSummary summary;
  summary.keep_cells = write_diagnostics_;

  // Total fraction of the source emitted into the source cells.
  double source_angle_fraction = 0.0;

  // With exact_near_source_leakage, the near-source cells that share a vertex with a source cell
  // are integrated exactly over direction. A source close to an edge or vertex of its cell is close
  // to these cells too, where the spatial quadratures cannot resolve the flux.
  std::vector<bool> is_exact_neighbor(grid_->GetLocalCellCount(), false);
  size_t exact_neighbor_count = 0;
  size_t exact_neighbor_fallback_count = 0;
  if (exact_near_source_leakage_)
  {
    std::vector<uint64_t> source_vertex_ids;
    for (const auto& subscriber : source_point.subscribers)
    {
      const auto& vertex_ids = grid_->GetLocalCell(subscriber.cell_local_id).vertex_ids;
      source_vertex_ids.insert(source_vertex_ids.end(), vertex_ids.begin(), vertex_ids.end());
    }
    // The cells of the source can be on other ranks.
    if (opensn::mpi_comm.size() > 1)
    {
      source_vertex_ids = GatherOnRoot(source_vertex_ids);
      opensn::mpi_comm.broadcast(source_vertex_ids, 0);
    }
    std::sort(source_vertex_ids.begin(), source_vertex_ids.end());
    for (const size_t c : near_spls_)
    {
      if (std::any_of(source_point.subscribers.begin(),
                      source_point.subscribers.end(),
                      [c](const auto& subscriber) { return subscriber.cell_local_id == c; }))
        continue;
      for (const auto vid : grid_->GetLocalCell(c).vertex_ids)
        if (std::binary_search(source_vertex_ids.begin(), source_vertex_ids.end(), vid))
        {
          is_exact_neighbor[c] = true;
          break;
        }
    }
  }

  Timer phase_timer;
  // Trace every near-source cell. The ray-traced data of a cell does not depend on the other
  // cells, so the cells are shared among the threads; each thread writes to the cells it takes.
  const size_t num_near_cells = near_spls_.size();
  std::vector<RawNearSourceCell> raw_cells(num_near_cells);
  {
    const size_t num_threads =
      std::min<size_t>(std::max(1U, opensn_num_threads), std::max<size_t>(1, num_near_cells));
    if (num_threads > 1)
      log.Log() << "Tracing " << num_near_cells << " near-source cells with " << num_threads
                << " threads.";
    std::atomic<size_t> next_cell = 0;
    std::exception_ptr worker_exception;
    std::mutex exception_mutex;
    auto TraceCells = [&]()
    {
      try
      {
        RayTracer ray_tracer(grid_, &cell_sizes_, not all_cells_convex_);
        for (size_t n = next_cell.fetch_add(1); n < num_near_cells; n = next_cell.fetch_add(1))
          TraceNearSourceCell(ray_tracer,
                              source_point,
                              near_spls_[n],
                              is_exact_neighbor[near_spls_[n]],
                              raw_cells[n]);
      }
      catch (...)
      {
        // Let the other threads stop at their next cell.
        next_cell.store(num_near_cells);
        std::scoped_lock lock(exception_mutex);
        if (not worker_exception)
          worker_exception = std::current_exception();
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(num_threads - 1);
    for (size_t thread_id = 1; thread_id < num_threads; ++thread_id)
      workers.emplace_back(TraceCells);
    TraceCells();
    for (auto& worker : workers)
      worker.join();
    if (worker_exception)
      std::rethrow_exception(worker_exception);
  }

  timings_.ray_tracing += phase_timer.GetTime() / 1000.0;
  phase_timer.Reset();

  // Number of cells ray traced, over all ranks.
  size_t num_traced_cells = num_near_cells;
  if (ray_trace_all_cells_)
    CorrectRayTracedCells(source_point,
                          raw_cells,
                          summary,
                          source_angle_fraction,
                          exact_neighbor_count,
                          exact_neighbor_fallback_count,
                          num_traced_cells);
  else
  {
    // Position of each near-source cell in the sweep order.
    constexpr size_t not_near_source = std::numeric_limits<size_t>::max();
    std::vector<size_t> near_index(grid_->GetLocalCellCount(), not_near_source);
    for (size_t n = 0; n < num_near_cells; ++n)
      near_index[near_spls_[n]] = n;

    // Enforce conservation cell by cell, from the source outward: the corrected leakage out of a
    // cell is the leakage into the cells downwind of it.
    for (size_t n = 0; n < num_near_cells; ++n)
    {
      const size_t c = near_spls_[n];
      auto& raw = raw_cells[n];
      const Cell& cell = grid_->GetLocalCell(c);

      const auto& cell_mapping = sdm.GetCellMapping(cell);
      const size_t cell_num_faces = cell.faces.size();
      const size_t cell_num_nodes = cell_mapping.GetNumNodes();
      const bool is_source_cell = raw.is_source_cell;
      auto& cell_bulk_rhs = raw.bulk_rhs;
      const auto& bulk_face_rhs_weights = raw.bulk_face_rhs_weights;
      auto& phi = raw.phi;
      auto& cell_leakage = raw.leakage;

      if (raw.is_exact_neighbor)
        ++(raw.integrated_exactly ? exact_neighbor_count : exact_neighbor_fallback_count);
      if (is_source_cell and not raw.integrated_exactly)
        log.Log0Warning() << GetName() << ": cell " << cell.global_id
                          << " containing the point source at " << pt_loc.PrintStr()
                          << " has a face that is not directly visible from the source. Its "
                             "flux is projected from quadrature points, which is inaccurate "
                             "near the source.";

      // Retrieve leakage in from incoming faces. Nothing enters through a boundary face, whose
      // leakage stays zero.
      for (size_t f = 0; f < cell_num_faces; ++f)
        if (cell_face_orientations_[c][f] == FOINCOMING and cell.faces[f].has_neighbor)
        {
          const size_t neigh_id = cell.faces[f].GetNeighborLocalID(grid_.get());
          const size_t neigh_face_ind = cell.faces[f].GetNeighborAdjacentFaceIndex(grid_.get());
          OpenSnLogicalErrorIf(near_index[neigh_id] == not_near_source,
                               GetName() + ": near-source cell " + std::to_string(cell.global_id) +
                                 " has an upwind cell outside the near-source region.");
          cell_leakage[f] = raw_cells[near_index[neigh_id]].leakage[neigh_face_ind];
        }

      // Transport view
      const auto& transport_view = cell_transport_views_[c];
      const auto& xs = transport_view.GetXS();
      const auto& sigma_t = xs.GetSigmaTotal();

      const auto& fe_intgrl_values = unit_cell_matrices_[cell.local_id];
      const auto& IntV_shapeI = fe_intgrl_values.intV_shapeI;

      // Enforce conservation
      std::vector<double> source(num_groups_, 0.);

      // Use the point source subscriber list as the source of truth instead of
      // re-testing geometry.  This matters for sources exactly on a face or
      // vertex: subscriber discovery intentionally includes every incident cell,
      // while a sharp point-in-cell test can reject some of those cells through
      // roundoff.  The uncollided flux streams radially from the source, so each
      // incident cell receives the fraction of the source emitted into the solid
      // angle it subtends, not a volume-weighted share.
      if (is_source_cell)
      {
        const double angle_fraction = SubtendedAngleFraction(cell, pt_loc);
        source_angle_fraction += angle_fraction;
        for (size_t g = 0; g < num_groups_; ++g)
          source[g] += strength[g] * angle_fraction;
      }

      // Integral of the projected flux, before the correction.
      std::vector<double> projected_integral(num_groups_, 0.0);
      for (size_t g = 0; g < num_groups_; ++g)
        for (size_t i = 0; i < cell_num_nodes; ++i)
          projected_integral[g] += IntV_shapeI(i) * phi[g](i);

      std::vector<bool> vacuum_boundary_faces(cell_num_faces, false);
      for (size_t f = 0; f < cell_num_faces; ++f)
        vacuum_boundary_faces[f] =
          not cell.faces[f].has_neighbor and not IsReflectingBoundary(cell.faces[f].neighbor_id);

      // Apply the cell-wise correction from Woodsford et al. (2026), Eqs. (24)-(25), to the
      // leakages, then to the flux and to the bulk interface source.
      std::vector<double> alpha;
      std::vector<char> leakage_fallback;
      CorrectCellBalance(cell.global_id,
                         cell_face_orientations_[c],
                         vacuum_boundary_faces,
                         sigma_t,
                         projected_integral,
                         source,
                         cell_leakage,
                         alpha,
                         leakage_fallback,
                         summary,
                         out_flow_);
      for (size_t g = 0; g < num_groups_; ++g)
      {
        phi[g].Scale(alpha[g]);
        if (leakage_fallback[g] != 0)
        {
          // The outgoing faces share the source equally, as a uniform face flux.
          for (auto& [jr, rhs_g] : cell_bulk_rhs)
            rhs_g[g] = 0.0;
          for (size_t f = 0; f < cell_num_faces; ++f)
            if (cell_face_orientations_[c][f] == FOOUTGOING)
              for (const auto& [jr, weight] : bulk_face_rhs_weights[f])
                cell_bulk_rhs[jr][g] += cell_leakage[f][g] * weight;
        }
        else
          for (auto& [jr, rhs_g] : cell_bulk_rhs)
            rhs_g[g] *= alpha[g];
      }

      // The bulk sweep receives the analytic ray-traced interface current.
      for (const auto& [ir, rhs_g] : cell_bulk_rhs)
        for (size_t g = 0; g < num_groups_; ++g)
          destination_phi_[ir * num_groups_ + g] += rhs_g[g];

      // Update flux solution
      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        const auto ir = sdm.MapDOFLocal(cell, i);
        for (size_t g = 0; g < num_groups_; ++g)
          destination_phi_[ir * num_groups_ + g] = phi[g](i);
      }
    }
  }

  // A source on the domain boundary emits part of its strength directly out of the domain.
  // Reflecting setups reject sources on faces, so this part always leaves through a
  // non-reflecting boundary.
  timings_.conservation += phase_timer.GetTime() / 1000.0;

  // Rank 0 holds the fraction emitted into the cells of all ranks.
  const double escaped_fraction = std::max(0.0, 1.0 - source_angle_fraction);
  if (opensn::mpi_comm.rank() == 0)
    for (size_t g = 0; g < num_groups_; ++g)
      out_flow_ += strength[g] * escaped_fraction;

  if (exact_near_source_leakage_)
    log.Log()
      << "Exact near-source leakage: " << exact_neighbor_count
      << " neighbor cells integrated over direction, " << exact_neighbor_fallback_count
      << " cells projected from quadrature points because they are non-convex or an outgoing face "
         "is not directly visible from the source.";

  const size_t near_source_pair_count = num_traced_cells * num_groups_;
  const double mismatched_cell_fraction =
    num_traced_cells == 0
      ? 0.0
      : static_cast<double>(summary.mismatched_cell_count) / static_cast<double>(num_traced_cells);
  const double aggregate_relative_change =
    summary.original_outgoing_sum > 0.0
      ? summary.outgoing_change_sum / summary.original_outgoing_sum
      : 0.0;
  log.Log() << std::setprecision(6) << std::scientific
            << "Near-source consistency: " << summary.mismatched_cell_count << " / "
            << num_traced_cells << " cells required a conservation correction greater than 1%"
            << " (aggregate correction " << aggregate_relative_change << ").";
  log.Log0Verbose1() << " Near-source conservation correction:\n"
                     << "  Corrected cells             = " << summary.mismatched_cell_count << " / "
                     << num_traced_cells << "\n"
                     << "  Corrected cell-group pairs  = " << summary.mismatched_cell_group_count
                     << " / " << near_source_pair_count << "\n"
                     << std::setprecision(6) << std::scientific
                     << "  Mismatched cell fraction     = " << mismatched_cell_fraction << "\n"
                     << "  Aggregate relative mismatch  = " << aggregate_relative_change << "\n"
                     << "  Maximum relative mismatch    = " << summary.max_relative_outgoing_change
                     << "\n"
                     << " NearSourceCurrentMismatchedCellFraction=" << mismatched_cell_fraction
                     << "\n"
                     << " NearSourceCurrentAggregateRelativeMismatch=" << aggregate_relative_change
                     << "\n";

  if (summary.num_cells > 0)
  {
    std::ostringstream report;
    report << std::setprecision(6) << std::scientific << "Conservation correction of "
           << summary.num_cells << " cells in " << num_groups_ << " groups:\n"
           << " Smallest scale factor alpha            = " << summary.min_alpha << "\n"
           << " Largest scale factor alpha             = " << summary.max_alpha << "\n"
           << " Cell-group pairs by |alpha - 1|        :";
    for (size_t band = 0; band < summary.alpha_histogram.size(); ++band)
    {
      if (band < ConservationSummary::alpha_bands.size())
        report << " below " << std::setprecision(0) << ConservationSummary::alpha_bands[band]
               << ": ";
      else
        report << " above: ";
      report << summary.alpha_histogram[band]
             << (band + 1 < summary.alpha_histogram.size() ? "," : "");
    }
    report << std::setprecision(6) << "\n"
           << " Pairs whose outgoing faces share source = " << summary.leakage_fallback_count
           << "\n"
           << " Largest cell balance residual          = " << summary.max_balance_residual << "\n";
    if (summary.largest_change > 0.1)
      report << " Largest correction                     : cell " << summary.largest_change_cell
             << ", group " << summary.largest_change_group
             << ", alpha = " << summary.largest_change_alpha << "\n";
    log.Log() << report.str();
  }
  if (write_diagnostics_ and opensn::mpi_comm.rank() == 0)
  {
    auto& diagnostics = source_diagnostics_.emplace_back();
    diagnostics.cell_ids = std::move(summary.cell_ids);
    diagnostics.alphas = std::move(summary.cell_alphas);
    diagnostics.residuals = std::move(summary.cell_residuals);
  }

  constexpr double aggregate_current_change_warning = 0.5;
  if (aggregate_relative_change > aggregate_current_change_warning)
    log.Log0Warning()
      << GetName()
      << ": near-source face-current and volume-removal quadratures differ significantly. "
      << std::fixed << std::setprecision(1) << 100.0 * mismatched_cell_fraction
      << "% of near-source cells exceed the correction threshold, with "
      << 100.0 * aggregate_relative_change
      << "% aggregate relative mismatch. The ray-traced outgoing leakage and the ray-traced "
      << "volume-removal are reconciled with a single shared scale factor. Large values here "
      << "mean that correction is significant and the mesh may need refinement near the source.";
}

std::vector<double>
UncollidedProblem::RaytraceLine(RayTracer& ray_tracer,
                                const Cell& cell,
                                const Vector3& qp_xyz,
                                const SourcePoint& source_point,
                                const int face_index,
                                const double tolerance)
{
  std::vector<double> phi(num_groups_, 0.0);
  std::vector<std::pair<size_t, double>> scratch_segs;
  std::vector<double> scratch_bp;
  std::vector<double> scratch_mfp(num_groups_, 0.0);
  scratch_segs.reserve(16);
  scratch_bp.reserve(8);
  RaytraceLineInto(ray_tracer,
                   cell,
                   qp_xyz,
                   source_point,
                   phi,
                   scratch_segs,
                   scratch_bp,
                   scratch_mfp,
                   face_index,
                   tolerance);
  return phi;
}

namespace
{

/// Calls f(x, w) at the points and weights of a composite 8-point Gauss-Legendre rule on [a, b]
/// whose panels are at most 1 wide.
template <typename Function>
void
ForEachCompositeGaussPoint(const double a, const double b, const Function& f)
{
  static const GaussLegendreQuadrature panel_quadrature(8U);
  const auto num_panels = std::max<size_t>(1, static_cast<size_t>(std::ceil(std::abs(b - a))));
  const double width = (b - a) / static_cast<double>(num_panels);
  for (size_t p = 0; p < num_panels; ++p)
  {
    const double left = a + static_cast<double>(p) * width;
    for (size_t q = 0; q < panel_quadrature.qpoints.size(); ++q)
      f(left + 0.5 * width * (panel_quadrature.qpoints[q].x + 1.0),
        0.5 * width * panel_quadrature.weights[q]);
  }
}

/// Calls visit(x, w) at quadrature points x on triangle T and their solid-angle weights w as seen
/// from a source at perpendicular distance h > 0 from the plane of T, so that the sum of
/// w f(x) approximates the integral of f over the directions from the source through T.
///
/// T is fanned into sub-triangles (A, P, Q) from the point A of T closest to the source, so every
/// x lies in T. In each, theta is the angle about A, and with x measured along PQ from the foot of
/// the perpendicular from A (at distance d), x = d sinh(v) gives dtheta = sech(v) dv. The radius
/// rho from A is graded as rho = r_min sinh(w), where r_min is the distance from the source to A,
/// so the integrand stays smooth however close the source is to T. With r the distance from the
/// source to x, dOmega = h / r^3 dA and dA = rho drho dtheta.
template <typename Visit>
void
ForEachTriangleDirection(const std::array<Vector3, 3>& triangle,
                         const Vector3& source,
                         const double h,
                         const Visit& visit)
{
  constexpr double edge_tolerance = 1.0e-12;

  const Vector3 apex = ClosestPointOnTriangle(source, triangle[0], triangle[1], triangle[2]);
  const double r_min = (apex - source).Norm();
  for (size_t e = 0; e < 3; ++e)
  {
    const Vector3& P = triangle[e];
    const Vector3& Q = triangle[(e + 1) % 3];
    const double edge_length = (Q - P).Norm();
    if (edge_length == 0.0)
      continue;
    const Vector3 tangent = (Q - P) / edge_length;
    const double x0 = tangent.Dot(P - apex);
    const double d = ((P - apex) - x0 * tangent).Norm();
    // The apex lies on this edge, so its sub-triangle is degenerate.
    if (d <= edge_tolerance * edge_length)
      continue;

    const Vector3 edge_foot = P - x0 * tangent;
    ForEachCompositeGaussPoint(
      std::asinh(x0 / d),
      std::asinh((x0 + edge_length) / d),
      [&](const double v, const double wv)
      {
        const double rho_max = d * std::cosh(v);
        const Vector3 direction = (edge_foot + d * std::sinh(v) * tangent - apex) / rho_max;
        const double w_theta = wv / std::cosh(v);
        ForEachCompositeGaussPoint(0.0,
                                   std::asinh(rho_max / r_min),
                                   [&](const double w, const double ww)
                                   {
                                     const double rho = r_min * std::sinh(w);
                                     const Vector3 x = apex + rho * direction;
                                     const double r = (x - source).Norm();
                                     const double drho = r_min * std::cosh(w) * ww;
                                     visit(x, w_theta * rho * drho * h / (r * r * r));
                                   });
      });
  }
}

/// Calls f(triangle, h) for each PWL side triangle (two face vertices and the face centroid) of a
/// face whose plane is at distance h > 0 from the source on the side opposite its outward normal.
/// Side triangles whose plane contains the source are skipped. Returns false, without visiting
/// further triangles, if the source is on the outer side of a side-triangle plane.
template <typename Function>
bool
ForEachVisibleSideTriangle(const Mesh& grid,
                           const CellFace& face,
                           const Vector3& source,
                           const Function& f)
{
  // Relative tolerance on the distance from the source to a side plane.
  constexpr double plane_tolerance = 1.0e-12;

  const auto& vertex_ids = face.vertex_ids;
  const size_t num_sides = vertex_ids.size();
  for (size_t side = 0; side < num_sides; ++side)
  {
    const std::array<Vector3, 3> triangle = {grid.GlobalVertex(vertex_ids[side]),
                                             grid.GlobalVertex(vertex_ids[(side + 1) % num_sides]),
                                             face.centroid};
    Vector3 normal = (triangle[1] - triangle[0]).Cross(triangle[2] - triangle[0]);
    const double normal_norm = normal.Norm();
    if (normal_norm == 0.0)
      continue;
    normal = normal / normal_norm;
    if (normal.Dot(face.normal) < 0.0)
      normal = -1.0 * normal;

    double length = 0.0;
    for (size_t e = 0; e < 3; ++e)
      length = std::max(length, (triangle[(e + 1) % 3] - triangle[e]).Norm());

    const double h = normal.Dot(triangle[0] - source);
    if (h < -plane_tolerance * length)
      return false;
    if (h <= plane_tolerance * length)
      continue;
    f(triangle, h);
  }
  return true;
}

} // namespace

std::optional<UncollidedProblem::ExactCellIntegrals>
UncollidedProblem::IntegrateCellExactly(RayTracer& ray_tracer,
                                        const Cell& cell,
                                        const SourcePoint& source_point)
{
  const Vector3& source = source_point.location;
  const bool contains_source = std::any_of(source_point.subscribers.begin(),
                                           source_point.subscribers.end(),
                                           [&cell](const auto& subscriber)
                                           { return subscriber.cell_local_id == cell.local_id; });

  // The entry distance below assumes that a ray from the source crosses the cell at most once.
  // That holds for a convex cell, and for the cell containing the source when every face is
  // visible from it, which the face loop checks.
  if (not contains_source and not IsConvex(cell))
    return std::nullopt;

  const auto& sigma_t = cell_transport_views_[cell.local_id].GetXS().GetSigmaTotal();

  // Planes of the side triangles of every face, for the distance at which a ray from the source
  // enters the cell. It is zero for the cell containing the source.
  std::vector<std::pair<Vector3, Vector3>> side_planes;
  for (const auto& face : cell.faces)
  {
    const auto& vertex_ids = face.vertex_ids;
    for (size_t side = 0; side < vertex_ids.size(); ++side)
    {
      const Vector3& P = grid_->GlobalVertex(vertex_ids[side]);
      const Vector3& Q = grid_->GlobalVertex(vertex_ids[(side + 1) % vertex_ids.size()]);
      Vector3 normal = (Q - P).Cross(face.centroid - P);
      if (normal.Norm() == 0.0)
        continue;
      normal = normal.Normalized();
      if (normal.Dot(face.normal) < 0.0)
        normal = -1.0 * normal;
      side_planes.emplace_back(P, normal);
    }
  }

  // Along a ray from the source the flux is S exp(-tau(r)) / (4 pi r^2). Every direction through
  // the cell leaves it through an outgoing face at a distance R, having entered at R_in, with the
  // chord Delta = R - R_in in this homogeneous cell. With tau_in the optical depth from the
  // source to the entry point, the leakage integrand is S exp(-tau_in - sigma_t Delta) / (4 pi)
  // and the volume integrand is S exp(-tau_in) (1 - exp(-sigma_t Delta)) / (4 pi sigma_t).
  std::vector<double> volume(num_groups_, 0.0);
  std::vector<std::vector<double>> leakage(cell.faces.size(), std::vector<double>(num_groups_));
  std::vector<double> phi(num_groups_, 0.0);
  std::vector<std::pair<size_t, double>> scratch_segs;
  std::vector<double> scratch_bp;
  std::vector<double> scratch_mfp(num_groups_, 0.0);

  for (size_t f = 0; f < cell.faces.size(); ++f)
  {
    const bool outgoing = cell_face_orientations_[cell.local_id][f] == FaceOrientation::OUTGOING;
    bool visible_through_non_outgoing_face = false;
    const auto visible = ForEachVisibleSideTriangle(
      *grid_,
      cell.faces[f],
      source,
      [&](const std::array<Vector3, 3>& triangle, const double h)
      {
        if (not outgoing)
        {
          visible_through_non_outgoing_face = true;
          return;
        }
        ForEachTriangleDirection(
          triangle,
          source,
          h,
          [&](const Vector3& x, const double w)
          {
            const double R = (x - source).Norm();
            const Vector3 omega = (x - source) / R;
            double R_in = 0.0;
            for (const auto& [point, normal] : side_planes)
            {
              const double mu = omega.Dot(normal);
              if (mu < 0.0)
                R_in = std::max(R_in, normal.Dot(point - source) / mu);
            }
            const double chord = std::max(0.0, R - R_in);

            // On return, scratch_mfp holds the optical depth from the source to x.
            RaytraceLineInto(ray_tracer,
                             cell,
                             x,
                             source_point,
                             phi,
                             scratch_segs,
                             scratch_bp,
                             scratch_mfp,
                             static_cast<int>(f));
            for (size_t g = 0; g < num_groups_; ++g)
            {
              const double optical_chord = sigma_t[g] * chord;
              const double tau_in = std::max(0.0, scratch_mfp[g] - optical_chord);
              const double entering =
                w * source_point.strength[g] / (4.0 * M_PI) * std::exp(-tau_in);
              leakage[f][g] += entering * std::exp(-optical_chord);
              volume[g] +=
                entering * (optical_chord > 0.0 ? -std::expm1(-optical_chord) / sigma_t[g] : chord);
            }
          });
      });
    // The faces through which the source's flux leaves the cell must all be outgoing faces that
    // are entirely visible from the source. Other faces of a cell that does not contain the
    // source face the source and are skipped.
    if (visible_through_non_outgoing_face or (not visible and (contains_source or outgoing)))
      return std::nullopt;
  }

  return ExactCellIntegrals{std::move(volume), std::move(leakage)};
}

bool
UncollidedProblem::IsConvex(const Cell& cell) const
{
  const double tol = cell_sizes_[cell.local_id] * 1.0e-8;
  for (const auto& face : cell.faces)
  {
    const auto& v0 = grid_->GlobalVertex(face.vertex_ids.front());
    const auto& n = face.normal;
    for (const auto vid : cell.vertex_ids)
    {
      if (std::find(face.vertex_ids.begin(), face.vertex_ids.end(), vid) != face.vertex_ids.end())
        continue;
      if (n.Dot(grid_->GlobalVertex(vid) - v0) > tol)
        return false;
    }
  }
  return true;
}

double
UncollidedProblem::SubtendedAngleFraction(const Cell& cell, const Vector3& point) const
{
  // Relative tolerance on the distance from the point to a face or side plane. A face whose plane
  // contains the point subtends no angle.
  constexpr double plane_tolerance = 1.0e-12;

  double angle = 0.0;
  for (const auto& face : cell.faces)
  {
    const auto& vertex_ids = face.vertex_ids;

    // Van Oosterom-Strackee solid angle of each PWL side triangle, signed by the side of its plane
    // the point is on.
    const size_t num_sides = vertex_ids.size();
    for (size_t side = 0; side < num_sides; ++side)
    {
      const Vector3 a = grid_->GlobalVertex(vertex_ids[side]) - point;
      const Vector3 b = grid_->GlobalVertex(vertex_ids[(side + 1) % num_sides]) - point;
      const Vector3 c = face.centroid - point;
      Vector3 normal = (b - a).Cross(c - a);
      const double normal_norm = normal.Norm();
      if (normal_norm == 0.0)
        continue;
      if (normal.Dot(face.normal) < 0.0)
        normal = -1.0 * normal;

      const double length = std::max({(b - a).Norm(), (c - b).Norm(), (a - c).Norm()});
      const double h = normal.Dot(a) / normal_norm;
      if (std::abs(h) <= plane_tolerance * length)
        continue;

      const double a_norm = a.Norm();
      const double b_norm = b.Norm();
      const double c_norm = c.Norm();
      const double denominator =
        a_norm * b_norm * c_norm + a.Dot(b) * c_norm + a.Dot(c) * b_norm + b.Dot(c) * a_norm;
      angle += 2.0 * std::atan2(normal.Dot(a), denominator);
    }
  }

  return std::clamp(angle / (4.0 * M_PI), 0.0, 1.0);
}

void
UncollidedProblem::SweepBulkRegion(const SourcePoint& source_point)
{
  CALI_CXX_MARK_SCOPE("SweepBulkRegion");
  log.Log() << "Sweeping bulk region.\n";

  const auto& sdm = *discretization_;
  const auto& pt_loc = source_point.location;
  const size_t num_group_threads = std::min<size_t>(std::max(1U, opensn_num_threads), num_groups_);

  if (num_group_threads > 1)
    log.Log() << "Bulk sweep will process " << num_groups_ << " groups over " << num_group_threads
              << " threads.";

  // Relative balance residual of each bulk cell and group, for the diagnostics file.
  std::vector<double> bulk_residuals(write_diagnostics_ ? bulk_spls_.size() * num_groups_ : 0);

  auto SweepGroups = [&](const size_t thread_id)
  {
    // Sweep bulk region cells
    for (size_t bulk_index = 0; bulk_index < bulk_spls_.size(); ++bulk_index)
    {
      const size_t c = bulk_spls_[bulk_index];
      const Cell& cell = grid_->GetLocalCell(c);

      // Cell data
      const auto& cell_mapping = sdm.GetCellMapping(cell);
      const size_t cell_num_faces = cell.faces.size();
      const size_t cell_num_nodes = cell_mapping.GetNumNodes();

      const auto& transport_view = cell_transport_views_[c];
      const auto& xs = transport_view.GetXS();
      const auto& sigma_t = xs.GetSigmaTotal();

      // Compute matrices
      const auto matrices = ComputeUncollidedIntegrals(cell, pt_loc);
      const auto& base_matrix = matrices.intV_shapeJ_omega_gradshapeI;
      const auto& mass_matrix = unit_cell_matrices_[c].intV_shapeI_shapeJ;

      for (size_t g = thread_id; g < num_groups_; g += num_group_threads)
      {
        // Keep per-cell work arrays at the active size because mixed
        // unstructured meshes can have different node counts in consecutive
        // sweep cells.
        DenseMatrix<double> Atemp(cell_num_nodes, cell_num_nodes, 0.0);
        Vector<double> phi(cell_num_nodes, 0.0);

        // Contributions from near-source cells are precomputed during the
        // ray-tracing stage. Seed the bulk-cell RHS with those values once per
        // cell; the face loop below only adds already-swept bulk upwind cells.
        // For an incoming face whose neighbor is in near_spls_, the loop
        // deliberately does nothing so the precomputed interface RHS is not
        // counted twice.
        for (size_t i = 0; i < cell_num_nodes; ++i)
        {
          const auto ir = sdm.MapDOFLocal(cell, i);
          phi(i) += destination_phi_[ir * num_groups_ + g];
        }

        auto Amat = base_matrix;

        // Surface matrices
        for (size_t f = 0; f < cell_num_faces; ++f)
        {
          const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
          const auto& surface_matrix = matrices.intS_omega_n_shapeI_shapeJ[f];

          // Incoming faces (source terms)
          if (cell_face_orientations_[c][f] == FaceOrientation::INCOMING)
          {
            if (not cell.faces[f].has_neighbor)
              continue;

            size_t neighbor_id = cell.faces[f].GetNeighborLocalID(grid_.get());

            // Near-source/bulk region interface
            if (cell_regions_[neighbor_id] == CellRegion::NEAR_SOURCE)
            {
              continue;
            }

            // Bulk region cell neighbor
            else
            {
              size_t f_ = cell.faces[f].GetNeighborAdjacentFaceIndex(grid_.get());

              const Cell& neighbor = grid_->GetLocalCell(neighbor_id);
              const auto& neighbor_mapping = sdm.GetCellMapping(neighbor);
              const size_t neighbor_num_face_nodes = neighbor_mapping.GetNumFaceNodes(f_);

              for (size_t fi = 0; fi < num_face_nodes; ++fi)
              {
                const int i = cell_mapping.MapFaceNode(f, fi);

                for (size_t fj = 0; fj < num_face_nodes; ++fj)
                {
                  const int j = cell_mapping.MapFaceNode(f, fj);

                  int k = -1;

                  // Match the current cell's face node to the neighbor's
                  // local node index by global vertex id. Face-node ordering
                  // is not reliable across cells on unstructured meshes, and
                  // the neighboring face can have its own local orientation.
                  // Failing to find a match is a mesh/connectivity error;
                  // using a default or stale index here would inject the wrong
                  // upwind DOF into the sweep.
                  for (size_t fk = 0; fk < neighbor_num_face_nodes; ++fk)
                  {
                    const int kn = neighbor_mapping.MapFaceNode(f_, fk);
                    if (neighbor.vertex_ids[kn] == cell.vertex_ids[j])
                    {
                      k = kn;
                      break;
                    }
                  }
                  OpenSnLogicalErrorIf(
                    k < 0,
                    "Failed to map incoming uncollided sweep face node from cell " +
                      std::to_string(cell.global_id) + " face " + std::to_string(f) +
                      " to neighbor cell " + std::to_string(neighbor.global_id) + " face " +
                      std::to_string(f_) + ".");

                  const auto jr = sdm.MapDOFLocal(neighbor, k);
                  const double phi_j = destination_phi_[jr * num_groups_ + g];
                  phi(i) -= surface_matrix(i, j) * phi_j;
                }
              }
            }
          }

          // Outgoing faces (coefficient matrix)
          if (cell_face_orientations_[c][f] == FaceOrientation::OUTGOING)
          {
            for (size_t fi = 0; fi < num_face_nodes; ++fi)
            {
              const int i = cell_mapping.MapFaceNode(f, fi);

              for (size_t fj = 0; fj < num_face_nodes; ++fj)
              {
                const int j = cell_mapping.MapFaceNode(f, fj);
                Amat(i, j) += surface_matrix(i, j);
              }
            }
          }
        }

        // Construct and solve linear system
        for (size_t i = 0; i < cell_num_nodes; ++i)
          for (size_t j = 0; j < cell_num_nodes; ++j)
            Atemp(i, j) = Amat(i, j) + sigma_t[g] * mass_matrix(i, j);

        // Solve system
        const Vector<double> rhs = phi;
        GaussElimination(Atemp, phi, static_cast<int>(cell_num_nodes));

        // Update flux solution
        for (size_t i = 0; i < cell_num_nodes; ++i)
        {
          const auto ir = sdm.MapDOFLocal(cell, i);
          destination_phi_[ir * num_groups_ + g] = phi(i);
        }

        if (write_diagnostics_)
        {
          // The basis functions sum to one, so the equations of the cell sum to its balance:
          // inflow = outflow + removal. Report what is left of it, relative to the inflow.
          double inflow = 0.0;
          double residual = 0.0;
          for (size_t i = 0; i < cell_num_nodes; ++i)
          {
            inflow += rhs(i);
            residual += rhs(i);
            for (size_t j = 0; j < cell_num_nodes; ++j)
              residual -= (Amat(i, j) + sigma_t[g] * mass_matrix(i, j)) * phi(j);
          }
          bulk_residuals[bulk_index * num_groups_ + g] =
            inflow != 0.0 ? std::abs(residual / inflow) : 0.0;
        }
      }
    }
  };

  // An exception escaping a worker thread would terminate the process; rethrow it after the join.
  std::exception_ptr worker_exception;
  std::mutex exception_mutex;
  const auto SweepGroupsCapturingExceptions = [&](const size_t thread_id)
  {
    try
    {
      SweepGroups(thread_id);
    }
    catch (...)
    {
      std::scoped_lock lock(exception_mutex);
      if (not worker_exception)
        worker_exception = std::current_exception();
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(num_group_threads > 0 ? num_group_threads - 1 : 0);
  for (size_t thread_id = 1; thread_id < num_group_threads; ++thread_id)
    workers.emplace_back(SweepGroupsCapturingExceptions, thread_id);

  SweepGroupsCapturingExceptions(0);

  for (auto& worker : workers)
    worker.join();
  if (worker_exception)
    std::rethrow_exception(worker_exception);

  if (write_diagnostics_)
  {
    // RaytraceNearSourceRegion has added the entry of this source.
    auto& diagnostics = source_diagnostics_.back();
    for (const size_t c : bulk_spls_)
      diagnostics.bulk_cell_ids.push_back(grid_->GetLocalCell(c).global_id);
    diagnostics.bulk_residuals = std::move(bulk_residuals);
  }
}

void
UncollidedProblem::WriteDiagnostics(const std::string& file_name) const
{
  RunOnRoot(
    [this, &file_name]()
    {
      const auto raw_file = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      OpenSnInvalidArgumentIf(raw_file < 0,
                              GetName() + ": failed to create uncollided diagnostics file \"" +
                                file_name + "\".");
      const H5FileHandle file_handle(raw_file);
      const auto file = file_handle.Id();

      bool written = H5CreateAttribute<unsigned int>(file, "num groups", num_groups_) and
                     H5CreateAttribute<uint64_t>(file, "num sources", source_diagnostics_.size());
      for (size_t source = 0; source < source_diagnostics_.size() and written; ++source)
      {
        const auto& diagnostics = source_diagnostics_[source];
        const std::string suffix = " " + std::to_string(source);
        written =
          H5WriteDataset1D<uint64_t>(file, "cell ids" + suffix, diagnostics.cell_ids) and
          H5WriteDataset1D<double>(file, "alpha" + suffix, diagnostics.alphas) and
          H5WriteDataset1D<double>(file, "balance residual" + suffix, diagnostics.residuals) and
          H5WriteDataset1D<uint64_t>(file, "bulk cell ids" + suffix, diagnostics.bulk_cell_ids) and
          H5WriteDataset1D<double>(
            file, "bulk balance residual" + suffix, diagnostics.bulk_residuals);
      }
      OpenSnLogicalErrorIf(not written,
                           GetName() + ": failed to write the uncollided diagnostics file.");
    });
}

UncollidedMatrices
UncollidedProblem::ComputeUncollidedIntegrals(const Cell& cell, const Vector3& pt_loc)
{
  const auto& sdm = *discretization_;

  // Cell mapping
  auto coord_sys = grid_->GetCoordinateSystem();
  auto swf = SpatialWeightFunction::FromCoordinateType(coord_sys);
  const auto& cell_mapping = sdm.GetCellMapping(cell);
  const size_t cell_num_faces = cell.faces.size();
  const size_t cell_num_nodes = cell_mapping.GetNumNodes();
  const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();

  // Matrices
  DenseMatrix<double> IntV_shapeJ_omega_gradshapeI(cell_num_nodes, cell_num_nodes, 0.);
  std::vector<DenseMatrix<double>> IntS_omega_n_shapeI_shapeJ(cell_num_faces);

  // Gradient Matrix
  for (unsigned int i = 0; i < cell_num_nodes; ++i)
  {
    for (unsigned int j = 0; j < cell_num_nodes; ++j)
    {
      for (const auto& qp : fe_vol_data.GetQuadraturePointIndices())
      {
        const Vector3& qp_xyz = fe_vol_data.QPointXYZ(qp);
        Vector3 omega = ComputeOmega(pt_loc, qp_xyz);

        IntV_shapeJ_omega_gradshapeI(i, j) -= (*swf)(qp_xyz)*fe_vol_data.ShapeValue(j, qp) *
                                              omega.Dot(fe_vol_data.ShapeGrad(i, qp)) *
                                              fe_vol_data.JxW(qp);
      } // for qp
    } // for j
  } // for i

  // Surface matrices
  for (size_t f = 0; f < cell_num_faces; ++f)
  {
    const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);
    IntS_omega_n_shapeI_shapeJ[f] = DenseMatrix<double>(cell_num_nodes, cell_num_nodes, 0.0);

    for (unsigned int i = 0; i < cell_num_nodes; ++i)
    {
      for (unsigned int j = 0; j < cell_num_nodes; ++j)
      {
        for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
        {
          const Vector3& qp_xyz = fe_srf_data.QPointXYZ(qp);
          Vector3 omega = ComputeOmega(pt_loc, qp_xyz);

          IntS_omega_n_shapeI_shapeJ[f](i, j) +=
            (*swf)(qp_xyz)*omega.Dot(cell.faces[f].normal) * fe_srf_data.ShapeValue(i, qp) *
            fe_srf_data.ShapeValue(j, qp) * fe_srf_data.JxW(qp);

        } // for qp
      } // for j
    } // for i
  } // for f

  return UncollidedMatrices{IntV_shapeJ_omega_gradshapeI, IntS_omega_n_shapeI_shapeJ};
}

void
UncollidedProblem::UpdateBalance(const SourcePoint& source_point)
{
  CALI_CXX_MARK_SCOPE("UpdateBalance");

  const auto& sdm = *discretization_;

  // Point source data
  const Vector3& pt_loc = source_point.location;
  const std::vector<double>& strength = source_point.strength;

  // Each rank holds its share of the balance; the source is counted once.
  if (opensn::mpi_comm.rank() == 0)
    for (size_t g = 0; g < num_groups_; ++g)
      production_ += strength[g];

  for (const auto& cell : grid_->GetLocalCells())
  {
    const uint64_t c = cell->local_id;
    const auto& cell_mapping = sdm.GetCellMapping(*cell);
    const size_t cell_num_nodes = cell_mapping.GetNumNodes();
    const auto& sigma_t = cell_transport_views_[c].GetXS().GetSigmaTotal();
    const auto& intV_shapeI = unit_cell_matrices_[c].intV_shapeI;

    // Removal rate in cell
    for (size_t g = 0; g < num_groups_; ++g)
      for (size_t i = 0; i < cell_num_nodes; ++i)
      {
        const auto ir = sdm.MapDOFLocal(*cell, i);
        physical_removal_ += sigma_t[g] * destination_phi_[ir * num_groups_ + g] * intV_shapeI(i);
      }
  }

  const auto swf = SpatialWeightFunction::FromCoordinateType(grid_->GetCoordinateSystem());
  for (const size_t c : bulk_spls_)
  {
    const auto& cell = grid_->GetLocalCell(c);
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t cell_num_nodes = cell_mapping.GetNumNodes();
    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      if (face.has_neighbor or IsReflectingBoundary(face.neighbor_id) or
          cell_face_orientations_[c][f] != FaceOrientation::OUTGOING)
        continue;

      const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);
      for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
      {
        const auto& qp_xyz = fe_srf_data.QPointXYZ(qp);
        const auto omega = ComputeOmega(pt_loc, qp_xyz);
        const double integrand = (*swf)(qp_xyz)*omega.Dot(face.normal) * fe_srf_data.JxW(qp);

        for (size_t g = 0; g < num_groups_; ++g)
          for (size_t i = 0; i < cell_num_nodes; ++i)
          {
            const auto ir = sdm.MapDOFLocal(cell, i);
            out_flow_ +=
              destination_phi_[ir * num_groups_ + g] * integrand * fe_srf_data.ShapeValue(i, qp);
          }
      }
    }
  }
}

void
UncollidedProblem::AccumulateMoments(const Vector3& pt_loc)
{
  CALI_CXX_MARK_SCOPE("AccumulateMoments");

  const auto& sdm = *discretization_;
  const auto swf = SpatialWeightFunction::FromCoordinateType(grid_->GetCoordinateSystem());

  for (const auto& cell : grid_->GetLocalCells())
  {
    const auto& cell_mapping = sdm.GetCellMapping(*cell);
    const size_t cell_num_nodes = cell_mapping.GetNumNodes();
    const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();
    std::vector<std::vector<Vector<double>>> moment_rhs(
      moments_.size(),
      std::vector<Vector<double>>(num_groups_, Vector<double>(cell_num_nodes, 0.0)));

    for (const auto& qp : fe_vol_data.GetQuadraturePointIndices())
    {
      const auto& qp_xyz = fe_vol_data.QPointXYZ(qp);
      const auto omega = ComputeOmega(pt_loc, qp_xyz);
      OpenSnLogicalErrorIf(omega.Norm() == 0.0,
                           GetName() + ": point source coincides with a volume quadrature point.");
      const double theta = std::acos(std::clamp(omega.z, -1.0, 1.0));
      const double varphi = std::atan2(omega.y, omega.x);
      const double weighted_jacobian = (*swf)(qp_xyz)*fe_vol_data.JxW(qp);

      std::vector<double> phi_qp(num_groups_, 0.0);
      for (size_t j = 0; j < cell_num_nodes; ++j)
      {
        const auto jr = sdm.MapDOFLocal(*cell, j);
        const double shape = fe_vol_data.ShapeValue(j, qp);
        for (size_t g = 0; g < num_groups_; ++g)
          phi_qp[g] += shape * destination_phi_[jr * num_groups_ + g];
      }

      for (size_t moment_index = 0; moment_index < moments_.size(); ++moment_index)
      {
        const auto& moment = moments_[moment_index];
        const double harmonic = Ylm(moment.ell, moment.m, varphi, theta);
        for (size_t i = 0; i < cell_num_nodes; ++i)
        {
          const double weight = fe_vol_data.ShapeValue(i, qp) * harmonic * weighted_jacobian;
          for (size_t g = 0; g < num_groups_; ++g)
            moment_rhs[moment_index][g](i) += phi_qp[g] * weight;
        }
      }
    }

    for (size_t moment_index = 0; moment_index < moments_.size(); ++moment_index)
      for (size_t g = 0; g < num_groups_; ++g)
      {
        auto mass_matrix = unit_cell_matrices_[cell->local_id].intV_shapeI_shapeJ;
        GaussElimination(
          mass_matrix, moment_rhs[moment_index][g], static_cast<int>(cell_num_nodes));
        for (size_t i = 0; i < cell_num_nodes; ++i)
        {
          const auto ir = sdm.MapDOFLocal(*cell, i);
          accumulated_moments_[moment_index][ir * num_groups_ + g] +=
            moment_rhs[moment_index][g](i);
        }
      }
  }
}

void
UncollidedProblem::WriteFluxMoments(hid_t file)
{
  CALI_CXX_MARK_SCOPE("WriteFluxMoments");

  // One dataset at a time, so that rank 0 holds one gathered moment at a time.
  const auto scalar_flux = GatherOnRoot(phi_new_local_);
  RunOnRoot(
    [this, file, &scalar_flux]()
    {
      OpenSnLogicalErrorIf(not H5WriteDataset1D<double>(file, "0,0", scalar_flux),
                           GetName() + ": failed to write uncollided scalar flux.");
    });
  for (size_t i = 0; i < moment_names_.size(); ++i)
  {
    const auto moment = GatherOnRoot(accumulated_moments_[i]);
    RunOnRoot(
      [this, file, i, &moment]()
      {
        OpenSnLogicalErrorIf(not H5WriteDataset1D<double>(file, moment_names_[i], moment),
                             GetName() + ": failed to write uncollided moment \"" +
                               moment_names_[i] + "\".");
      });
  }
}

void
UncollidedProblem::FinalizeBalance(hid_t file)
{
  CALI_CXX_MARK_SCOPE("FinalizeBalance");

  // Every rank holds its share of the removal and of the outflow.
  if (opensn::mpi_comm.size() > 1)
    for (double* rate : {&production_, &physical_removal_, &reflected_removal_, &out_flow_})
      opensn::mpi_comm.all_reduce(*rate, mpi::op::sum<double>());

  const double correction_scale = std::max(1.0, production_);
  OpenSnLogicalErrorIf(
    physical_removal_ > production_ + 1.0e-10 * correction_scale,
    GetName() + ": physical uncollided removal (" + std::to_string(physical_removal_) +
      ") exceeds the physical source rate (" + std::to_string(production_) + ").");

  const double total_removal = physical_removal_ + reflected_removal_;
  const double balance = production_ - (total_removal + out_flow_);
  const double conservation_error = (production_ == 0.0) ? 0.0 : (balance / production_);

  constexpr double balance_warning_tolerance = 1.0e-3;
  if (std::abs(balance) > balance_warning_tolerance * correction_scale)
    log.Log0Warning() << GetName() << ": uncollided balance residual is " << std::setprecision(6)
                      << std::scientific << balance << ".";

  log.Log() << "\nBalance table:\n"
            << std::setprecision(6) << std::scientific
            << " Removal rate                = " << total_removal << "\n"
            << " Production rate             = " << production_ << "\n"
            << " Out-flow rate               = " << out_flow_ << "\n"
            << " Balance (Production - Loss) = " << balance << "\n"
            << " Conservation error          = " << conservation_error << "\n\n";

  // Write balance parameters to h5
  RunOnRoot(
    [this, file, total_removal]()
    {
      OpenSnLogicalErrorIf(not H5CreateAttribute<double>(file, "production", production_),
                           GetName() + ": failed to write uncollided production rate.");
      OpenSnLogicalErrorIf(not H5CreateAttribute<double>(file, "removal", total_removal),
                           GetName() + ": failed to write uncollided removal rate.");
      OpenSnLogicalErrorIf(not H5CreateAttribute<double>(file, "out-flow", out_flow_),
                           GetName() + ": failed to write uncollided outflow rate.");
    });
}

} // namespace opensn
