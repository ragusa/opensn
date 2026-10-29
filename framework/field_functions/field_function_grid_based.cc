// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/field_functions/field_function_grid_based.h"
#include "framework/math/spatial_discretization/cell_mappings/cell_mapping.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_continuous.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_discontinuous.h"
#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/piecewise_quadratic_polyhedron_mapping.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/mesh/grid_vtk_utils.h"
#include "framework/logging/log.h"
#include "framework/utils/error.h"
#include "framework/runtime.h"
#include <array>
#include <limits>
#include <petsc.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkIntArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>

namespace opensn
{

InputParameters
FieldFunctionGridBased::GetInputParameters()
{
  InputParameters params = FieldFunction::GetInputParameters();

  params.AddOptionalParameter(
    "discretization", "PWLC", "The spatial discretization type to be used");
  params.AddOptionalParameter(
    "coord_sys", "cartesian", "Coordinate system to apply to element mappings");
  params.AddOptionalParameter("quadrature_order",
                              0,
                              "If supplied, will overwrite the default for the "
                              "specific discretization-coordinate system combination.");

  params.AddOptionalParameter(
    "initial_value", 0.0, "The initial value to assign to the field function");

  params.ConstrainParameterRange("discretization", AllowableRangeList::New({"PWLC", "PWLD"}));
  params.ConstrainParameterRange(
    "coord_sys", AllowableRangeList::New({"cartesian", "cylindrical", "spherical"}));

  return params;
}

FieldFunctionGridBased::FieldFunctionGridBased(const InputParameters& params)
  : FieldFunction(params),
    discretization_(MakeSpatialDiscretization(params)),
    ghosted_field_vector_(MakeFieldVector(*discretization_, GetUnknownManager())),
    local_grid_bounding_box_(params.GetSharedPtrParam<Mesh>("mesh")->GetLocalBoundingBox())
{
  ghosted_field_vector_->Set(params.GetParamValue<double>("initial_value"));
}

FieldFunctionGridBased::FieldFunctionGridBased(
  std::string name, std::shared_ptr<SpatialDiscretization>& discretization_ptr, Unknown unknown)
  : FieldFunction(std::move(name), std::move(unknown)),
    discretization_(discretization_ptr),
    ghosted_field_vector_(MakeFieldVector(*discretization_, GetUnknownManager())),
    local_grid_bounding_box_(discretization_->GetGrid()->GetLocalBoundingBox())
{
}

FieldFunctionGridBased::FieldFunctionGridBased(
  std::string name,
  std::shared_ptr<SpatialDiscretization>& discretization_ptr,
  Unknown unknown,
  const std::vector<double>& field_vector)
  : FieldFunction(std::move(name), std::move(unknown)),
    discretization_(discretization_ptr),
    ghosted_field_vector_(MakeFieldVector(*discretization_, GetUnknownManager())),
    local_grid_bounding_box_(discretization_->GetGrid()->GetLocalBoundingBox())
{
  OpenSnInvalidArgumentIf(field_vector.size() != ghosted_field_vector_->GetLocalSize(),
                          "Constructor called with incompatible size field vector.");

  ghosted_field_vector_->Set(field_vector);
}

FieldFunctionGridBased::FieldFunctionGridBased(
  std::string name,
  std::shared_ptr<SpatialDiscretization>& discretization_ptr,
  Unknown unknown,
  double field_value)
  : FieldFunction(std::move(name), std::move(unknown)),
    discretization_(discretization_ptr),
    ghosted_field_vector_(MakeFieldVector(*discretization_, GetUnknownManager())),
    local_grid_bounding_box_(discretization_->GetGrid()->GetLocalBoundingBox())
{
  ghosted_field_vector_->Set(field_value);
}

const SpatialDiscretization&
FieldFunctionGridBased::GetSpatialDiscretization() const
{
  return *discretization_;
}

std::vector<double>&
FieldFunctionGridBased::GetLocalFieldVector()
{
  return ghosted_field_vector_->GetLocalSTLData();
}

const std::vector<double>&
FieldFunctionGridBased::GetLocalFieldVector() const
{
  return ghosted_field_vector_->GetLocalSTLData();
}

std::vector<double>
FieldFunctionGridBased::GetGhostedFieldVector() const
{
  return ghosted_field_vector_->GetLocalSTLData();
}

void
FieldFunctionGridBased::UpdateFieldVector(const std::vector<double>& field_vector)
{
  OpenSnInvalidArgumentIf(field_vector.size() < ghosted_field_vector_->GetLocalSize(),
                          "Attempted update with a vector of insufficient size.");

  ghosted_field_vector_->Set(field_vector);
  ghosted_field_vector_->CommunicateGhostEntries();
}

void
FieldFunctionGridBased::UpdateFieldVector(const Vec& field_vector)
{
  ghosted_field_vector_->CopyLocalValues(field_vector);

  ghosted_field_vector_->CommunicateGhostEntries();
}

bool
FieldFunctionGridBased::CanUpdate() const
{
  return static_cast<bool>(update_callback_) and
         (not can_update_callback_ or can_update_callback_());
}

void
FieldFunctionGridBased::Update()
{
  OpenSnLogicalErrorIf(not CanUpdate(),
                       "Field function \"" + GetName() + "\" is not currently updateable.");
  update_callback_(*this);
}

void
FieldFunctionGridBased::SetUpdateCallback(UpdateCallback callback,
                                          CanUpdateCallback can_update_callback)
{
  update_callback_ = std::move(callback);
  can_update_callback_ = std::move(can_update_callback);
}

std::vector<double>
FieldFunctionGridBased::GetPointValue(const Vector3& point) const
{
  const auto& uk_man = GetUnknownManager();
  const size_t num_components = uk_man.GetTotalUnknownStructureSize();

  size_t local_num_point_hits = 0;
  std::vector<double> local_point_value(num_components, 0.0);

  const auto& xyz_min = local_grid_bounding_box_.first;
  const auto& xyz_max = local_grid_bounding_box_.second;

  const auto xmin = xyz_min.x;
  const auto ymin = xyz_min.y;
  const auto zmin = xyz_min.z;

  const auto xmax = xyz_max.x;
  const auto ymax = xyz_max.y;
  const auto zmax = xyz_max.z;

  const auto& field_vector = *ghosted_field_vector_;

  if (point.x >= xmin and point.x <= xmax and point.y >= ymin and point.y <= ymax and
      point.z >= zmin and point.z <= zmax)
  {
    const auto& grid = discretization_->GetGrid();
    for (const auto& cell : grid->GetLocalCells())
    {
      if (grid->CheckPointInsideCell(*cell, point))
      {
        const auto& cell_mapping = discretization_->GetCellMapping(*cell);
        Vector<double> shape_values;
        cell_mapping.ShapeValues(point, shape_values);

        local_num_point_hits += 1;

        const auto num_nodes = cell_mapping.GetNumNodes();
        for (size_t c = 0; c < num_components; ++c)
        {
          for (size_t j = 0; j < num_nodes; ++j)
          {
            const auto dof_map = discretization_->MapDOFLocal(*cell, j, uk_man, 0, c);
            const double dof_value = field_vector[dof_map];

            local_point_value[c] += dof_value * shape_values(j);
          } // for node i
        } // for component c
      } // if inside cell
    } // for cell
  } // if in bounding box

  // Communicate number of point hits
  size_t global_num_point_hits = 0;
  mpi_comm.all_reduce(local_num_point_hits, global_num_point_hits, mpi::op::sum<size_t>());
  OpenSnLogicalErrorIf(global_num_point_hits == 0,
                       "FieldFunctionGridBased::GetPointValue: No cell identified containing "
                       "the specified point.");

  std::vector<double> global_point_value(num_components, 0.0);
  OpenSnLogicalErrorIf(num_components > static_cast<size_t>(std::numeric_limits<int>::max()),
                       "FieldFunctionGridBased::GetPointValue: Number of components exceeds "
                       "MPI reduction count limit.");
  const auto point_value_size = static_cast<int>(num_components);
  mpi_comm.all_reduce(
    local_point_value.data(), point_value_size, global_point_value.data(), mpi::op::sum<double>());

  Scale(global_point_value, 1.0 / static_cast<double>(global_num_point_hits));

  return global_point_value;
}

double
FieldFunctionGridBased::Evaluate(const Cell& cell, const Vector3& position, int component) const
{
  const auto& field_vector = *ghosted_field_vector_;

  const auto& cell_mapping = discretization_->GetCellMapping(cell);

  Vector<double> shape_values;
  cell_mapping.ShapeValues(position, shape_values);

  double value = 0.0;
  const size_t num_nodes = cell_mapping.GetNumNodes();
  for (size_t j = 0; j < num_nodes; ++j)
  {
    const auto dof_map = discretization_->MapDOFLocal(cell, j, GetUnknownManager(), 0, component);
    value += field_vector[dof_map] * shape_values(j);
  }

  return value;
}

namespace
{

/**
 * For the quadratic (PWQD) export every polyhedron is written as its side tetrahedra, each a
 * VTK_QUADRATIC_TETRA whose 10 nodes carry the values of the PWQD field at the sub-mesh nodes
 * (vertices, edge midpoints, and the tied face and cell nodes obtained from the tying operator),
 * so the exported field is exactly the quadratic finite element field. Returns, for the
 * reference P2 tetrahedron, the reference node index of each VTK node: corners 0..3, then the
 * midpoints of the edges (0,1), (1,2), (0,2), (0,3), (1,3), (2,3).
 */
std::array<size_t, 10>
VtkQuadraticTetraNodeOrder(const TetLagrangeReference& ref)
{
  const auto& nodes = ref.GetNodes();
  std::array<size_t, 10> order{};
  auto find = [&nodes](unsigned int a, unsigned int b)
  {
    for (size_t j = 0; j < nodes.size(); ++j)
      if ((a == b and nodes[j][a] == 2) or (a != b and nodes[j][a] == 1 and nodes[j][b] == 1))
        return j;
    throw std::logic_error("VtkQuadraticTetraNodeOrder: node not found");
  };
  for (unsigned int c = 0; c < 4; ++c)
    order[c] = find(c, c);
  const std::array<std::pair<unsigned int, unsigned int>, 6> edges = {
    std::pair{0u, 1u}, {1u, 2u}, {0u, 2u}, {0u, 3u}, {1u, 3u}, {2u, 3u}};
  for (size_t e = 0; e < 6; ++e)
    order[4 + e] = find(edges[e].first, edges[e].second);
  return order;
}

const PieceWiseQuadraticPolyhedronMapping&
QuadraticMapping(const SpatialDiscretization& sdm, const Cell& cell)
{
  const auto* mapping =
    dynamic_cast<const PieceWiseQuadraticPolyhedronMapping*>(&sdm.GetCellMapping(cell));
  if (mapping == nullptr)
    throw std::logic_error("FieldFunctionGridBased: quadratic export requires PWQD cell mappings");
  return *mapping;
}

/// The side-tetrahedron sub-mesh of all local cells as VTK quadratic tetrahedra.
vtkNew<vtkUnstructuredGrid>
PrepareVtkQuadraticSubmeshGrid(const Mesh& grid,
                               const SpatialDiscretization& sdm,
                               const std::array<size_t, 10>& order)
{
  vtkNew<vtkUnstructuredGrid> ugrid;
  vtkNew<vtkPoints> points;
  vtkNew<vtkIntArray> block_array;
  vtkNew<vtkIntArray> partition_id_array;
  points->SetDataType(VTK_DOUBLE);
  block_array->SetName("Block");
  partition_id_array->SetName("Partition");

  vtkIdType node_counter = 0;
  for (const auto& cell : grid.GetLocalCells())
  {
    const auto& tying = QuadraticMapping(sdm, *cell).GetTying();
    const auto& X = tying.GetSubNodeLocations();
    for (size_t s = 0; s < tying.GetNumSides(); ++s)
    {
      const auto& side_nodes = tying.GetSideNodes(s);
      std::array<vtkIdType, 10> ids{};
      for (size_t k = 0; k < 10; ++k)
      {
        const auto& x = X[side_nodes[order[k]]];
        points->InsertPoint(node_counter, x.x, x.y, x.z);
        ids[k] = node_counter++;
      }
      ugrid->InsertNextCell(VTK_QUADRATIC_TETRA, 10, ids.data());
      block_array->InsertNextValue(static_cast<int>(cell->block_id));
      partition_id_array->InsertNextValue(cell->partition_id);
    }
  }
  ugrid->SetPoints(points);
  ugrid->GetCellData()->AddArray(block_array);
  ugrid->GetCellData()->AddArray(partition_id_array);
  return ugrid;
}

} // namespace

void
FieldFunctionGridBased::ExportMultipleToPVTU(
  const std::string& file_base_name,
  const std::vector<std::shared_ptr<const FieldFunctionGridBased>>& ff_list)
{
  const std::string fname = "FieldFunctionGridBased::ExportMultipleToPVTU";
  log.Log() << "Exporting field functions to PVTU with file base \"" << file_base_name << "\"";

  if (ff_list.empty())
    throw std::logic_error(fname + ": Cannot be used with empty field-function"
                                   " list");

  // Setup master and check slaves
  const auto& master_ff_ptr = ff_list.front();
  const auto& master_ff = *master_ff_ptr;

  for (const auto& ff_ptr : ff_list)
    if (ff_ptr != master_ff_ptr)
      if (ff_ptr->discretization_->GetGrid() != master_ff_ptr->discretization_->GetGrid())
        throw std::logic_error(fname +
                               ": Cannot be used with field functions based on different grids.");

  // Get grid
  const auto& grid = master_ff.discretization_->GetGrid();

  // Quadratic discretizations are exported on their side-tetrahedron sub-mesh
  const bool quadratic = master_ff.discretization_->GetType() ==
                         SpatialDiscretizationType::PIECEWISE_QUADRATIC_DISCONTINUOUS;
  if (quadratic)
    for (const auto& ff_ptr : ff_list)
      if (ff_ptr->discretization_->GetType() !=
          SpatialDiscretizationType::PIECEWISE_QUADRATIC_DISCONTINUOUS)
        throw std::logic_error(fname + ": Cannot mix quadratic and non-quadratic field functions.");

  // VTK node order of the P2 reference tetrahedron (the same for every PWQD cell)
  const auto vtk_order = VtkQuadraticTetraNodeOrder(TetLagrangeReference(2));
  auto ugrid = quadratic
                 ? PrepareVtkQuadraticSubmeshGrid(*grid, *master_ff.discretization_, vtk_order)
                 : PrepareVtkUnstructuredGrid(grid);

  // Upload cell/point data
  auto* point_data = ugrid->GetPointData();
  for (const auto& ff_ptr : ff_list)
  {
    const auto field_vector = ff_ptr->GetGhostedFieldVector();

    const auto& uk_man = ff_ptr->GetUnknownManager();
    const auto& unknown = ff_ptr->GetUnknown();
    const auto& sdm = ff_ptr->discretization_;
    const size_t num_comps = unknown.GetNumComponents();

    for (uint c = 0; c < num_comps; ++c)
    {
      std::string component_name = ff_ptr->GetName() + unknown.name;
      if (num_comps > 1)
        component_name += unknown.component_names[c];

      vtkNew<vtkDoubleArray> point_array;

      point_array->SetName(component_name.c_str());

      // Populate the array here
      for (const auto& cell : grid->GetLocalCells())
      {
        const size_t num_nodes = sdm->GetCellNumNodes(*cell);

        if (quadratic)
        {
          // Values at the sub-mesh nodes: T times the cell's nodal values
          const auto& tying = QuadraticMapping(*sdm, *cell).GetTying();
          const auto& T = tying.GetTyingOperator();
          if (tying.GetDegree() != 2)
            throw std::logic_error(fname + ": quadratic export expects degree-2 cell mappings.");
          std::vector<double> nodal(num_nodes, 0.0);
          for (size_t n = 0; n < num_nodes; ++n)
            nodal[n] = field_vector[sdm->MapDOFLocal(*cell, n, uk_man, 0, c)];
          std::vector<double> sub(tying.GetNumSubNodes(), 0.0);
          for (size_t i = 0; i < sub.size(); ++i)
            for (size_t n = 0; n < num_nodes; ++n)
              sub[i] += T(i, n) * nodal[n];
          for (size_t s = 0; s < tying.GetNumSides(); ++s)
          {
            const auto& side_nodes = tying.GetSideNodes(s);
            for (size_t k = 0; k < 10; ++k)
              point_array->InsertNextValue(sub[side_nodes[vtk_order[k]]]);
          }
          continue;
        }

        const size_t num_verts = cell->vertex_ids.size();
        // Discretizations whose first nodes are the cell vertices (PWLD, PWQD) export the vertex
        // nodal values; any other node layout exports the nodal average at every vertex.
        bool vertex_nodes_first = num_nodes >= num_verts;
        if (vertex_nodes_first and num_nodes != num_verts)
        {
          const auto& node_locations = sdm->GetCellMapping(*cell).GetNodeLocations();
          for (size_t v = 0; v < num_verts and vertex_nodes_first; ++v)
            vertex_nodes_first =
              (node_locations[v] - grid->GlobalVertex(cell->vertex_ids[v])).NormSquare() < 1.0e-24;
        }

        if (vertex_nodes_first)
        {
          for (size_t n = 0; n < num_verts; ++n)
          {
            const auto nmap = sdm->MapDOFLocal(*cell, n, uk_man, 0, c);

            const double field_value = field_vector[nmap];

            point_array->InsertNextValue(field_value);
          } // for node
        }
        else
        {
          double node_average = 0.0;
          for (size_t n = 0; n < num_nodes; ++n)
          {
            const auto nmap = sdm->MapDOFLocal(*cell, n, uk_man, 0, c);

            const double field_value = field_vector[nmap];
            node_average += field_value;
          } // for node
          node_average /= static_cast<double>(num_nodes);
          for (std::size_t n = 0; n < cell->vertex_ids.size(); ++n)
          {
            point_array->InsertNextValue(node_average);
          } // for vertex
        }

      } // for cell

      point_data->AddArray(point_array);
    } // for component
  } // for ff_ptr

  WritePVTUFiles(ugrid, file_base_name);

  log.Log() << "Done exporting field functions to PVTU.";
  opensn::mpi_comm.barrier();
}

std::shared_ptr<SpatialDiscretization>
FieldFunctionGridBased::MakeSpatialDiscretization(const InputParameters& params)
{
  const auto grid = params.GetSharedPtrParam<Mesh>("mesh");
  const auto sdm_type = params.GetParamValue<std::string>("discretization");

  std::string cs = "cartesian";
  if (params.IsParameterValid("coord_sys"))
    cs = params.GetParamValue<std::string>("coord_sys");

  QuadratureOrder q_order = QuadratureOrder::SECOND;

  if (params.IsParameterValid("quadrature_order"))
  {
    const auto max_order = static_cast<uint32_t>(QuadratureOrder::FORTYTHIRD);
    const auto q_order_int = params.GetParamValue<uint32_t>("quadrature_order");
    OpenSnInvalidArgumentIf(q_order_int > max_order,
                            "Invalid quadrature point order " + std::to_string(q_order_int));
    q_order = static_cast<QuadratureOrder>(q_order_int);
  }
  else // Defaulted
  {
    if (cs == "cartesian")
      q_order = QuadratureOrder::SECOND;
    else if (cs == "cylindrical")
      q_order = QuadratureOrder::THIRD;
    else if (cs == "spherical")
      q_order = QuadratureOrder::FOURTH;
    else
      throw std::logic_error("Unrecognized coordinate system.");
  }

  if (sdm_type == "PWLC")
    return PieceWiseLinearContinuous::New(grid, q_order);
  else if (sdm_type == "PWLD")
    return PieceWiseLinearDiscontinuous::New(grid, q_order);

  // If not returned by now
  OpenSnInvalidArgument("Unsupported discretization \"" + sdm_type + "\"");
}

std::unique_ptr<GhostedParallelSTLVector>
FieldFunctionGridBased::MakeFieldVector(const SpatialDiscretization& discretization,
                                        const UnknownManager& uk_man)
{
  auto field = std::make_unique<GhostedParallelSTLVector>(discretization.GetNumLocalDOFs(uk_man),
                                                          discretization.GetNumGlobalDOFs(uk_man),
                                                          discretization.GetGhostDOFIndices(uk_man),
                                                          mpi_comm);

  return field;
}

} // namespace opensn
