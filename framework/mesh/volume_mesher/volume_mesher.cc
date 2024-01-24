#include "framework/mesh/volume_mesher/volume_mesher.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/mesh/mesh_handler/mesh_handler.h"
#include "framework/mesh/unpartitioned_mesh/unpartitioned_mesh.h"
#include "framework/mesh/logical_volume/logical_volume.h"
#include "framework/utils/timer.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"
#include "framework/mpi/mpi.h"

namespace opensn
{

VolumeMesher::VolumeMesher(VolumeMesherType type) : type_(type)
{
}

void
VolumeMesher::SetContinuum(MeshContinuumPtr& grid)
{
  grid_ptr_ = grid;
}

MeshContinuumPtr&
VolumeMesher::GetContinuum()
{
  return grid_ptr_;
}

void
VolumeMesher::SetGridAttributes(MeshAttributes new_attribs, std::array<size_t, 3> ortho_Nis)
{
  grid_ptr_->SetAttributes(new_attribs, ortho_Nis);
}

VolumeMesherType
VolumeMesher::Type() const
{
  return type_;
}

void
VolumeMesher::SetMatIDFromLogical(const LogicalVolume& log_vol, bool sense, int mat_id)
{
  log.Log0Verbose1() << program_timer.GetTimeString()
                     << " Setting material id from logical volume.";
  // Get current mesh handler
  auto& handler = GetCurrentHandler();

  // Get back mesh
  MeshContinuumPtr vol_cont = handler.GetGrid();

  int num_cells_modified = 0;
  for (auto& cell : vol_cont->local_cells)
  {
    if (log_vol.Inside(cell.centroid_) && sense)
    {
      cell.material_id_ = mat_id;
      ++num_cells_modified;
    }
  }

  const auto& ghost_ids = vol_cont->cells.GetGhostGlobalIDs();
  for (uint64_t ghost_id : ghost_ids)
  {
    auto& cell = vol_cont->cells[ghost_id];
    if (log_vol.Inside(cell.centroid_) && sense) cell.material_id_ = mat_id;
  }

  int global_num_cells_modified;
  MPI_Allreduce(&num_cells_modified, &global_num_cells_modified, 1, MPI_INT, MPI_SUM, mpi.comm);

  log.Log0Verbose1() << program_timer.GetTimeString()
                     << " Done setting material id from logical volume. "
                     << "Number of cells modified = " << global_num_cells_modified << ".";
}

void
VolumeMesher::SetBndryIDFromLogical(const LogicalVolume& log_vol,
                                    bool sense,
                                    const std::string& bndry_name)
{
  log.Log() << program_timer.GetTimeString() << " Setting boundary id from logical volume.";
  // Get current mesh handler
  auto& handler = GetCurrentHandler();

  // Get back mesh
  MeshContinuumPtr vol_cont = handler.GetGrid();

  // Check if name already has id
  auto& grid_bndry_id_map = vol_cont->GetBoundaryIDMap();
  uint64_t bndry_id = vol_cont->MakeBoundaryID(bndry_name);

  // Loop over cells
  int num_faces_modified = 0;
  for (auto& cell : vol_cont->local_cells)
  {
    for (auto& face : cell.faces_)
    {
      if (face.has_neighbor_) continue;
      if (log_vol.Inside(face.centroid_) && sense)
      {
        face.neighbor_id_ = bndry_id;
        ++num_faces_modified;
      }
    }
  }

  int global_num_faces_modified;
  MPI_Allreduce(&num_faces_modified, &global_num_faces_modified, 1, MPI_INT, MPI_SUM, mpi.comm);

  if (global_num_faces_modified > 0 and grid_bndry_id_map.count(bndry_id) == 0)
    grid_bndry_id_map[bndry_id] = bndry_name;

  log.Log() << program_timer.GetTimeString() << " Done setting boundary id from logical volume. "
            << "Number of faces modified = " << global_num_faces_modified << ".";
}

void
VolumeMesher::SetMatIDToAll(int mat_id)
{
  log.Log() << program_timer.GetTimeString() << " Setting material id " << mat_id
            << " to all cells.";

  // Get current mesh handler
  auto& handler = GetCurrentHandler();

  // Get back mesh
  auto vol_cont = handler.GetGrid();

  for (auto& cell : vol_cont->local_cells)
    cell.material_id_ = mat_id;

  const auto& ghost_ids = vol_cont->cells.GetGhostGlobalIDs();
  for (uint64_t ghost_id : ghost_ids)
    vol_cont->cells[ghost_id].material_id_ = mat_id;

  opensn::mpi.Barrier();
  log.Log() << program_timer.GetTimeString() << " Done setting material id " << mat_id
            << " to all cells";
}

void
VolumeMesher::SetupOrthogonalBoundaries()
{
  log.Log() << program_timer.GetTimeString() << " Setting orthogonal boundaries.";

  // Get current mesh handler
  auto& handler = GetCurrentHandler();

  // Get back mesh
  auto vol_cont = handler.GetGrid();

  const Vector3 ihat(1.0, 0.0, 0.0);
  const Vector3 jhat(0.0, 1.0, 0.0);
  const Vector3 khat(0.0, 0.0, 1.0);

  for (auto& cell : vol_cont->local_cells)
    for (auto& face : cell.faces_)
      if (not face.has_neighbor_)
      {
        Vector3& n = face.normal_;

        std::string boundary_name;
        if (n.Dot(ihat) > 0.999) boundary_name = "XMAX";
        else if (n.Dot(ihat) < -0.999)
          boundary_name = "XMIN";
        else if (n.Dot(jhat) > 0.999)
          boundary_name = "YMAX";
        else if (n.Dot(jhat) < -0.999)
          boundary_name = "YMIN";
        else if (n.Dot(khat) > 0.999)
          boundary_name = "ZMAX";
        else if (n.Dot(khat) < -0.999)
          boundary_name = "ZMIN";

        uint64_t bndry_id = vol_cont->MakeBoundaryID(boundary_name);

        face.neighbor_id_ = bndry_id;

        vol_cont->GetBoundaryIDMap()[bndry_id] = boundary_name;
      } // if bndry

  opensn::mpi.Barrier();
  log.Log() << program_timer.GetTimeString() << " Done setting orthogonal boundaries.";
}

void
VolumeMesher::Execute()
{
  log.Log() << "Empty volume mesher, nothing executed.";
  log.Log() << std::endl;
}

} // namespace opensn
