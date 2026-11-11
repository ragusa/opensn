// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mesh/raytrace/material_hull.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/mesh/cell.h"
#include "framework/utils/error.h"
#include "framework/runtime.h"
#include <algorithm>
#include <set>
#include <string>
#include <tuple>

namespace opensn
{

namespace
{

/// Number of coordinates and of integer attributes that describe one triangle in the gather.
constexpr std::size_t coords_per_triangle = 9;
constexpr std::size_t keys_per_triangle = 6;

/// Gathers the values of every rank, in rank order, on every rank.
template <typename T>
std::vector<T>
GatherFromAllRanks(const std::vector<T>& local_values)
{
  OpenSnLogicalErrorIf(local_values.size() >
                         static_cast<std::size_t>(std::numeric_limits<int>::max()),
                       "The local material hull is too large to gather.");
  const auto local_size = static_cast<int>(local_values.size());
  std::vector<int> sizes(mpi_comm.size(), 0);
  mpi_comm.all_gather(local_size, sizes);

  std::vector<int> displacements(mpi_comm.size(), 0);
  std::size_t total_size = 0;
  for (int rank = 0; rank < mpi_comm.size(); ++rank)
  {
    displacements[rank] = static_cast<int>(total_size);
    total_size += static_cast<std::size_t>(sizes[rank]);
    OpenSnLogicalErrorIf(total_size > static_cast<std::size_t>(std::numeric_limits<int>::max()),
                         "The material hull is too large to gather.");
  }

  std::vector<T> global_values(total_size);
  mpi_comm.all_gather(local_values, global_values, sizes, displacements);
  return global_values;
}

double
LongestEdge(const HullTriangle& triangle)
{
  const auto& v = triangle.vertices;
  return std::max({(v[1] - v[0]).Norm(), (v[2] - v[1]).Norm(), (v[0] - v[2]).Norm()});
}

} // namespace

std::shared_ptr<MaterialHull>
MaterialHull::Extract(const Mesh& grid, const FaceSelection selection)
{
  OpenSnInvalidArgumentIf(grid.GetDimension() != 3,
                          "A material hull requires a three-dimensional mesh.");

  // Block id that stands for the exterior while the materials are not yet indexed.
  constexpr std::uint64_t exterior_block = std::numeric_limits<std::uint64_t>::max();

  std::vector<double> local_coords;
  std::vector<std::uint64_t> local_keys;
  const auto emit = [&local_coords, &local_keys](const Vector3& a,
                                                 const Vector3& b,
                                                 const Vector3& c,
                                                 const std::uint64_t back_block,
                                                 const std::uint64_t front_block,
                                                 const std::uint64_t boundary_id,
                                                 const std::uint64_t owner_cell,
                                                 const std::uint64_t face,
                                                 const std::uint64_t side)
  {
    for (const auto& vertex : {a, b, c})
      local_coords.insert(local_coords.end(), {vertex.x, vertex.y, vertex.z});
    local_keys.insert(local_keys.end(),
                      {back_block, front_block, boundary_id, owner_cell, face, side});
  };

  for (const auto& cell : grid.GetLocalCells())
    for (std::size_t f = 0; f < cell->faces.size(); ++f)
    {
      const auto& face = cell->faces[f];
      std::uint64_t front_block = exterior_block;
      std::uint64_t boundary_id = NO_BOUNDARY;
      if (face.has_neighbor)
      {
        // A face between two cells is emitted by the cell with the lower global id, which is a
        // local cell on exactly one rank.
        if (cell->global_id > face.neighbor_id)
          continue;
        const auto& neighbor = grid.GetGlobalCell(face.neighbor_id);
        if (selection == FaceSelection::MATERIAL_INTERFACES and neighbor.block_id == cell->block_id)
          continue;
        front_block = neighbor.block_id;
      }
      else
        boundary_id = face.neighbor_id;

      // The face vertices are ordered counter-clockwise seen from outside the cell, so the
      // triangles below have the outward normal of the owner cell.
      const auto& vertex_ids = face.vertex_ids;
      const std::size_t num_vertices = vertex_ids.size();
      OpenSnLogicalErrorIf(num_vertices < 3,
                           "Cell " + std::to_string(cell->global_id) + " face " +
                             std::to_string(f) + " has fewer than three vertices.");
      if (num_vertices == 3)
        emit(grid.GlobalVertex(vertex_ids[0]),
             grid.GlobalVertex(vertex_ids[1]),
             grid.GlobalVertex(vertex_ids[2]),
             cell->block_id,
             front_block,
             boundary_id,
             cell->global_id,
             f,
             0);
      else
        for (std::size_t side = 0; side < num_vertices; ++side)
          emit(grid.GlobalVertex(vertex_ids[side]),
               grid.GlobalVertex(vertex_ids[(side + 1) % num_vertices]),
               face.centroid,
               cell->block_id,
               front_block,
               boundary_id,
               cell->global_id,
               f,
               side);
    }

  const auto coords = GatherFromAllRanks(local_coords);
  const auto keys = GatherFromAllRanks(local_keys);
  OpenSnLogicalErrorIf(coords.size() % coords_per_triangle != 0 or
                         keys.size() % keys_per_triangle != 0 or
                         coords.size() / coords_per_triangle != keys.size() / keys_per_triangle,
                       "The gathered material hull is inconsistent.");
  const std::size_t num_triangles = keys.size() / keys_per_triangle;
  OpenSnLogicalErrorIf(num_triangles > std::numeric_limits<std::uint32_t>::max(),
                       "The material hull has too many triangles.");

  std::set<unsigned int> unique_block_ids;
  for (std::size_t t = 0; t < num_triangles; ++t)
    for (std::size_t k = 0; k < 2; ++k)
      if (const auto block = keys[t * keys_per_triangle + k]; block != exterior_block)
        unique_block_ids.insert(static_cast<unsigned int>(block));
  std::vector<unsigned int> block_ids(unique_block_ids.begin(), unique_block_ids.end());

  const auto material_index = [&block_ids](const std::uint64_t block)
  {
    if (block == exterior_block)
      return static_cast<std::uint32_t>(block_ids.size());
    const auto it = std::lower_bound(block_ids.begin(), block_ids.end(), block);
    return static_cast<std::uint32_t>(it - block_ids.begin());
  };

  std::vector<HullTriangle> triangles(num_triangles);
  for (std::size_t t = 0; t < num_triangles; ++t)
  {
    auto& triangle = triangles[t];
    const double* xyz = &coords[t * coords_per_triangle];
    for (std::size_t v = 0; v < 3; ++v)
      triangle.vertices[v] = Vector3(xyz[3 * v], xyz[3 * v + 1], xyz[3 * v + 2]);
    const std::uint64_t* key = &keys[t * keys_per_triangle];
    triangle.back_material = material_index(key[0]);
    triangle.front_material = material_index(key[1]);
    triangle.boundary_id = key[2];
    triangle.owner_cell = key[3];
    triangle.face = static_cast<std::uint32_t>(key[4]);
    triangle.side = static_cast<std::uint32_t>(key[5]);
  }

  // The gather order depends on the partitioning; this order does not.
  std::sort(
    triangles.begin(),
    triangles.end(),
    [](const HullTriangle& a, const HullTriangle& b)
    { return std::tie(a.owner_cell, a.face, a.side) < std::tie(b.owner_cell, b.face, b.side); });

  return std::make_shared<MaterialHull>(std::move(triangles), std::move(block_ids));
}

MaterialHull::MaterialHull(std::vector<HullTriangle> triangles, std::vector<unsigned int> block_ids)
  : triangles_(std::move(triangles)), block_ids_(std::move(block_ids))
{
  OpenSnInvalidArgumentIf(std::adjacent_find(block_ids_.begin(),
                                             block_ids_.end(),
                                             std::greater_equal<>()) != block_ids_.end(),
                          "The block ids of a material hull must be sorted and unique.");
  OpenSnInvalidArgumentIf(triangles_.size() > std::numeric_limits<std::uint32_t>::max(),
                          "A material hull has too many triangles.");

  Vector3 lower(std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max());
  Vector3 upper = -1.0 * lower;
  for (const auto& triangle : triangles_)
  {
    OpenSnInvalidArgumentIf(triangle.back_material > GetExteriorIndex() or
                              triangle.front_material > GetExteriorIndex(),
                            "A material hull triangle refers to an unknown material.");
    for (const auto& vertex : triangle.vertices)
    {
      lower = Vector3(
        std::min(lower.x, vertex.x), std::min(lower.y, vertex.y), std::min(lower.z, vertex.z));
      upper = Vector3(
        std::max(upper.x, vertex.x), std::max(upper.y, vertex.y), std::max(upper.z, vertex.z));
    }
  }
  size_ = triangles_.empty() ? 0.0 : (upper - lower).Norm();
}

std::uint32_t
MaterialHull::GetMaterialIndex(const unsigned int block_id) const
{
  const auto it = std::lower_bound(block_ids_.begin(), block_ids_.end(), block_id);
  OpenSnInvalidArgumentIf(it == block_ids_.end() or *it != block_id,
                          "Block id " + std::to_string(block_id) +
                            " does not appear in the material hull.");
  return static_cast<std::uint32_t>(it - block_ids_.begin());
}

std::vector<std::uint32_t>
MaterialHull::FindTrianglesAt(const Vector3& point, const double relative_tolerance) const
{
  std::vector<std::uint32_t> found;
  for (std::size_t t = 0; t < triangles_.size(); ++t)
  {
    const auto& v = triangles_[t].vertices;
    const double distance = (ClosestPointOnTriangle(point, v[0], v[1], v[2]) - point).Norm();
    if (distance <= relative_tolerance * LongestEdge(triangles_[t]))
      found.push_back(static_cast<std::uint32_t>(t));
  }
  return found;
}

std::vector<std::uint32_t>
MaterialHull::FindEndPointExclusions(const Vector3& point, const double relative_tolerance) const
{
  auto found = FindTrianglesAt(point, relative_tolerance);
  for (const auto t : found)
  {
    const auto& triangle = triangles_[t];
    OpenSnInvalidArgumentIf(triangle.back_material != triangle.front_material and
                              triangle.back_material != GetExteriorIndex() and
                              triangle.front_material != GetExteriorIndex(),
                            "The point " + point.PrintStr() +
                              " lies on the interface between block ids " +
                              std::to_string(block_ids_[triangle.back_material]) + " and " +
                              std::to_string(block_ids_[triangle.front_material]) +
                              ", so a segment that ends there has no defined end material.");
  }
  return found;
}

Vector3
ClosestPointOnTriangle(const Vector3& P, const Vector3& A, const Vector3& B, const Vector3& C)
{
  const Vector3 ab = B - A;
  const Vector3 ac = C - A;
  const Vector3 ap = P - A;
  const double d1 = ab.Dot(ap);
  const double d2 = ac.Dot(ap);
  if (d1 <= 0.0 and d2 <= 0.0)
    return A;

  const Vector3 bp = P - B;
  const double d3 = ab.Dot(bp);
  const double d4 = ac.Dot(bp);
  if (d3 >= 0.0 and d4 <= d3)
    return B;

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0)
    return A + (d1 / (d1 - d3)) * ab;

  const Vector3 cp = P - C;
  const double d5 = ab.Dot(cp);
  const double d6 = ac.Dot(cp);
  if (d6 >= 0.0 and d5 <= d6)
    return C;

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0)
    return A + (d2 / (d2 - d6)) * ac;

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0)
    return B + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (C - B);

  const double denominator = 1.0 / (va + vb + vc);
  return A + (vb * denominator) * ab + (vc * denominator) * ac;
}

} // namespace opensn
