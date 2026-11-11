// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "test/unit/common/mesh_builders.h"
#include "framework/mesh/logical_volume/rpp_logical_volume.h"
#include "framework/mesh/raytrace/material_hull.h"
#include "framework/parameters/parameter_block.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

namespace hull_test
{

/**
 * Materials on a regular grid of box-shaped voxels. This is an independent reference for the
 * material hull: its interface triangles are written down directly, and the length of a segment
 * in each material is obtained by clipping the segment against every voxel.
 *
 * A material of -1 marks a voxel that is not part of the domain.
 */
struct VoxelModel
{
  VoxelModel(int nx, int ny, int nz, opensn::Vector3 origin, opensn::Vector3 spacing)
    : n{nx, ny, nz},
      origin(origin),
      spacing(spacing),
      materials(static_cast<std::size_t>(nx) * ny * nz, 0)
  {
  }

  std::size_t Index(int i, int j, int k) const
  {
    return (static_cast<std::size_t>(k) * n[1] + j) * n[0] + i;
  }

  int& At(int i, int j, int k) { return materials[Index(i, j, k)]; }

  /// Material of a voxel; -1 outside the grid.
  int Material(int i, int j, int k) const
  {
    if (i < 0 or j < 0 or k < 0 or i >= n[0] or j >= n[1] or k >= n[2])
      return -1;
    return materials[Index(i, j, k)];
  }

  int NumMaterials() const { return *std::max_element(materials.begin(), materials.end()) + 1; }

  /// Block ids 0 ... NumMaterials() - 1.
  std::vector<unsigned int> BlockIDs() const
  {
    std::vector<unsigned int> ids(NumMaterials());
    for (std::size_t m = 0; m < ids.size(); ++m)
      ids[m] = static_cast<unsigned int>(m);
    return ids;
  }

  opensn::Vector3 Point(double i, double j, double k) const
  {
    return {origin.x + i * spacing.x, origin.y + j * spacing.y, origin.z + k * spacing.z};
  }

  /// Centre of a voxel.
  opensn::Vector3 Center(int i, int j, int k) const { return Point(i + 0.5, j + 0.5, k + 0.5); }

  /// Material at a point that is not on a voxel face; -1 outside the grid.
  int MaterialAt(const opensn::Vector3& point) const
  {
    return Material(static_cast<int>(std::floor((point.x - origin.x) / spacing.x)),
                    static_cast<int>(std::floor((point.y - origin.y) / spacing.y)),
                    static_cast<int>(std::floor((point.z - origin.z) / spacing.z)));
  }

  /**
   * Triangles of the faces between voxels of different materials and of the faces on the outside
   * of the domain. fan is true to split each face into four triangles around its centroid, as
   * MaterialHull::Extract does, and false to split it into two.
   */
  std::vector<opensn::HullTriangle> Triangles(bool fan) const
  {
    // Corners of the six faces of the unit cube, counter-clockwise seen from outside.
    static const int corners[6][4][3] = {{{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}},
                                         {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
                                         {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}},
                                         {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},
                                         {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}},
                                         {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}};
    static const int offsets[6][3] = {
      {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

    const auto exterior = static_cast<std::uint32_t>(NumMaterials());
    std::vector<opensn::HullTriangle> triangles;
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
        {
          const int back = Material(i, j, k);
          if (back < 0)
            continue;
          for (int f = 0; f < 6; ++f)
          {
            const int ni = i + offsets[f][0];
            const int nj = j + offsets[f][1];
            const int nk = k + offsets[f][2];
            const int front = Material(ni, nj, nk);
            if (front == back)
              continue;
            // A face between two voxels of the domain is written by the first of them.
            if (front >= 0 and Index(ni, nj, nk) < Index(i, j, k))
              continue;

            std::array<opensn::Vector3, 4> v;
            for (int c = 0; c < 4; ++c)
              v[c] = Point(i + corners[f][c][0], j + corners[f][c][1], k + corners[f][c][2]);
            opensn::HullTriangle triangle;
            triangle.back_material = static_cast<std::uint32_t>(back);
            triangle.front_material = front < 0 ? exterior : static_cast<std::uint32_t>(front);
            triangle.owner_cell = Index(i, j, k);
            triangle.face = static_cast<std::uint32_t>(f);
            if (fan)
            {
              const opensn::Vector3 centroid = (v[0] + v[1] + v[2] + v[3]) / 4.0;
              for (std::uint32_t side = 0; side < 4; ++side)
              {
                triangle.vertices = {v[side], v[(side + 1) % 4], centroid};
                triangle.side = side;
                triangles.push_back(triangle);
              }
            }
            else
            {
              triangle.vertices = {v[0], v[1], v[2]};
              triangle.side = 0;
              triangles.push_back(triangle);
              triangle.vertices = {v[0], v[2], v[3]};
              triangle.side = 1;
              triangles.push_back(triangle);
            }
          }
        }
    return triangles;
  }

  std::shared_ptr<opensn::MaterialHull> Hull(bool fan) const
  {
    return std::make_shared<opensn::MaterialHull>(Triangles(fan), BlockIDs());
  }

  /**
   * Length of a segment in each material; the last entry is the length outside the domain. The
   * result is ambiguous for a segment that runs within a voxel face between two materials.
   */
  std::vector<double> Lengths(const opensn::Vector3& start, const opensn::Vector3& end) const
  {
    const int num_materials = NumMaterials();
    std::vector<double> lengths(num_materials + 1, 0.0);
    const double length = (end - start).Norm();
    const double a[3] = {start.x, start.y, start.z};
    const double d[3] = {end.x - start.x, end.y - start.y, end.z - start.z};
    const double o[3] = {origin.x, origin.y, origin.z};
    const double h[3] = {spacing.x, spacing.y, spacing.z};

    double inside = 0.0;
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
        {
          const int ijk[3] = {i, j, k};
          double enter = 0.0;
          double leave = 1.0;
          for (int axis = 0; axis < 3 and enter < leave; ++axis)
          {
            const double lower = o[axis] + ijk[axis] * h[axis];
            const double upper = lower + h[axis];
            if (d[axis] == 0.0)
            {
              if (a[axis] < lower or a[axis] > upper)
                leave = enter;
              continue;
            }
            const double t0 = (lower - a[axis]) / d[axis];
            const double t1 = (upper - a[axis]) / d[axis];
            enter = std::max(enter, std::min(t0, t1));
            leave = std::min(leave, std::max(t0, t1));
          }
          if (leave <= enter)
            continue;
          const double chord = (leave - enter) * length;
          const int material = Material(i, j, k);
          inside += chord;
          lengths[material < 0 ? num_materials : material] += chord;
        }
    lengths[num_materials] += std::max(0.0, length - inside);
    return lengths;
  }

  std::array<int, 3> n;
  opensn::Vector3 origin;
  opensn::Vector3 spacing;
  std::vector<int> materials;
};

/// Geometry and materials of a triangle in a form that does not depend on which cell wrote it or
/// on its first vertex: the normal points from the lower to the higher material index and the
/// lexicographically smallest vertex comes first.
using TriangleKey = std::array<double, 11>;

inline TriangleKey
MakeTriangleKey(const opensn::HullTriangle& triangle)
{
  std::array<std::array<double, 3>, 3> v{};
  for (int i = 0; i < 3; ++i)
    v[i] = {triangle.vertices[i].x, triangle.vertices[i].y, triangle.vertices[i].z};
  auto back = static_cast<double>(triangle.back_material);
  auto front = static_cast<double>(triangle.front_material);
  if (back > front)
  {
    std::swap(back, front);
    std::swap(v[1], v[2]);
  }
  std::rotate(v.begin(), std::min_element(v.begin(), v.end()), v.end());
  return {
    v[0][0], v[0][1], v[0][2], v[1][0], v[1][1], v[1][2], v[2][0], v[2][1], v[2][2], back, front};
}

inline std::vector<TriangleKey>
MakeTriangleKeys(const std::vector<opensn::HullTriangle>& triangles)
{
  std::vector<TriangleKey> keys;
  keys.reserve(triangles.size());
  for (const auto& triangle : triangles)
    keys.push_back(MakeTriangleKey(triangle));
  std::sort(keys.begin(), keys.end());
  return keys;
}

/// Sum of the areas of the triangles that separate the two given materials.
inline double
InterfaceArea(const opensn::MaterialHull& hull, std::uint32_t material_a, std::uint32_t material_b)
{
  double area = 0.0;
  for (const auto& triangle : hull.GetTriangles())
    if ((triangle.back_material == material_a and triangle.front_material == material_b) or
        (triangle.back_material == material_b and triangle.front_material == material_a))
    {
      const auto& v = triangle.vertices;
      area += 0.5 * (v[1] - v[0]).Cross(v[2] - v[0]).Norm();
    }
  return area;
}

/**
 * True when every edge of the surface that bounds one material belongs to an even number of its
 * triangles, which holds for a closed surface.
 */
inline bool
IsClosedSurface(const opensn::MaterialHull& hull, std::uint32_t material)
{
  using Point = std::array<double, 3>;
  std::map<std::pair<Point, Point>, int> edge_counts;
  for (const auto& triangle : hull.GetTriangles())
  {
    if ((triangle.back_material == material) == (triangle.front_material == material))
      continue;
    for (int e = 0; e < 3; ++e)
    {
      const auto& p = triangle.vertices[e];
      const auto& q = triangle.vertices[(e + 1) % 3];
      Point a = {p.x, p.y, p.z};
      Point b = {q.x, q.y, q.z};
      if (b < a)
        std::swap(a, b);
      ++edge_counts[{a, b}];
    }
  }
  return std::all_of(edge_counts.begin(),
                     edge_counts.end(),
                     [](const auto& entry) { return entry.second % 2 == 0; });
}

/// A box of voxels, given by its first and last voxel along each axis, and its material.
struct VoxelRegion
{
  std::array<int, 3> first;
  std::array<int, 3> last;
  int material;
};

/**
 * Regions of the test problem on a 4 x 4 x 4 grid over [-1, 1]^3: material 1 is an L-shaped duct
 * that reaches the boundary at both ends, material 2 is a column next to it, and material 0 is the
 * rest. The duct is not convex, and materials 0, 1 and 2 meet along two edges.
 */
inline std::vector<VoxelRegion>
DoglegRegions()
{
  return {{{0, 1, 1}, {2, 1, 2}, 1}, {{2, 1, 1}, {2, 3, 2}, 1}, {{3, 3, 0}, {3, 3, 3}, 2}};
}

inline VoxelModel
MakeDoglegVoxelModel()
{
  VoxelModel model(4, 4, 4, opensn::Vector3(-1.0, -1.0, -1.0), opensn::Vector3(0.5, 0.5, 0.5));
  for (const auto& region : DoglegRegions())
    for (int k = region.first[2]; k <= region.last[2]; ++k)
      for (int j = region.first[1]; j <= region.last[1]; ++j)
        for (int i = region.first[0]; i <= region.last[0]; ++i)
          model.At(i, j, k) = region.material;
  return model;
}

/// Hexahedral mesh with the materials of MakeDoglegVoxelModel() as block ids.
inline std::shared_ptr<opensn::Mesh>
BuildDoglegMesh()
{
  auto grid = BuildBoxMesh(2.0, 4, -1.0);
  grid->SetUniformBlockID(0);
  for (const auto& region : DoglegRegions())
  {
    auto params = opensn::RPPLogicalVolume::GetInputParameters();
    opensn::ParameterBlock block;
    const char* lower_names[3] = {"xmin", "ymin", "zmin"};
    const char* upper_names[3] = {"xmax", "ymax", "zmax"};
    for (int axis = 0; axis < 3; ++axis)
    {
      block.AddParameter(lower_names[axis], -1.0 + 0.5 * region.first[axis]);
      block.AddParameter(upper_names[axis], -1.0 + 0.5 * (region.last[axis] + 1));
    }
    params.AssignParameters(block);
    grid->SetBlockIDFromLogicalVolume(
      opensn::RPPLogicalVolume(params), static_cast<unsigned int>(region.material), true);
  }
  return grid;
}

} // namespace hull_test
