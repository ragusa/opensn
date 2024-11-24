// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <iostream>
#include <memory>

/**
 * Meshes in OpenSn follow the concept of Regions. In any given region the boundaries are a
 * collection of either line-meshes (2D) or surface-meshes (3D).
 */
namespace opensn
{

// forward declarations
class SPDS;
class MeshContinuum;

enum MeshType : int
{
  ORTHOGONAL,
  UNSTRUCTURED
};

enum BoundaryID : int
{
  XMIN = 0,
  XMAX = 1,
  YMIN = 2,
  YMAX = 3,
  ZMIN = 4,
  ZMAX = 5
};

struct OrthoMeshAttributes
{
  size_t Nx = 0;
  size_t Ny = 0;
  size_t Nz = 0;
};

/// Obtains the current mesh from the global stack.
std::shared_ptr<MeshContinuum> GetCurrentMesh();
// std::unique_ptr<MeshContinuum> GetCurrentMesh(); Why not this?

} // namespace opensn

#include "framework/mesh/mesh_vector.h"                   // For Vector3
#include "framework/mesh/mesh_matrix3x3.h"                // For Matrix3x3
#include "framework/mesh/mesh_face.h"                     // For Face
#include "framework/mesh/mesh_edge_loops.h"               // For Edge
#include "framework/mesh/cell/cell.h"                     // For Cell
#include "framework/mesh/logical_volume/logical_volume.h" // For Cell
