// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/vector3.h"
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace opensn
{

class Mesh;

/**
 * A triangle of a material hull.
 *
 * The normal (v1 - v0) x (v2 - v0) points from the back material to the front material. The
 * materials are indices into MaterialHull::GetBlockIDs(); the index
 * MaterialHull::GetExteriorIndex() denotes the exterior of the mesh.
 */
struct HullTriangle
{
  std::array<Vector3, 3> vertices;
  /// Material on the side the normal points away from. This is the material of the owner cell.
  std::uint32_t back_material = 0;
  /// Material on the side the normal points to.
  std::uint32_t front_material = 0;
  /// Boundary id of a domain-boundary face, MaterialHull::NO_BOUNDARY otherwise.
  std::uint64_t boundary_id = std::numeric_limits<std::uint64_t>::max();
  /// Global id of the cell that emitted the triangle.
  std::uint64_t owner_cell = 0;
  /// Face index in the owner cell.
  std::uint32_t face = 0;
  /// Index of the triangle within the face.
  std::uint32_t side = 0;
};

/**
 * The surfaces of a mesh across which the material changes.
 *
 * Cross sections are constant in a material, so the optical depth along a segment depends only
 * on where the segment crosses a material interface or the domain boundary. The hull holds these
 * surfaces as triangles: a face is kept when its two cells have different block ids or when it is
 * a domain-boundary face, whose far side is the exterior.
 *
 * Faces with more than three vertices are split into triangles around the face centroid, which is
 * the surface the piecewise-linear spatial discretization uses. A face with three vertices is
 * planar and is kept as one triangle.
 *
 * The hull of the whole mesh is replicated on every MPI rank and its triangles are sorted by
 * (owner cell global id, face, side), so it does not depend on the partitioning. Its memory cost
 * is O(number of ranks times number of hull triangles).
 */
class MaterialHull
{
public:
  static constexpr std::uint64_t NO_BOUNDARY = std::numeric_limits<std::uint64_t>::max();

  /// Which faces of the mesh are kept.
  enum class FaceSelection
  {
    /// Faces between cells with different block ids, and domain-boundary faces.
    MATERIAL_INTERFACES,
    /// Every face. Used to verify tracing on the hull against tracing on all cell faces.
    ALL_FACES
  };

  /**
   * Builds the hull of a mesh. Collective: every rank contributes the faces whose lower-global-id
   * cell it owns and receives the hull of the whole mesh.
   */
  static std::shared_ptr<MaterialHull>
  Extract(const Mesh& grid, FaceSelection selection = FaceSelection::MATERIAL_INTERFACES);

  /**
   * Builds a hull from triangles whose materials are indices into block_ids, or block_ids.size()
   * for the exterior. block_ids must be sorted and unique.
   */
  MaterialHull(std::vector<HullTriangle> triangles, std::vector<unsigned int> block_ids);

  const std::vector<HullTriangle>& GetTriangles() const { return triangles_; }

  /// Sorted block ids of the materials. The material index is the position in this list.
  const std::vector<unsigned int>& GetBlockIDs() const { return block_ids_; }

  /// Number of materials, not counting the exterior.
  std::uint32_t GetNumMaterials() const { return static_cast<std::uint32_t>(block_ids_.size()); }

  /// Material index that denotes the exterior of the mesh.
  std::uint32_t GetExteriorIndex() const { return GetNumMaterials(); }

  /// Returns the material index of a block id. Throws std::invalid_argument for an unknown id.
  std::uint32_t GetMaterialIndex(unsigned int block_id) const;

  /// Diagonal of the bounding box of the hull.
  double GetSize() const { return size_; }

  /**
   * Returns the sorted indices of the triangles that contain a point, to within relative_tolerance
   * times the longest edge of the triangle.
   */
  std::vector<std::uint32_t> FindTrianglesAt(const Vector3& point,
                                             double relative_tolerance = 1.0e-10) const;

  /**
   * Returns the sorted indices of the domain-boundary triangles that contain a segment end point,
   * to be excluded when tracing segments that end there. A segment that ends on a material
   * interface has no defined end material, so this throws std::invalid_argument when the point
   * lies on a triangle between two materials.
   */
  std::vector<std::uint32_t> FindEndPointExclusions(const Vector3& point,
                                                    double relative_tolerance = 1.0e-10) const;

private:
  std::vector<HullTriangle> triangles_;
  std::vector<unsigned int> block_ids_;
  double size_ = 0.0;
};

/// Returns the point of triangle ABC closest to P (Ericson, Real-Time Collision Detection, 5.1.5).
Vector3
ClosestPointOnTriangle(const Vector3& P, const Vector3& A, const Vector3& B, const Vector3& C);

} // namespace opensn
