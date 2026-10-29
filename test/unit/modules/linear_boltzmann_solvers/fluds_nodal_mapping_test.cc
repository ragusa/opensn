// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

// Behaviour-preservation tests for the sweep plumbing that identifies face nodes through the
// spatial discretization instead of through the face vertices: for PWLD the new nodal mappings
// and face histogram must coincide with the vertex-based ones; for PWQD every face node must be
// matched to the neighbour's node at the same location.

#include "test/unit/common/mesh_builders.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aah_fluds_common_data.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_discontinuous.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_quadratic/piecewise_quadratic_discontinuous.h"
#include "framework/math/spatial_discretization/cell_mappings/cell_mapping.h"
#include "framework/mesh/mesh/grid_face_histogram.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/mesh/cell.h"
#include "framework/runtime.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

using namespace opensn;

namespace
{

void
ExpectMatchesVertexMapping(const std::shared_ptr<Mesh>& grid)
{
  auto sdm = PieceWiseLinearDiscontinuous::New(grid);
  const auto mappings = MakeGridNodalMappings(*grid, *sdm);
  ASSERT_EQ(mappings.size(), grid->GetLocalCellCount());

  for (const auto& cell : grid->GetLocalCells())
  {
    const auto& cell_mappings = mappings[cell->local_id];
    ASSERT_EQ(cell_mappings.size(), cell->faces.size());
    for (size_t f = 0; f < cell->faces.size(); ++f)
    {
      const auto& face = cell->faces[f];
      const auto& fnm = cell_mappings[f];
      if (not face.has_neighbor)
      {
        EXPECT_EQ(fnm.associated_face_, -1);
        EXPECT_TRUE(fnm.face_node_mapping_.empty());
        EXPECT_TRUE(fnm.cell_node_mapping_.empty());
        continue;
      }
      std::vector<short> face_node_mapping, cell_node_mapping;
      grid->FindAssociatedVertices(face, face_node_mapping);
      grid->FindAssociatedCellVertices(face, cell_node_mapping);
      EXPECT_EQ(fnm.associated_face_, face.GetNeighborAdjacentFaceIndex(grid.get()));
      EXPECT_EQ(fnm.face_node_mapping_, face_node_mapping);
      EXPECT_EQ(fnm.cell_node_mapping_, cell_node_mapping);
    }
  }
}

} // namespace

TEST(FLUDSNodalMapping, PWLDMatchesVertexMapping1D)
{
  ExpectMatchesVertexMapping(BuildLineMesh(1.0, 8, 0.0));
}

TEST(FLUDSNodalMapping, PWLDMatchesVertexMapping2D)
{
  ExpectMatchesVertexMapping(BuildSquareMesh(1.0, 4, 0.0));
}

TEST(FLUDSNodalMapping, PWLDMatchesVertexMapping3D)
{
  ExpectMatchesVertexMapping(BuildBoxMesh(1.0, 3, 0.0));
}

TEST(FLUDSNodalMapping, PWLDHistogramMatchesVertexHistogram)
{
  for (const auto& grid :
       {BuildLineMesh(1.0, 8, 0.0), BuildSquareMesh(1.0, 4, 0.0), BuildBoxMesh(1.0, 3, 0.0)})
  {
    auto sdm = PieceWiseLinearDiscontinuous::New(grid);
    std::vector<size_t> face_sizes;
    for (const auto& cell : grid->GetLocalCells())
    {
      const auto& cm = sdm->GetCellMapping(*cell);
      for (size_t f = 0; f < cell->faces.size(); ++f)
        face_sizes.push_back(cm.GetNumFaceNodes(f));
    }
    const auto from_nodes = GridFaceHistogram::Build(face_sizes);
    const auto from_vertices = grid->MakeGridFaceHistogram();
    ASSERT_EQ(from_nodes->GetNumberOfFaceHistogramBins(),
              from_vertices->GetNumberOfFaceHistogramBins());
    for (size_t b = 0; b < from_nodes->GetNumberOfFaceHistogramBins(); ++b)
      EXPECT_EQ(from_nodes->GetFaceHistogramBinDOFSize(b),
                from_vertices->GetFaceHistogramBinDOFSize(b));
  }
}

TEST(FLUDSNodalMapping, CompactViewSerializationRoundTrip)
{
  std::vector<CompactCellView> views;
  views.push_back({17, {{0, 2, 4}, {4, 5, 4}}});
  views.push_back({3, {{8, 0, 8}}});
  views.push_back({1024, {{16, 3, 6}, {22, 1, 3}, {25, 4, 4}}});

  std::vector<int64_t> buffer;
  AAH_FLUDSCommonData::SerializeCellInfo(views, buffer, 29);
  EXPECT_EQ(buffer.size(), 2 + 4 * 6);

  std::vector<CompactCellView> decoded;
  int64_t num_face_dofs = 0;
  AAH_FLUDSCommonData::DeSerializeCellInfo(decoded, &buffer, num_face_dofs);
  EXPECT_EQ(num_face_dofs, 29);
  ASSERT_EQ(decoded.size(), views.size());
  for (size_t c = 0; c < views.size(); ++c)
  {
    EXPECT_EQ(decoded[c].first, views[c].first);
    ASSERT_EQ(decoded[c].second.size(), views[c].second.size());
    for (size_t f = 0; f < views[c].second.size(); ++f)
    {
      EXPECT_EQ(decoded[c].second[f].face_slot, views[c].second[f].face_slot);
      EXPECT_EQ(decoded[c].second[f].face_index, views[c].second[f].face_index);
      EXPECT_EQ(decoded[c].second[f].num_face_nodes, views[c].second[f].num_face_nodes);
    }
  }
}

TEST(FLUDSNodalMapping, PWQDFaceNodesMatchByLocation)
{
  auto grid = BuildBoxMesh(1.0, 3, 0.0);
  auto sdm = PieceWiseQuadraticDiscontinuous::New(grid);
  const auto mappings = MakeGridNodalMappings(*grid, *sdm);

  for (const auto& cell : grid->GetLocalCells())
  {
    const auto& cm = sdm->GetCellMapping(*cell);
    for (size_t f = 0; f < cell->faces.size(); ++f)
    {
      const auto& face = cell->faces[f];
      const auto& fnm = mappings[cell->local_id][f];
      if (not face.has_neighbor)
        continue;
      ASSERT_EQ(fnm.face_node_mapping_.size(), 8u); // 4 vertices + 4 edge midpoints
      const auto& adj_cell = grid->GetGlobalCell(face.neighbor_id);
      const auto& acm = sdm->GetCellMapping(adj_cell);
      for (size_t fi = 0; fi < 8; ++fi)
      {
        const auto& x = cm.GetNodeLocations()[cm.MapFaceNode(f, fi)];
        const int ai = fnm.cell_node_mapping_[fi];
        EXPECT_EQ(acm.MapFaceNode(fnm.associated_face_, fnm.face_node_mapping_[fi]), ai);
        EXPECT_NEAR((x - acm.GetNodeLocations()[ai]).Norm(), 0.0, 1e-14);
      }
    }
  }
}
