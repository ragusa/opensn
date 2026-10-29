// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

// The PVTU export of a PWQD field function writes the side-tetrahedron sub-mesh as VTK
// quadratic tetrahedra carrying the field values at the sub-mesh nodes. A quadratic polynomial
// is represented exactly by PWQD on planar-faced cells, so after export every point value must
// equal the polynomial at the point's coordinates: this checks the node ordering of the VTK
// cells and the tied-node evaluation independently of the solver.

#include "test/unit/common/mesh_builders.h"
#include "framework/field_functions/field_function_grid_based.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_quadratic/piecewise_quadratic_discontinuous.h"
#include "framework/math/spatial_discretization/cell_mappings/cell_mapping.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/mesh/cell.h"
#include "framework/runtime.h"
#include "gtest/gtest.h"
#include <vtkCellType.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <filesystem>
#include <unistd.h>

using namespace opensn;
namespace fs = std::filesystem;

namespace
{

double
Quadratic(double x, double y, double z)
{
  return 1.0 + x - 2.0 * y + 3.0 * z + x * x - x * y + 2.0 * z * z + 0.5 * y * z;
}

} // namespace

TEST(PWQDVtuExport, QuadraticFieldIsExact)
{
  if (opensn::mpi_comm.size() != 1)
    return;

  auto grid = BuildBoxMesh(1.0, 2, 0.0);
  auto sdm = PieceWiseQuadraticDiscontinuous::New(grid);
  const auto& uk_man = sdm->UNITARY_UNKNOWN_MANAGER;

  // Nodal values of the quadratic at the PWQD nodes
  std::vector<double> values(sdm->GetNumLocalDOFs(uk_man), 0.0);
  for (const auto& cell : grid->GetLocalCells())
  {
    const auto& X = sdm->GetCellMapping(*cell).GetNodeLocations();
    for (size_t n = 0; n < X.size(); ++n)
      values[sdm->MapDOFLocal(*cell, n, uk_man, 0, 0)] = Quadratic(X[n].x, X[n].y, X[n].z);
  }
  std::shared_ptr<SpatialDiscretization> sdm_base = sdm;
  auto ff = std::make_shared<FieldFunctionGridBased>("phi", sdm_base, uk_man.unknowns.front());
  ff->UpdateFieldVector(values);

  const fs::path dir =
    fs::temp_directory_path() / ("opensn_pwqd_export_" + std::to_string(getpid()));
  fs::create_directories(dir);
  const std::string base = (dir / "phi").string();
  FieldFunctionGridBased::ExportMultipleToPVTU(base, {ff});

  auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName((base + "_0.vtu").c_str());
  reader->Update();
  auto* ugrid = reader->GetOutput();
  ASSERT_NE(ugrid, nullptr);

  // 24 side tetrahedra per hexahedron (6 faces x 4 edges), 10 points each
  ASSERT_EQ(ugrid->GetNumberOfCells(), 24 * 8);
  ASSERT_EQ(ugrid->GetNumberOfPoints(), 24 * 8 * 10);
  for (vtkIdType c = 0; c < ugrid->GetNumberOfCells(); ++c)
    EXPECT_EQ(ugrid->GetCellType(c), VTK_QUADRATIC_TETRA);

  auto* arr = vtkDoubleArray::SafeDownCast(ugrid->GetPointData()->GetArray("phiUnknown_0"));
  ASSERT_NE(arr, nullptr);
  double x[3];
  for (vtkIdType p = 0; p < ugrid->GetNumberOfPoints(); ++p)
  {
    ugrid->GetPoint(p, x);
    EXPECT_NEAR(arr->GetValue(p), Quadratic(x[0], x[1], x[2]), 1e-12) << "point " << p;
  }

  fs::remove_all(dir);
}
