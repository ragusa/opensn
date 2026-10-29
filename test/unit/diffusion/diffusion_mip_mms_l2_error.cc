// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "gmock/gmock.h"
#include "test/unit/common/mesh_builders.h"

#include "modules/diffusion/diffusion_mip_solver.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/acceleration.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_structs.h"

#include "framework/field_functions/field_function_grid_based.h"
#include "framework/logging/log.h"
#include "framework/math/functions/function.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_discontinuous.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_quadratic/piecewise_quadratic_discontinuous.h"
#include "framework/math/spatial_discretization/finite_element/unit_cell_matrices.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/runtime.h"

#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <utility>
#include <vector>

using namespace opensn;

namespace
{

// Manufactured solution: phi = cos(pi x) + cos(pi y) [+ cos(pi z) in 3D], D = 1, sigma_r = 0,
// so q = -div(D grad phi) = pi^2 phi.
double
MMS_phi_2d(const Vector3& pt)
{
  return std::cos(M_PI * pt.x) + std::cos(M_PI * pt.y);
}

double
MMS_q_2d(const Vector3& pt)
{
  return M_PI * M_PI * MMS_phi_2d(pt);
}

double
MMS_phi_3d(const Vector3& pt)
{
  return std::cos(M_PI * pt.x) + std::cos(M_PI * pt.y) + std::cos(M_PI * pt.z);
}

double
MMS_q_3d(const Vector3& pt)
{
  return M_PI * M_PI * MMS_phi_3d(pt);
}

std::pair<double, double>
SimTest_IP_MMS_L2error(std::shared_ptr<Mesh> grid,
                       const std::string& sdm_type = "PWLD",
                       bool three_d = false)
{
  using MatID2XSMap = std::map<unsigned int, Multigroup_D_and_sigR>;

  opensn::log.Log() << "SimTest_IP_MMS_L2error (" << sdm_type << ")";
  opensn::log.Log() << "Global num cells: " << grid->GetGlobalNumberOfCells();

  std::shared_ptr<SpatialDiscretization> sdm_ptr;
  if (sdm_type == "PWLD")
    sdm_ptr = PieceWiseLinearDiscontinuous::New(grid);
  else if (sdm_type == "PWQD")
    sdm_ptr = PieceWiseQuadraticDiscontinuous::New(grid);
  else
    throw std::invalid_argument("unknown sdm_type " + sdm_type);
  const auto& sdm = *sdm_ptr;
  const auto& one_dof_per_node = sdm.UNITARY_UNKNOWN_MANAGER;

  const size_t num_local_dofs = sdm.GetNumLocalDOFs(one_dof_per_node);
  const size_t num_global_dofs = sdm.GetNumGlobalDOFs(one_dof_per_node);

  opensn::log.Log() << "Num local DOFs: " << num_local_dofs;
  opensn::log.Log() << "Num global DOFs: " << num_global_dofs;

  std::map<uint64_t, BoundaryCondition> bcs;
  bcs[0] = {BCType::DIRICHLET, {2, 0, 0}}, bcs[1] = {BCType::DIRICHLET, {2, 0, 0}},
  bcs[2] = {BCType::DIRICHLET, {2, 0, 0}}, bcs[3] = {BCType::DIRICHLET, {2, 0, 0}},
  bcs[4] = {BCType::DIRICHLET, {2, 0, 0}}, bcs[5] = {BCType::DIRICHLET, {2, 0, 0}};

  MatID2XSMap matid_2_xs_map;
  matid_2_xs_map.insert(std::make_pair(0, Multigroup_D_and_sigR{{1.0}, {0.0}}));

  std::vector<UnitCellMatrices> unit_cell_matrices(grid->GetLocalCellCount());
  for (const auto& cell : grid->GetLocalCells())
  {
    unit_cell_matrices[cell->local_id] = ComputeUnitCellIntegrals(sdm, *cell);
  }

  ScalarSpatialFunction mms_phi = three_d ? MMS_phi_3d : MMS_phi_2d;
  ScalarSpatialFunction mms_q = three_d ? MMS_q_3d : MMS_q_2d;

  DiffusionMIPSolver solver("SimTest_IP_MMS_L2error",
                            sdm,
                            one_dof_per_node,
                            bcs,
                            matid_2_xs_map,
                            unit_cell_matrices,
                            false,
                            true);
  solver.options.verbose = true;
  solver.options.residual_tolerance = 1.0e-10;
  solver.options.perform_symmetry_check = true;
  solver.SetReferenceSolutionFunction(mms_phi);
  solver.SetSourceFunction(mms_q);
  solver.Initialize();

  std::vector<double> q_vector(num_local_dofs, 1.0);
  std::vector<double> x_vector(num_local_dofs, 0.0);
  solver.AssembleAand_b_wQpoints(q_vector);
  solver.Solve(x_vector);

  auto ff =
    std::make_shared<FieldFunctionGridBased>("Phi", sdm_ptr, one_dof_per_node.unknowns.front());

  auto compute_l2_error = [&grid, &sdm, &mms_phi, &ff](const std::vector<double>& solution)
  {
    ff->UpdateFieldVector(solution);
    const auto field_wg = ff->GetGhostedFieldVector();

    double local_error = 0.0;
    for (const auto& cell : grid->GetLocalCells())
    {
      const auto& cell_mapping = sdm.GetCellMapping(*cell);
      const size_t num_nodes = cell_mapping.GetNumNodes();
      const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();

      std::vector<double> nodal_phi(num_nodes, 0.0);
      for (size_t j = 0; j < num_nodes; ++j)
      {
        const auto jmap = sdm.MapDOFLocal(*cell, j);
        nodal_phi[j] = field_wg[jmap];
      }

      for (size_t qp : fe_vol_data.GetQuadraturePointIndices())
      {
        double phi_fem = 0.0;
        for (size_t j = 0; j < num_nodes; ++j)
          phi_fem += nodal_phi[j] * fe_vol_data.ShapeValue(j, qp);

        const double phi_true = mms_phi(fe_vol_data.QPointXYZ(qp));
        local_error += std::pow(phi_true - phi_fem, 2.0) * fe_vol_data.JxW(qp);
      }
    }

    double global_error = 0.0;
    opensn::mpi_comm.all_reduce(local_error, global_error, mpi::op::sum<double>());
    return std::sqrt(global_error);
  };

  const double full_assembly_error = compute_l2_error(x_vector);

  solver.Assemble_b_wQpoints(q_vector);
  solver.Solve(x_vector);
  const double rhs_assembly_error = compute_l2_error(x_vector);

  opensn::log.Log() << "Error: " << std::scientific << rhs_assembly_error
                    << " Num-cells: " << grid->GetGlobalNumberOfCells();

  return {full_assembly_error, rhs_assembly_error};
}

} // namespace

TEST(DiffusionMIP, MMSL2Convergence)
{
  if (opensn::mpi_comm.size() != 1)
    return;

  const std::array<unsigned int, 5> refinements = {5, 10, 20, 40, 80};
  const double length = 2.0;

  std::vector<double> errors;
  errors.reserve(refinements.size());

  for (const unsigned int num_cells : refinements)
  {
    auto grid = BuildSquareMesh(length, num_cells, -length / 2.0);
    grid->SetUniformBlockID(0);
    grid->SetOrthogonalBoundaries();

    const auto [full_assembly_error, rhs_assembly_error] = SimTest_IP_MMS_L2error(grid);
    errors.push_back(full_assembly_error);

    EXPECT_NEAR(rhs_assembly_error, full_assembly_error, 1.0e-12);
  }

  for (size_t k = 1; k < errors.size(); ++k)
  {
    EXPECT_LT(errors[k], errors[k - 1]);

    const double rate = std::log(errors[k - 1] / errors[k]) / std::log(2.0);
    opensn::log.Log() << "MIP MMS N=" << refinements[k] << " L2 rate=" << rate;
    EXPECT_GT(rate, 1.75);
  }

  opensn::mpi_comm.barrier();
}

/**
 * 3D PWQD: the MMS error must converge at third order (p + 1 = 3) on orthogonal hexes, against
 * second order for PWLD on the same meshes. The reference is the exact manufactured solution; the
 * rates are compared with the theoretical ones.
 */
TEST(DiffusionMIP, MMSL2ConvergencePWQD3D)
{
  if (opensn::mpi_comm.size() != 1)
    return;

  const std::array<unsigned int, 3> refinements = {2, 4, 8};
  const double length = 2.0;

  std::map<std::string, std::vector<double>> errors;
  for (const std::string sdm_type : {"PWLD", "PWQD"})
    for (const unsigned int num_cells : refinements)
    {
      auto grid = BuildBoxMesh(length, num_cells, -length / 2.0);
      grid->SetUniformBlockID(0);
      grid->SetOrthogonalBoundaries();

      const auto [full_assembly_error, rhs_assembly_error] =
        SimTest_IP_MMS_L2error(grid, sdm_type, true);
      errors[sdm_type].push_back(full_assembly_error);
      EXPECT_NEAR(rhs_assembly_error, full_assembly_error, 1.0e-12);
    }

  // Two cells per side are pre-asymptotic for a cosine (PWLD rate 0.13 on 2^3 -> 4^3), so the
  // rate is checked on the finest pair; monotone decrease is required everywhere.
  for (const auto& [sdm_type, err] : errors)
    for (size_t k = 1; k < err.size(); ++k)
    {
      EXPECT_LT(err[k], err[k - 1]);
      const double rate = std::log(err[k - 1] / err[k]) / std::log(2.0);
      opensn::log.Log() << "MIP MMS 3D " << sdm_type << " N=" << refinements[k]
                        << " L2 err=" << std::scientific << err[k] << " rate=" << rate;
      if (k + 1 == err.size())
        EXPECT_GT(rate, sdm_type == "PWQD" ? 2.6 : 1.75);
    }
  // At equal h the quadratic basis is far more accurate than the linear one
  EXPECT_LT(errors["PWQD"].back(), 0.25 * errors["PWLD"].back());

  opensn::mpi_comm.barrier();
}
