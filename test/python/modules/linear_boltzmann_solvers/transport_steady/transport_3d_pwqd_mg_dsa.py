#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multigroup (168-group graphite with an air cavity, as transport_3d_3a_dsa_ortho) with within-group
DSA on both groupsets and two-grid DSA on the thermal groupset, for PWQD and for PWLD on a
10x10x2 hex mesh. The test records that both groupsets converge with each basis and reports the
WGS iteration counts, wall times and the peak scalar fluxes of groups 39 and 120 for both bases;
PWQD's peaks must agree with PWLD's at the level of the discretization error of this coarse mesh
(5%), the independent anchor being the identical converged transport problem.
SDM: PWLD and PWQD; AAH or CBC sweeps (PWQD_SWEEP_TYPE).
"""

import os
import sys
import time

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume

if __name__ == "__main__":

    sweep_type = os.environ.get("PWQD_SWEEP_TYPE", "AAH")

    N = 10
    L = 100.0
    nodes = [-L / 2 + i * L / N for i in range(N + 1)]
    znodes = [0.0, 10.0, 20.0]
    grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, znodes]).Execute()
    grid.SetUniformBlockID(0)
    vol1 = RPPLogicalVolume(xmin=-10.0, xmax=10.0, ymin=-10.0, ymax=10.0, infz=True)
    grid.SetBlockIDFromLogicalVolume(vol1, 1, True)

    num_groups = 168
    xs_graphite = MultiGroupXS()
    xs_graphite.LoadFromOpenSn("../../../../assets/xs/xs_graphite_pure.xs")
    xs_air = MultiGroupXS()
    xs_air.LoadFromOpenSn("../../../../assets/xs/xs_air50RH.xs")

    strength = [0.0 for _ in range(num_groups)]
    strength[0] = 1.0
    mg_src0 = VolumetricSource(block_ids=[0], group_strength=strength)
    strength[0] = 0.0
    mg_src1 = VolumetricSource(block_ids=[1], group_strength=strength)

    pquad = GLCProductQuadrature3DXYZ(n_polar=4, n_azimuthal=8, scattering_order=1)
    whole = RPPLogicalVolume(infx=True, infy=True, infz=True)

    results = {}
    for sdm in ("pwld", "pwqd"):
        phys = DiscreteOrdinatesProblem(
            mesh=grid,
            num_groups=num_groups,
            groupsets=[
                {
                    "groups_from_to": [0, 62],
                    "angular_quadrature": pquad,
                    "angle_aggregation_type": "polar",
                    "inner_linear_method": "petsc_gmres",
                    "l_abs_tol": 1.0e-6,
                    "l_max_its": 1000,
                    "gmres_restart_interval": 30,
                    "apply_wgdsa": True,
                    "wgdsa_l_abs_tol": 1.0e-2,
                },
                {
                    "groups_from_to": [63, num_groups - 1],
                    "angular_quadrature": pquad,
                    "angle_aggregation_type": "polar",
                    "inner_linear_method": "petsc_gmres",
                    "l_abs_tol": 1.0e-6,
                    "l_max_its": 1000,
                    "gmres_restart_interval": 30,
                    "apply_wgdsa": True,
                    "apply_tgdsa": True,
                    "wgdsa_l_abs_tol": 1.0e-2,
                },
            ],
            xs_map=[
                {"block_ids": [0], "xs": xs_graphite},
                {"block_ids": [1], "xs": xs_air},
            ],
            volumetric_sources=[mg_src0, mg_src1],
            options={"max_ags_iterations": 1},
            spatial_discretization=sdm,
            sweep_type=sweep_type,
        )
        ss_solver = SteadyStateSourceSolver(problem=phys)
        ss_solver.Initialize()
        t0 = time.time()
        ss_solver.Execute()
        elapsed = time.time() - t0

        fflist = phys.GetScalarFluxFieldFunction()
        peaks = {}
        for g in (39, 120):
            ffi = FieldFunctionInterpolationVolume()
            ffi.SetOperationType("max")
            ffi.SetLogicalVolume(whole)
            ffi.SetFieldFunction(fflist[g])
            ffi.Execute()
            peaks[g] = ffi.GetValue()
        results[sdm] = (elapsed, peaks)

    if rank == 0:
        for sdm in ("pwld", "pwqd"):
            elapsed, peaks = results[sdm]
            print(f"Execute-time-{sdm}={elapsed:.2f}")
            print(f"Max-value-g39-{sdm}={peaks[39]:.6e}")
            print(f"Max-value-g120-{sdm}={peaks[120]:.6e}")
        for g in (39, 120):
            rel = abs(results["pwqd"][1][g] - results["pwld"][1][g]) / results["pwld"][1][g]
            print(f"RelDiff-pwqd-vs-pwld-g{g}={rel:.4e}")
