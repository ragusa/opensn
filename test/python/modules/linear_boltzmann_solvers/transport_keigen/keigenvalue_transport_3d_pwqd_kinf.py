#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infinite-medium k-eigenvalue invariant for the PWQD basis with PWLD as the control: unit cube of
2x2x2 hexes, reflecting on all six faces, one-group fissile material without scattering
(simple_fissile_1g.xs: sigma_t = 1, sigma_f = 0.5, nu = 2), so k_inf = nu sigma_f / sigma_a = 1
exactly for any basis that contains the constants. PWQD is also run with the SCDSA acceleration
(MIP on the transport discretization), which exercises the k-eigenvalue DSA path with PWQD.
SDM: PWLD and PWQD; AAH or CBC sweeps (PWQD_SWEEP_TYPE).
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, PowerIterationKEigenSolver, SCDSAAcceleration

if __name__ == "__main__":

    sweep_type = os.environ.get("PWQD_SWEEP_TYPE", "AAH")

    nodes = [i / 2 for i in range(2 + 1)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes]).Execute()
    grid.SetUniformBlockID(0)

    xs = MultiGroupXS()
    xs.LoadFromOpenSn("../../../../assets/xs/simple_fissile_1g.xs")
    pquad = GLCProductQuadrature3DXYZ(n_polar=2, n_azimuthal=4, scattering_order=0)

    def make_problem(sdm):
        return DiscreteOrdinatesProblem(
            mesh=grid,
            num_groups=1,
            groupsets=[
                {
                    "groups_from_to": (0, 0),
                    "angular_quadrature": pquad,
                    "inner_linear_method": "petsc_gmres",
                    "l_abs_tol": 1.0e-12,
                    "l_max_its": 500,
                    "gmres_restart_interval": 50,
                }
            ],
            xs_map=[{"block_ids": [0], "xs": xs}],
            boundary_conditions=[
                {"name": n, "type": "reflecting"}
                for n in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
            ],
            options={"use_precursors": False},
            spatial_discretization=sdm,
            sweep_type=sweep_type,
        )

    results = {}
    for sdm in ("pwld", "pwqd"):
        phys = make_problem(sdm)
        k_solver = PowerIterationKEigenSolver(problem=phys, max_iters=200, k_tol=1.0e-10)
        k_solver.Initialize()
        k_solver.Execute()
        results[sdm] = k_solver.GetEigenvalue()

    phys = make_problem("pwqd")
    accel = SCDSAAcceleration(problem=phys, sdm="pwld", l_abs_tol=1.0e-12)
    k_solver = PowerIterationKEigenSolver(problem=phys, acceleration=accel, max_iters=200,
                                          k_tol=1.0e-10)
    k_solver.Initialize()
    k_solver.Execute()
    results["pwqd-scdsa"] = k_solver.GetEigenvalue()

    if rank == 0:
        for key in ("pwld", "pwqd", "pwqd-scdsa"):
            print(f"k-eigenvalue-{key}={results[key]:.10f}")
