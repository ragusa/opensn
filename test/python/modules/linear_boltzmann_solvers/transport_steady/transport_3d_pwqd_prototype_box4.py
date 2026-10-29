#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-check of OpenSn's PWLD against the Beyond_PWLD prototype (brief Requirement 6) on the
shared problem: unit cube, 4x4x4 orthogonal hexes, GLC 2x4, sigma_t = 1, sigma_s = 0.5, unit
isotropic source (4 pi-integrated), vacuum boundaries. The reference cell-average scalar fluxes
below were generated independently by the prototype (snproto, PWLD, GMRES to 1e-12;
Beyond_PWLD branch phase0-prototype, studies/out/opensn_box4/prototype.npz, 2026-10-28) and are
listed with x fastest, then y, then z. The same input is also run with PWQD; its peak cell
average is reported for information (the PWQD anchors are the thick-limit and infinite-medium
tests).
SDM: PWLD (checked) and PWQD; AAH or CBC sweeps (PWQD_SWEEP_TYPE).
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
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume

PROTOTYPE_PWLD_CELL_AVERAGES = [
    2.605708814710e-01, 3.200155513557e-01, 3.200155513557e-01, 2.605708814710e-01,
    3.200155513557e-01, 4.210712299250e-01, 4.210712299250e-01, 3.200155513557e-01,
    3.200155513557e-01, 4.210712299250e-01, 4.210712299250e-01, 3.200155513557e-01,
    2.605708814710e-01, 3.200155513557e-01, 3.200155513557e-01, 2.605708814710e-01,
    3.200155513557e-01, 4.210712299250e-01, 4.210712299250e-01, 3.200155513557e-01,
    4.210712299250e-01, 5.786963206481e-01, 5.786963206481e-01, 4.210712299250e-01,
    4.210712299250e-01, 5.786963206481e-01, 5.786963206481e-01, 4.210712299250e-01,
    3.200155513557e-01, 4.210712299250e-01, 4.210712299250e-01, 3.200155513557e-01,
    3.200155513557e-01, 4.210712299250e-01, 4.210712299250e-01, 3.200155513557e-01,
    4.210712299250e-01, 5.786963206481e-01, 5.786963206481e-01, 4.210712299250e-01,
    4.210712299250e-01, 5.786963206481e-01, 5.786963206481e-01, 4.210712299250e-01,
    3.200155513557e-01, 4.210712299250e-01, 4.210712299250e-01, 3.200155513557e-01,
    2.605708814710e-01, 3.200155513557e-01, 3.200155513557e-01, 2.605708814710e-01,
    3.200155513557e-01, 4.210712299250e-01, 4.210712299250e-01, 3.200155513557e-01,
    3.200155513557e-01, 4.210712299250e-01, 4.210712299250e-01, 3.200155513557e-01,
    2.605708814710e-01, 3.200155513557e-01, 3.200155513557e-01, 2.605708814710e-01,
]

if __name__ == "__main__":

    # Sweep type from the environment so the same input is registered for AAH and CBC
    sweep_type = os.environ.get("PWQD_SWEEP_TYPE", "AAH")

    N = 4
    nodes = [i / N for i in range(N + 1)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes]).Execute()
    grid.SetUniformBlockID(0)

    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=1.0, c=0.5)
    src = VolumetricSource(block_ids=[0], group_strength=[1.0])
    pquad = GLCProductQuadrature3DXYZ(n_polar=2, n_azimuthal=4, scattering_order=0)

    h = 1.0 / N
    cell_volumes = []
    for k in range(N):
        for j in range(N):
            for i in range(N):
                cell_volumes.append(RPPLogicalVolume(xmin=i * h, xmax=(i + 1) * h,
                                                     ymin=j * h, ymax=(j + 1) * h,
                                                     zmin=k * h, zmax=(k + 1) * h))

    results = {}
    for sdm in ("pwld", "pwqd"):
        phys = DiscreteOrdinatesProblem(
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
            volumetric_sources=[src],
            boundary_conditions=[
                {"name": n, "type": "vacuum"}
                for n in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
            ],
            spatial_discretization=sdm,
            sweep_type=sweep_type,
        )
        solver = SteadyStateSourceSolver(problem=phys)
        solver.Initialize()
        solver.Execute()

        ff = phys.GetScalarFluxFieldFunction()[0]
        avgs = []
        for vol in cell_volumes:
            ffi = FieldFunctionInterpolationVolume()
            ffi.SetOperationType("avg")
            ffi.SetLogicalVolume(vol)
            ffi.SetFieldFunction(ff)
            ffi.Execute()
            avgs.append(ffi.GetValue())
        results[sdm] = avgs

    if rank == 0:
        ref_max = max(abs(v) for v in PROTOTYPE_PWLD_CELL_AVERAGES)
        diff = max(abs(a - r) for a, r in zip(results["pwld"], PROTOTYPE_PWLD_CELL_AVERAGES))
        print(f"MaxRelDiff-pwld-vs-prototype={diff / ref_max:.6e}")
        print(f"Peak-cell-avg-pwld={max(results['pwld']):.10f}")
        print(f"Peak-cell-avg-pwqd={max(results['pwqd']):.10f}")
