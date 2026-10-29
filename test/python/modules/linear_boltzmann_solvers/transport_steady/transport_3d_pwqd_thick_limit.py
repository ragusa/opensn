#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thick diffusion limit (Larsen-Morel-Miller scaling) for the PWQD basis with PWLD as the control.
Unit cube, 4x4x4 orthogonal hexes, vacuum boundaries, sigma_t = 1/eps, sigma_a = eps, q = eps
(4 pi-integrated), eps = 1e-4, GLC 2x4 product quadrature, within-group MIP DSA.

Reference: the eps -> 0 limit is the diffusion solution of -(1/3) Lap u + u = 1 with u = 0 on the
boundary, given by the triple sine series
    u = sum_{l,m,n odd} 64 / (pi^3 l m n) / (1 + pi^2 (l^2 + m^2 + n^2) / 3)
          sin(l pi x) sin(m pi y) sin(n pi z),
whose cell averages are evaluated analytically below. Reported: the maximum over cells and the
volume-weighted L2 norm of the cell-average error, relative to the peak cell average of the
reference. Both bases preserve the limit (no locking): the errors are the discretization error of
the limit solution at h = 1/4. Independent reference values at this configuration from the
Beyond_PWLD prototype (RESULTS.md, "Thick limit under refinement", 2026-10-28): PWQD 0.0125
(max) / 0.0057 (L2), PWLD 0.0724 / 0.0625.
SDM: PWLD and PWQD; AAH or CBC sweeps (PWQD_SWEEP_TYPE).
"""

import math
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


def limit_cell_averages(n, modes=41):
    """Cell averages of the limit solution on the n^3 mesh of the unit cube (x fastest)."""
    h = 1.0 / n
    odd = range(1, modes + 1, 2)

    # 1D factors: average of sin(l pi x) over cell i, for every odd l
    def avg_sin(l, i):
        a, b = i * h, (i + 1) * h
        return (math.cos(l * math.pi * a) - math.cos(l * math.pi * b)) / (l * math.pi * h)

    s1d = {l: [avg_sin(l, i) for i in range(n)] for l in odd}
    coef = {}
    for l in odd:
        for m in odd:
            for p in odd:
                coef[(l, m, p)] = (64.0 / (math.pi ** 3 * l * m * p)) / (
                    1.0 + math.pi ** 2 * (l * l + m * m + p * p) / 3.0)
    avgs = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                u = 0.0
                for (l, m, p), c in coef.items():
                    u += c * s1d[l][i] * s1d[m][j] * s1d[p][k]
                avgs.append(u)
    return avgs


if __name__ == "__main__":

    # Sweep type from the environment so the same input is registered for AAH and CBC
    sweep_type = os.environ.get("PWQD_SWEEP_TYPE", "AAH")

    N = 4
    eps = 1.0e-4
    nodes = [i / N for i in range(N + 1)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes]).Execute()
    grid.SetUniformBlockID(0)

    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=1.0 / eps, c=1.0 - eps * eps)  # sigma_a = eps
    src = VolumetricSource(block_ids=[0], group_strength=[eps])
    pquad = GLCProductQuadrature3DXYZ(n_polar=2, n_azimuthal=4, scattering_order=0)

    ref = limit_cell_averages(N)
    ref_peak = max(ref)

    # One logical volume per cell to extract cell averages
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
                    "l_abs_tol": 1.0e-10,
                    "l_max_its": 500,
                    "gmres_restart_interval": 50,
                    "apply_wgdsa": True,
                    "wgdsa_l_abs_tol": 1.0e-12,
                    "wgdsa_l_max_its": 200,
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

        err_max = max(abs(a - r) for a, r in zip(avgs, ref)) / ref_peak
        err_l2 = math.sqrt(sum((a - r) ** 2 for a, r in zip(avgs, ref)) / len(ref)) / ref_peak
        results[sdm] = (err_max, err_l2, max(avgs))

    if rank == 0:
        print(f"Reference peak cell average = {ref_peak:.6e}")
        for sdm in ("pwld", "pwqd"):
            err_max, err_l2, peak = results[sdm]
            print(f"Peak-cell-avg-{sdm}={peak:.6e}")
            print(f"RelMaxCellAvgErr-{sdm}={err_max:.6e}")
            print(f"RelL2CellAvgErr-{sdm}={err_l2:.6e}")
