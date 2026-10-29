#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spatial convergence of PWQD at finite optical thickness: unit cube, sigma_t = 1, c = 0.5, the
smooth isotropic source q = [sin(pi x) sin(pi y) sin(pi z)]^2 (it vanishes quadratically on the
boundary, so with vacuum boundaries the angular flux of every discrete direction is smooth; a
uniform source would give derivative kinks along the characteristics through the box edges and
no scheme converges at its design order), GLC 2x4 product quadrature on every mesh (so the
reference is the S_N solution of this quadrature and only the spatial error is measured). OpenSn
evaluates the source at the nodes, i.e. as its nodal interpolant, which is third-order accurate
for PWQD and second-order for PWLD. PWQD on 2^3, 4^3, 8^3 and 16^3 hexes and PWLD on 4^3, 8^3,
16^3 and 32^3 hexes are compared, each on its own grid of cell averages, with the cell averages
of a 32^3 PWQD reference over the same cells (the grids nest). The reference is itself
cross-checked against a 64^3 PWLD solution on the 32^3 grid: the two reference solutions of
different bases must agree with each other far better than the coarse solutions differ from
them. Reported: the relative L2 norm of the cell-average error, the observed orders (PWQD third
order, PWLD second order), and the wall time of each solve, which gives the cost at equal error.
SDM: PWLD and PWQD, AAH sweeps.
"""

import math
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
    from pyopensn.math import VectorSpatialFunction

if __name__ == "__main__":

    def boxes_of(N):
        h = 1.0 / N
        out = []
        for k in range(N):
            for j in range(N):
                for i in range(N):
                    out.append(RPPLogicalVolume(xmin=i * h, xmax=(i + 1) * h, ymin=j * h,
                                                ymax=(j + 1) * h, zmin=k * h, zmax=(k + 1) * h))
        return out

    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=1.0, c=0.5)
    def smooth_source(p, n):
        return [(math.sin(math.pi * p.x) * math.sin(math.pi * p.y) * math.sin(math.pi * p.z)) ** 2] * n

    src = VolumetricSource(block_ids=[0], func=VectorSpatialFunction(smooth_source))
    pquad = GLCProductQuadrature3DXYZ(n_polar=2, n_azimuthal=4, scattering_order=0)

    def solve(sdm, N):
        """Returns (field function, wall time of the solve)."""
        nodes = [i / N for i in range(N + 1)]
        grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes]).Execute()
        grid.SetUniformBlockID(0)
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
        )
        solver = SteadyStateSourceSolver(problem=phys)
        solver.Initialize()
        t0 = time.time()
        solver.Execute()
        elapsed = time.time() - t0
        return phys, phys.GetScalarFluxFieldFunction()[0], elapsed

    def averages(ff, boxes):
        avgs = []
        for vol in boxes:
            ffi = FieldFunctionInterpolationVolume()
            ffi.SetOperationType("avg")
            ffi.SetLogicalVolume(vol)
            ffi.SetFieldFunction(ff)
            ffi.Execute()
            avgs.append(ffi.GetValue())
        return avgs

    def rel_l2(a, b):
        peak = max(abs(v) for v in b)
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(b)) / peak

    # References (kept alive: the field functions refer to their problems)
    ref_phys, ref_ff, ref_time = solve("pwqd", 32)
    chk_phys, chk_ff, chk_time = solve("pwld", 64)
    boxes32 = boxes_of(32)
    cross = rel_l2(averages(chk_ff, boxes32), averages(ref_ff, boxes32))

    err = {}
    times = {("pwqd", 32): ref_time, ("pwld", 64): chk_time}
    for sdm, N in (("pwqd", 2), ("pwqd", 4), ("pwqd", 8), ("pwqd", 16), ("pwld", 4), ("pwld", 8),
                   ("pwld", 16), ("pwld", 32)):
        phys, ff, elapsed = solve(sdm, N)
        boxes = boxes_of(N)
        err[(sdm, N)] = rel_l2(averages(ff, boxes), averages(ref_ff, boxes))
        times[(sdm, N)] = elapsed

    if rank == 0:
        print(f"RefCrossCheck-pwqd32-vs-pwld64={cross:.4e}")
        for key, e in err.items():
            print(f"RelL2Err-{key[0]}-{key[1]}={e:.4e}")
        for key, t in times.items():
            print(f"Execute-time-{key[0]}-{key[1]}={t:.3f}")
        for sdm, a, b in (("pwqd", 2, 4), ("pwqd", 4, 8), ("pwqd", 8, 16), ("pwld", 4, 8),
                          ("pwld", 8, 16), ("pwld", 16, 32)):
            print(f"Order-{sdm}-{a}-{b}={math.log(err[(sdm, a)] / err[(sdm, b)]) / math.log(2):.3f}")
