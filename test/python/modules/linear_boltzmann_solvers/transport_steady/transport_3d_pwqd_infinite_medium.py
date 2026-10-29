#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D infinite-medium invariant for the PWQD (piecewise quadratic discontinuous) basis, with PWLD as
the control: unit cube of 2x2x2 hexes, reflecting on all six faces, sigma_t = 1, c = 0.5, uniform
isotropic source q = 1. The exact solution is the constant phi = q / sigma_a = 2, which every
basis containing the constants must reproduce to iteration tolerance at every node.
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
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume

if __name__ == "__main__":

    # Sweep type from the environment so the same input is registered for AAH and CBC
    sweep_type = os.environ.get("PWQD_SWEEP_TYPE", "AAH")

    nodes = [i / 2 for i in range(2 + 1)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes]).Execute()
    grid.SetUniformBlockID(0)

    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=1.0, c=0.5)
    src = VolumetricSource(block_ids=[0], group_strength=[1.0])
    pquad = GLCProductQuadrature3DXYZ(n_polar=2, n_azimuthal=4, scattering_order=0)

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
                {"name": n, "type": "reflecting"}
                for n in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
            ],
            spatial_discretization=sdm,
            sweep_type=sweep_type,
        )
        solver = SteadyStateSourceSolver(problem=phys)
        solver.Initialize()
        solver.Execute()

        ff = phys.GetScalarFluxFieldFunction()[0]
        whole = RPPLogicalVolume(infx=True, infy=True, infz=True)
        vals = {}
        for op in ("max", "avg"):
            ffi = FieldFunctionInterpolationVolume()
            ffi.SetOperationType(op)
            ffi.SetLogicalVolume(whole)
            ffi.SetFieldFunction(ff)
            ffi.Execute()
            vals[op] = ffi.GetValue()
        results[sdm] = vals

    if rank == 0:
        for sdm in ("pwld", "pwqd"):
            print(f"Max-value-{sdm}={results[sdm]['max']:.10f}")
            print(f"Avg-value-{sdm}={results[sdm]['avg']:.10f}")
