#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Method of manufactured solutions for the PWQD basis on 2D polygonal meshes (angular volumetric
source, see transport_3d_mms_angular_source.py for the construction), with PWLD as the control.
Square [-1, 1]^2, sigma_t = 1, c = 0.5, GLC 4x8 product quadrature with scattering_order 1.

1. Patch tests with reflecting boundaries (exact for an isotropic manufactured flux):
   - quadratic phi = 1 + x + 2y + x^2 - xy + 2y^2 on 2x2 orthogonal quadrilaterals, on the
     triangles of square_mesh2x2_small.obj and on the mixed triangle/quadrilateral/pentagon/
     hexagon mesh quad_mesh_poly_mix.obj: PWQD contains the quadratics on every polygon (the
     least-squares tying of the spoke nodes reproduces them) and the nodal source is exact, so
     phi must be reproduced at every node to iteration tolerance;
   - the quadratic case on the quadrilaterals and on the polygon mix again with the exact
     incoming flux prescribed on every boundary through "arbitrary" boundaries with a
     spatial_function (psi_inc = phi at the boundary face nodes, including the edge midpoints).
2. Convergence with vacuum boundaries, phi = sin(pi x) sin(pi y): PWQD and PWLD on 4^2 .. 32^2
   quadrilaterals and on the triangle meshes square_mesh2x2_small.obj (32 cells) and
   square_mesh2x2.obj (2048 cells; h ratio 8); relative L2 errors integrated against the exact
   solution with the spatial operation function of the volume interpolator. PWQD is third order,
   PWLD second order.
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
    from pyopensn.mesh import OrthogonalMeshGenerator, FromFileMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.math import VectorSpatialAngularFunction

ASSETS = "../../../../assets/mesh/"
SIGMA_T = 1.0
C = 0.5
SIGMA_A = SIGMA_T * (1.0 - C)


def quadratic_phi(p):
    return 1.0 + p.x + 2.0 * p.y + p.x * p.x - p.x * p.y + 2.0 * p.y * p.y


def quadratic_source(p, omega, n):
    gx = 1.0 + 2.0 * p.x - p.y
    gy = 2.0 - p.x + 4.0 * p.y
    return [omega.x * gx + omega.y * gy + SIGMA_A * quadratic_phi(p)] * n


def sine_phi(p):
    return math.sin(math.pi * p.x) * math.sin(math.pi * p.y)


def sine_source(p, omega, n):
    sx, sy = math.sin(math.pi * p.x), math.sin(math.pi * p.y)
    cx, cy = math.cos(math.pi * p.x), math.cos(math.pi * p.y)
    streaming = math.pi * (omega.x * cx * sy + omega.y * sx * cy)
    return [streaming + SIGMA_A * sx * sy] * n


if __name__ == "__main__":

    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=SIGMA_T, c=C)
    pquad = GLCProductQuadrature2DXY(n_polar=4, n_azimuthal=8, scattering_order=1)
    whole = RPPLogicalVolume(infx=True, infy=True, infz=True)

    def quad_grid(N):
        nodes = [-1.0 + 2.0 * i / N for i in range(N + 1)]
        grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes]).Execute()
        grid.SetUniformBlockID(0)
        return grid

    def file_grid(name):
        grid = FromFileMeshGenerator(filename=ASSETS + name).Execute()
        grid.SetUniformBlockID(0)
        grid.SetOrthogonalBoundaries()
        return grid

    def solve(grid, sdm, source, bc_type, inflow=None):
        """bc_type on every boundary, or, with inflow (a function of (point, omega, num_groups)),
        "arbitrary" boundaries with that spatial_function."""
        src = VolumetricSource(block_ids=[0], angular_func=VectorSpatialAngularFunction(source))
        names = ("xmin", "xmax", "ymin", "ymax")
        if inflow is None:
            bcs = [{"name": n, "type": bc_type} for n in names]
        else:
            psi_inc = VectorSpatialAngularFunction(inflow)
            bcs = [{"name": n, "type": "arbitrary", "spatial_function": psi_inc} for n in names]
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
            boundary_conditions=bcs,
            spatial_discretization=sdm,
        )
        solver = SteadyStateSourceSolver(problem=phys)
        solver.Initialize()
        t0 = time.time()
        solver.Execute()
        return phys, phys.GetScalarFluxFieldFunction()[0], time.time() - t0

    def integrate(ff, op, func):
        ffi = FieldFunctionInterpolationVolume()
        ffi.SetOperationType(op)
        ffi.SetLogicalVolume(whole)
        ffi.SetFieldFunction(ff)
        ffi.SetSpatialOperationFunction(func)
        ffi.Execute()
        return ffi.GetValue()

    # 1. Patch tests
    patch = {}
    for label, grid in (("quads", quad_grid(2)),
                        ("triangles", file_grid("square_mesh2x2_small.obj")),
                        ("polygons", file_grid("quad_mesh_poly_mix.obj"))):
        phys, ff, _ = solve(grid, "pwqd", quadratic_source, "reflecting")
        patch[label] = integrate(ff, "max_func", lambda v, b, p: abs(v - quadratic_phi(p)))
    for label, grid in (("quads", quad_grid(2)), ("polygons", file_grid("quad_mesh_poly_mix.obj"))):
        phys, ff, _ = solve(grid, "pwqd", quadratic_source, None,
                            inflow=lambda p, omega, n: [quadratic_phi(p)] * n)
        patch[label + "-inflow"] = integrate(ff, "max_func",
                                             lambda v, b, p: abs(v - quadratic_phi(p)))

    # 2. Sine MMS convergence
    errors, times = {}, {}
    meshes = [(f"quad{N}", quad_grid(N)) for N in (4, 8, 16, 32)]
    meshes += [("tri32", file_grid("square_mesh2x2_small.obj")),
               ("tri2048", file_grid("square_mesh2x2.obj"))]
    for sdm in ("pwqd", "pwld"):
        for label, grid in meshes:
            phys, ff, elapsed = solve(grid, sdm, sine_source, "vacuum")
            err2 = integrate(ff, "sum_func", lambda v, b, p: (v - sine_phi(p)) ** 2)
            norm2 = integrate(ff, "sum_func", lambda v, b, p: sine_phi(p) ** 2)
            errors[(sdm, label)] = math.sqrt(err2 / norm2)
            times[(sdm, label)] = elapsed

    if rank == 0:
        for label, e in patch.items():
            print(f"PatchMaxNodalErr-pwqd-{label}={e:.3e}")
        for key, e in errors.items():
            print(f"RelL2Err-{key[0]}-{key[1]}={e:.4e}")
        for key, t in times.items():
            print(f"Execute-time-{key[0]}-{key[1]}={t:.3f}")
        for sdm in ("pwqd", "pwld"):
            for a, b, ratio in (("quad4", "quad8", 2.0), ("quad8", "quad16", 2.0),
                                ("quad16", "quad32", 2.0), ("tri32", "tri2048", 8.0)):
                order = math.log(errors[(sdm, a)] / errors[(sdm, b)]) / math.log(ratio)
                print(f"Order-{sdm}-{a}-{b}={order:.3f}")
