#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Method of manufactured solutions for the serendipity tying of the PWQD basis
(spatial_discretization="pwqd", pwqd_tying="serendipity"), with the least-squares tying as the
control. sigma_t = 1, c = 0.5, GLC 4x8 product quadratures with scattering_order 1, angular
volumetric sources as in transport_3d_mms_angular_source.py.

1. Patch tests, exact to iteration tolerance (the serendipity functions reproduce every quadratic
   on cells with planar faces and simple vertices): quadratic phi with reflecting boundaries and
   with the exact inflow through spatially varying arbitrary boundaries on 2x2x2 hexahedra, and
   with reflecting boundaries on 2x2 quadrilaterals and the 107-cell polygon mix
   quad_mesh_poly_mix.obj.
2. Convergence with vacuum boundaries, phi = sin(pi x) sin(pi y) [sin(pi z)]: both tyings on
   4^3 / 8^3 hexahedra of the unit cube and on 8^2 / 16^2 / 32^2 quadrilaterals of [-1, 1]^2.
   Both are third order; the serendipity tying is the more accurate one (the prototype measured
   a factor 1.7 on 6^3 hexahedra), reported as the error ratio serendipity / least squares.
SDM: PWQD with both tyings, AAH sweeps.
"""

import math
import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator, FromFileMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature3DXYZ, GLCProductQuadrature2DXY
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.math import VectorSpatialAngularFunction

ASSETS = "../../../../assets/mesh/"
SIGMA_T = 1.0
C = 0.5
SIGMA_A = SIGMA_T * (1.0 - C)


def quadratic_phi(p):
    return (1.0 + p.x + 2.0 * p.y + 3.0 * p.z + p.x * p.x - p.x * p.y + 2.0 * p.z * p.z
            + 0.5 * p.y * p.z)


def quadratic_source(p, omega, n):
    gx = 1.0 + 2.0 * p.x - p.y
    gy = 2.0 - p.x + 0.5 * p.z
    gz = 3.0 + 4.0 * p.z + 0.5 * p.y
    return [omega.x * gx + omega.y * gy + omega.z * gz + SIGMA_A * quadratic_phi(p)] * n


def quadratic_phi_2d(p):
    return 1.0 + p.x + 2.0 * p.y + p.x * p.x - p.x * p.y + 2.0 * p.y * p.y


def quadratic_source_2d(p, omega, n):
    gx = 1.0 + 2.0 * p.x - p.y
    gy = 2.0 - p.x + 4.0 * p.y
    return [omega.x * gx + omega.y * gy + SIGMA_A * quadratic_phi_2d(p)] * n


def sine_phi(p):
    return math.sin(math.pi * p.x) * math.sin(math.pi * p.y) * math.sin(math.pi * p.z)


def sine_source(p, omega, n):
    sx, sy, sz = math.sin(math.pi * p.x), math.sin(math.pi * p.y), math.sin(math.pi * p.z)
    cx, cy, cz = math.cos(math.pi * p.x), math.cos(math.pi * p.y), math.cos(math.pi * p.z)
    streaming = math.pi * (omega.x * cx * sy * sz + omega.y * sx * cy * sz
                           + omega.z * sx * sy * cz)
    return [streaming + SIGMA_A * sx * sy * sz] * n


def sine_phi_2d(p):
    return math.sin(math.pi * p.x) * math.sin(math.pi * p.y)


def sine_source_2d(p, omega, n):
    sx, sy = math.sin(math.pi * p.x), math.sin(math.pi * p.y)
    cx, cy = math.cos(math.pi * p.x), math.cos(math.pi * p.y)
    return [math.pi * (omega.x * cx * sy + omega.y * sx * cy) + SIGMA_A * sx * sy] * n


if __name__ == "__main__":

    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=SIGMA_T, c=C)
    quad3d = GLCProductQuadrature3DXYZ(n_polar=4, n_azimuthal=8, scattering_order=1)
    quad2d = GLCProductQuadrature2DXY(n_polar=4, n_azimuthal=8, scattering_order=1)
    whole = RPPLogicalVolume(infx=True, infy=True, infz=True)

    def hex_grid(N):
        nodes = [i / N for i in range(N + 1)]
        grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes]).Execute()
        grid.SetUniformBlockID(0)
        return grid

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

    def solve(grid, tying, source, bc_type, dim3, inflow=None):
        src = VolumetricSource(block_ids=[0], angular_func=VectorSpatialAngularFunction(source))
        names = ("xmin", "xmax", "ymin", "ymax") + (("zmin", "zmax") if dim3 else ())
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
                    "angular_quadrature": quad3d if dim3 else quad2d,
                    "inner_linear_method": "petsc_gmres",
                    "l_abs_tol": 1.0e-12,
                    "l_max_its": 500,
                    "gmres_restart_interval": 50,
                }
            ],
            xs_map=[{"block_ids": [0], "xs": xs}],
            volumetric_sources=[src],
            boundary_conditions=bcs,
            spatial_discretization="pwqd",
            pwqd_tying=tying,
        )
        solver = SteadyStateSourceSolver(problem=phys)
        solver.Initialize()
        solver.Execute()
        return phys.GetScalarFluxFieldFunction()[0]

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
    for label, grid, dim3 in (("hexes", hex_grid(2), True), ("quads", quad_grid(2), False),
                              ("polygons", file_grid("quad_mesh_poly_mix.obj"), False)):
        source, exact = (quadratic_source, quadratic_phi) if dim3 else (quadratic_source_2d,
                                                                        quadratic_phi_2d)
        ff = solve(grid, "serendipity", source, "reflecting", dim3)
        patch[label] = integrate(ff, "max_func", lambda v, b, p: abs(v - exact(p)))
    ff = solve(hex_grid(2), "serendipity", quadratic_source, None, True,
               inflow=lambda p, omega, n: [quadratic_phi(p)] * n)
    patch["hexes-inflow"] = integrate(ff, "max_func", lambda v, b, p: abs(v - quadratic_phi(p)))

    # 2. Sine MMS convergence
    errors = {}
    cases = [(f"hex{N}", hex_grid(N), sine_source, sine_phi, True) for N in (4, 8)]
    cases += [(f"quad{N}", quad_grid(N), sine_source_2d, sine_phi_2d, False) for N in (8, 16, 32)]
    for tying in ("serendipity", "least_squares"):
        for label, grid, source, exact, dim3 in cases:
            ff = solve(grid, tying, source, "vacuum", dim3)
            err2 = integrate(ff, "sum_func", lambda v, b, p: (v - exact(p)) ** 2)
            norm2 = integrate(ff, "sum_func", lambda v, b, p: exact(p) ** 2)
            errors[(tying, label)] = math.sqrt(err2 / norm2)

    if rank == 0:
        for label, e in patch.items():
            print(f"PatchMaxNodalErr-serendipity-{label}={e:.3e}")
        for key, e in errors.items():
            print(f"RelL2Err-{key[0]}-{key[1]}={e:.4e}")
        for tying in ("serendipity", "least_squares"):
            for a, b in (("hex4", "hex8"), ("quad8", "quad16"), ("quad16", "quad32")):
                order = math.log(errors[(tying, a)] / errors[(tying, b)]) / math.log(2)
                print(f"Order-{tying}-{a}-{b}={order:.3f}")
        for label in ("hex8", "quad32"):
            ratio = errors[("serendipity", label)] / errors[("least_squares", label)]
            print(f"ErrRatio-serendipity-over-lsq-{label}={ratio:.4f}")
