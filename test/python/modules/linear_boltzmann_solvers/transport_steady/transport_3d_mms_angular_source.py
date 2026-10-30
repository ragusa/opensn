#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Method of manufactured solutions with an angular volumetric source (VolumetricSource with
angular_func): for a manufactured angular flux psi(x, Omega) the source
    q = Omega . grad psi + sigma_t psi - sigma_s phi
is direction dependent; it is projected onto the moments of the groupset quadrature, which is
exact when its angular content lies in the harmonics kept (scattering_order 1 suffices for an
isotropic manufactured flux, whose streaming term is linear in Omega). Unit cube, sigma_t = 1,
c = 0.5, GLC 4x8 product quadrature, PWLD.

1. Patch test: psi = phi(x) = 1 + x + 2y + 3z (isotropic, linear in space) on 2x2x2 hexes with
   reflecting boundaries (satisfied exactly by an isotropic flux). The linear functions are in
   the PWLD space and the source is nodally exact, so the DG solution must reproduce phi at
   every node to iteration tolerance: this anchors the normalisation of the angular source, the
   discrete-to-moment / moment-to-discrete round trip and the reflecting boundary at once.
   The same patch test with the exact incoming flux prescribed on every boundary through an
   "arbitrary" boundary with a spatial_function (psi_inc = phi at the boundary face nodes)
   anchors the spatially varying boundary condition: face-node locations and normalisation.
2. Convergence: psi = phi(x) = sin(pi x) sin(pi y) sin(pi z) (zero on the boundary, so vacuum
   boundaries are exact) on 4^3, 8^3, 16^3 hexes; the L2 error against the exact solution is
   integrated with the volume interpolator's spatial operation function. PWLD is second order.
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
    from pyopensn.math import VectorSpatialAngularFunction

SIGMA_T = 1.0
C = 0.5
SIGMA_A = SIGMA_T * (1.0 - C)


def linear_phi(p):
    return 1.0 + p.x + 2.0 * p.y + 3.0 * p.z


def linear_source(p, omega, n):
    streaming = omega.x + 2.0 * omega.y + 3.0 * omega.z
    return [streaming + SIGMA_A * linear_phi(p)] * n


def linear_source_batched(p, omegas, n):
    """Batched form of linear_source: all directions of a node in one call, direction-major."""
    return [omega.x + 2.0 * omega.y + 3.0 * omega.z + SIGMA_A * linear_phi(p)
            for omega in omegas for _ in range(n)]


def sine_phi(p):
    return math.sin(math.pi * p.x) * math.sin(math.pi * p.y) * math.sin(math.pi * p.z)


def sine_source(p, omega, n):
    sx, sy, sz = math.sin(math.pi * p.x), math.sin(math.pi * p.y), math.sin(math.pi * p.z)
    cx, cy, cz = math.cos(math.pi * p.x), math.cos(math.pi * p.y), math.cos(math.pi * p.z)
    streaming = math.pi * (omega.x * cx * sy * sz + omega.y * sx * cy * sz
                           + omega.z * sx * sy * cz)
    return [streaming + SIGMA_A * sx * sy * sz] * n


if __name__ == "__main__":

    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=SIGMA_T, c=C)
    pquad = GLCProductQuadrature3DXYZ(n_polar=4, n_azimuthal=8, scattering_order=1)
    whole = RPPLogicalVolume(infx=True, infy=True, infz=True)

    def solve(N, source, bc_type, batched=False, inflow=None):
        """bc_type: "reflecting" or "vacuum" on every boundary; with inflow (a function of
        (point, omega, num_groups)) every boundary is "arbitrary" with that spatial_function."""
        nodes = [i / N for i in range(N + 1)]
        grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes]).Execute()
        grid.SetUniformBlockID(0)
        src = VolumetricSource(block_ids=[0],
                               angular_func=VectorSpatialAngularFunction(source, batched=batched))
        names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
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
        )
        solver = SteadyStateSourceSolver(problem=phys)
        solver.Initialize()
        solver.Execute()
        return phys, phys.GetScalarFluxFieldFunction()[0]

    def integrate(ff, op, func):
        ffi = FieldFunctionInterpolationVolume()
        ffi.SetOperationType(op)
        ffi.SetLogicalVolume(whole)
        ffi.SetFieldFunction(ff)
        ffi.SetSpatialOperationFunction(func)
        ffi.Execute()
        return ffi.GetValue()

    # 1. Linear patch test, with the per-direction and the batched callback forms
    phys, ff = solve(2, linear_source, "reflecting")
    patch_max_err = integrate(ff, "max_func", lambda v, b, p: abs(v - linear_phi(p)))
    phys, ff = solve(2, linear_source_batched, "reflecting", batched=True)
    patch_max_err_batched = integrate(ff, "max_func", lambda v, b, p: abs(v - linear_phi(p)))
    # and with the exact incoming flux prescribed on every boundary (spatially varying inflow)
    phys, ff = solve(2, linear_source, None, inflow=lambda p, omega, n: [linear_phi(p)] * n)
    patch_max_err_inflow = integrate(ff, "max_func", lambda v, b, p: abs(v - linear_phi(p)))

    # 2. Sine MMS convergence
    errors = {}
    for N in (4, 8, 16):
        phys, ff = solve(N, sine_source, "vacuum")
        err2 = integrate(ff, "sum_func", lambda v, b, p: (v - sine_phi(p)) ** 2)
        norm2 = integrate(ff, "sum_func", lambda v, b, p: sine_phi(p) ** 2)
        errors[N] = math.sqrt(err2 / norm2)

    if rank == 0:
        print(f"PatchMaxNodalErr-pwld={patch_max_err:.3e}")
        print(f"PatchMaxNodalErr-pwld-batched={patch_max_err_batched:.3e}")
        print(f"PatchMaxNodalErr-pwld-inflow={patch_max_err_inflow:.3e}")
        for N, e in errors.items():
            print(f"RelL2Err-pwld-{N}={e:.4e}")
        for a, b in ((4, 8), (8, 16)):
            print(f"Order-pwld-{a}-{b}={math.log(errors[a] / errors[b]) / math.log(2):.3f}")
