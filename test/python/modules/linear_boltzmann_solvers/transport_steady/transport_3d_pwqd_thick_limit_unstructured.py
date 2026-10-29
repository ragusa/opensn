#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thick diffusion limit (sigma_t = 1/eps, sigma_a = eps, q = eps, eps = 1e-4, vacuum, GLC 2x4,
within-group MIP DSA) on cells that are not orthogonal hexahedra, for PWQD and PWLD:
  tets       Kuhn subdivision (6 tets per cube) of the 4^3 and 8^3 unit-cube grids
             (pwqd_unit_cube_tets_{4,8}.vtu);
  perturbed  4^3 and 8^3 hexes of the unit cube with every interior vertex displaced randomly by
             up to 0.2 h, so faces are non-planar (pwqd_unit_cube_perturbed_{4,8}.vtu);
  prisms     the [-1,1]^2 triangle mesh triangle_mesh2x2_cuts.obj extruded into one and two layers
             of unit height (686 and 1372 prisms).
Reference: the eps -> 0 limit, the diffusion solution of -(1/3) Lap u + u = 1 with u = 0 on the
boundary of the box, as a triple sine series. The error is measured at the mesh vertices with the
point interpolator (a discrete nodal norm; the volume interpolator cannot evaluate a function of
position): the root-mean-square and maximum vertex error relative to the peak of the reference,
and the observed order of the RMS error between the two refinements of the tets and of the
perturbed hexes (the prisms are refined in z only, so no order is formed for them). The orders
are limited at the finer meshes by the O(eps) distance between the transport solution and the
limit solution (about 1e-3 of the peak at eps = 1e-4), where PWQD's errors already sit.

This test runs on one rank with a direct DSA solve, for two reasons that are independent of
the basis and were established by running it at 1, 2 and 4 ranks (2026-10-28):
  1. with several ranks the automatic DSA policy solves the MIP system with CG/BoomerAMG, which
     does not converge on these stiff systems (sigma_t h ~ 1e3; 200 iterations with residuals
     of 1e2 for a right-hand side of 1e-5 on the perturbed hexes), so the preconditioner is
     inexact while the outer GMRES still reports a converged preconditioned residual; the
     answer then depends on the partition. With wgdsa_solver_policy = "direct" the 4-rank
     answer equals the 1-rank one to every digit;
  2. on the 8^3 meshes the partitioner creates cyclic sweep dependencies across ranks whose
     lagged fluxes (0.4 to 2.5 % of the unknowns) stall the GMRES iteration at this optical
     thickness even with the direct DSA (500 iterations without convergence for PWLD and PWQD
     alike), whereas eps = 1e-2 converges in 40 iterations.
At one rank the discretization and the sweeps are unaffected, and the same solutions are
obtained with CBC sweeps.
SDM: PWLD and PWQD, AAH sweeps.
"""

import math
import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import ExtruderMeshGenerator, FromFileMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationPoint
    from pyopensn.math import Vector3

ASSETS = "../../../../assets/mesh/"


def limit_solution(box, modes=41):
    """u(x) on the box [x0,x1]x[y0,y1]x[z0,z1] (list of three (lo, hi) pairs)."""
    L = [hi - lo for lo, hi in box]
    odd = range(1, modes + 1, 2)
    coef = {}
    for l in odd:
        for m in odd:
            for p in odd:
                coef[(l, m, p)] = (64.0 / (math.pi ** 3 * l * m * p)) / (
                    1.0 + math.pi ** 2 * ((l / L[0]) ** 2 + (m / L[1]) ** 2 + (p / L[2]) ** 2) / 3.0)

    def u(x, y, z):
        sx = {l: math.sin(l * math.pi * (x - box[0][0]) / L[0]) for l in odd}
        sy = {m: math.sin(m * math.pi * (y - box[1][0]) / L[1]) for m in odd}
        sz = {p: math.sin(p * math.pi * (z - box[2][0]) / L[2]) for p in odd}
        return sum(c * sx[l] * sy[m] * sz[p] for (l, m, p), c in coef.items())

    # peak at the centre
    peak = u(*(0.5 * (lo + hi) for lo, hi in box))
    return u, peak


def vtu_points(path):
    """Vertex coordinates of an ASCII VTU written by the asset generator (Points array)."""
    txt = open(path).read()
    start = txt.index('Name="Points"')
    start = txt.index(">", start) + 1
    end = txt.index("</DataArray>", start)
    vals = [float(v) for v in txt[start:end].split()]
    return [vals[i:i + 3] for i in range(0, len(vals), 3)]


def obj_points(path, zlevels):
    pts = []
    for line in open(path):
        if line.startswith("v "):
            x, y = [float(v) for v in line.split()[1:3]]
            for z in zlevels:
                pts.append([x, y, z])
    return pts


def run(grid, sdm):
    eps = 1.0e-4
    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=1.0 / eps, c=1.0 - eps * eps)
    src = VolumetricSource(block_ids=[0], group_strength=[eps])
    pquad = GLCProductQuadrature3DXYZ(n_polar=2, n_azimuthal=4, scattering_order=0)
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
                "wgdsa_solver_policy": "direct",
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
    solver.Execute()
    return phys.GetScalarFluxFieldFunction()[0]


def vertex_errors(ff, points, u, peak):
    """(rms, max) of |phi(x_v) - u(x_v)| / peak over the given vertices."""
    ffi = FieldFunctionInterpolationPoint()
    ffi.SetFieldFunction(ff)
    sq = 0.0
    mx = 0.0
    for p in points:
        ffi.SetPointOfInterest(point=Vector3(p[0], p[1], p[2]))
        ffi.Execute()
        e = abs(ffi.GetPointValue() - u(*p))
        sq += e * e
        mx = max(mx, e)
    return math.sqrt(sq / len(points)) / peak, mx / peak


if __name__ == "__main__":

    unit = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    u_unit, peak_unit = limit_solution(unit)
    prism_box = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]
    u_prism, peak_prism = limit_solution(prism_box)

    cases = []
    for N in (4, 8):
        cases.append((f"tets-{N}", lambda N=N: FromFileMeshGenerator(
            filename=ASSETS + f"pwqd_unit_cube_tets_{N}.vtu").Execute(),
            vtu_points(ASSETS + f"pwqd_unit_cube_tets_{N}.vtu"), u_unit, peak_unit))
        cases.append((f"perturbed-{N}", lambda N=N: FromFileMeshGenerator(
            filename=ASSETS + f"pwqd_unit_cube_perturbed_{N}.vtu").Execute(),
            vtu_points(ASSETS + f"pwqd_unit_cube_perturbed_{N}.vtu"), u_unit, peak_unit))
    for n in (1, 2):
        cases.append((f"prisms-{n}", lambda n=n: ExtruderMeshGenerator(
            inputs=[FromFileMeshGenerator(filename=ASSETS + "triangle_mesh2x2_cuts.obj")],
            layers=[{"z": 1.0, "n": n}]).Execute(),
            obj_points(ASSETS + "triangle_mesh2x2_cuts.obj", [i / n for i in range(n + 1)]),
            u_prism, peak_prism))

    results = {}
    for name, make_grid, points, u, peak in cases:
        grid = make_grid()
        grid.SetOrthogonalBoundaries()
        grid.SetUniformBlockID(0)
        for sdm in ("pwld", "pwqd"):
            ff = run(grid, sdm)
            results[(name, sdm)] = vertex_errors(ff, points, u, peak)

    if rank == 0:
        for name, _, _, _, _ in cases:
            for sdm in ("pwld", "pwqd"):
                rms, mx = results[(name, sdm)]
                print(f"RelRmsVertexErr-{sdm}-{name}={rms:.4e}")
                print(f"RelMaxVertexErr-{sdm}-{name}={mx:.4e}")
        for family, (a, b) in (("tets", (4, 8)), ("perturbed", (4, 8))):
            for sdm in ("pwld", "pwqd"):
                ra = results[(f"{family}-{a}", sdm)][0]
                rb = results[(f"{family}-{b}", sdm)][0]
                print(f"Order-{sdm}-{family}={math.log(ra / rb) / math.log(2.0):.3f}")
