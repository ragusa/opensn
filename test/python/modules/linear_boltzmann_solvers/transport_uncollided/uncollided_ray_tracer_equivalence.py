#!/usr/bin/env python3
"""The cell-walk and material-hull ray tracers give the same uncollided flux.

Both tracers evaluate the same integral, the optical depth of the straight path between a
quadrature point and the source, at the same points. The cell walk sums sigma_t times the chord
over the cells along the path; the material-hull tracer sums sigma_t times the length of the path
in each material. The two sums are equal, so the nodal fluxes may differ by round-off only. The
problem has three materials, a reflecting boundary whose image source is traced along
folded paths, and exact integration next to the source, and the whole mesh is ray traced so that
every cell is compared.

The source position is generic. When a path passes exactly through an edge or a vertex of the
mesh, which happens on regular meshes for sources at simple coordinates, the cell walk finds no
exit face and continues from a displaced point, so its optical depth is then approximate while
the material-hull tracer remains exact (see the SegmentTracerTest unit tests).

The accuracy of the flux itself is checked against analytic solutions by the other uncollided
tests, which run with both tracers. This test adds two statements:

- cell walk against material hull: relative difference at round-off, with a bound of 1e-11. The
  optical depths reach about 5, each is a sum of fewer than 30 terms, and the fluxes are
  combined in dense solves on each cell, which leaves several orders of margin over 1e-16;
- bounding volume hierarchy against brute force: no difference at all, because both apply the
  same triangle test to the triangles a path crosses.
"""

import os
import sys

import numpy as np

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.solver import UncollidedProblem, UncollidedSolver
    from pyopensn.source import PointSource
    from pyopensn.xs import MultiGroupXS


def make_xs(sigma_t):
    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=sigma_t, c=0.0)
    return xs


def solve(grid, xs_map, ray_tracer, hull_tracer_backend):
    file_name = f"uncollided_ray_tracer_equivalence_{ray_tracer}_{hull_tracer_backend}.h5"
    problem = UncollidedProblem(
        mesh=grid,
        num_groups=1,
        groupsets=[{"groups_from_to": [0, 0]}],
        xs_map=xs_map,
        point_sources=[PointSource(location=[0.3127, -0.1713, 0.1189], strength=[1.0])],
        near_source=[
            RPPLogicalVolume(xmin=-1.1, xmax=1.1, ymin=-1.1, ymax=1.1, zmin=-1.1, zmax=1.1)
        ],
        boundary_conditions=[{"name": "xmin", "type": "reflecting"}],
        exact_near_source_leakage=True,
        scattering_order=0,
        ray_tracer=ray_tracer,
        hull_tracer_backend=hull_tracer_backend,
    )
    solver = UncollidedSolver(problem=problem, file_name=file_name, progress_interval=0)
    solver.Initialize()
    solver.Execute()
    phi = np.array(problem.GetPhiNewLocal())
    try:
        os.remove(file_name)
    except FileNotFoundError:
        pass
    return phi


if __name__ == "__main__":
    if size != 1:
        sys.exit(f"Expected one process, got {size}.")

    nodes = [-1.0 + 0.25 * i for i in range(9)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes] * 3).Execute()
    grid.SetUniformBlockID(0)
    # A slab and a block inside it, so that paths cross one, two or three materials and the
    # reflecting boundary meets two of them.
    grid.SetBlockIDFromLogicalVolume(
        RPPLogicalVolume(xmin=-2.0, xmax=2.0, ymin=0.25, ymax=0.75, zmin=-2.0, zmax=2.0), 1, True
    )
    grid.SetBlockIDFromLogicalVolume(
        RPPLogicalVolume(xmin=-0.5, xmax=0.25, ymin=0.25, ymax=0.5, zmin=-0.25, zmax=0.5), 2, True
    )
    grid.SetOrthogonalBoundaries()
    xs_map = [
        {"block_ids": [0], "xs": make_xs(0.4)},
        {"block_ids": [1], "xs": make_xs(2.5)},
        {"block_ids": [2], "xs": make_xs(0.05)},
    ]

    walk = solve(grid, xs_map, "cell_walk", "bvh")
    hull = solve(grid, xs_map, "material_hull", "bvh")
    brute_force = solve(grid, xs_map, "material_hull", "brute_force")

    scale = np.abs(walk).max()
    walk_hull_difference = np.abs(walk - hull).max() / scale
    backend_difference = np.abs(hull - brute_force).max() / scale

    if rank == 0:
        print(f"RayTracerEquivalenceWalkHullMaxRelDiff={walk_hull_difference:.8e}")
        print(f"RayTracerEquivalenceBackendMaxRelDiff={backend_difference:.8e}")
        print(f"RayTracerEquivalenceFluxNonzero={int(scale > 0.0)}")
        sys.stdout.flush()

    if walk_hull_difference > 1.0e-11:
        raise RuntimeError(
            f"The cell-walk and material-hull fluxes differ by {walk_hull_difference}."
        )
    if backend_difference != 0.0:
        raise RuntimeError(f"The two hull tracers differ by {backend_difference}.")
