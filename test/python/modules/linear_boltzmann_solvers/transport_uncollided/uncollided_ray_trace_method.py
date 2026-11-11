#!/usr/bin/env python3
"""The "ray_trace" method ray traces every cell.

With method="sweep", the cells whose centroid lies in the near-source region are ray traced and
corrected for conservation, and the others are swept. A near-source region that covers the mesh
leaves no cell to sweep. method="ray_trace" asks for exactly that without a region, so the two
must give the same nodal flux, bit for bit, with either ray tracer.

The accuracy of the fully ray-traced flux is checked against analytic solutions by
uncollided_source_cell_average.py, uncollided_3d_structured_analytic.py,
uncollided_3d_void_layers.py and uncollided_boundary_source_prism.py, whose near-source regions
cover their meshes.

The test also checks that a near-source region is rejected with method="ray_trace".
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


def make_problem(grid, xs_map, ray_tracer, method, with_region):
    arguments = dict(
        mesh=grid,
        num_groups=1,
        groupsets=[{"groups_from_to": [0, 0]}],
        xs_map=xs_map,
        point_sources=[PointSource(location=[0.31, -0.17, 0.12], strength=[1.0])],
        boundary_conditions=[{"name": "xmin", "type": "reflecting"}],
        exact_near_source_leakage=True,
        scattering_order=0,
        ray_tracer=ray_tracer,
        method=method,
    )
    if with_region:
        arguments["near_source"] = [
            RPPLogicalVolume(xmin=-1.1, xmax=1.1, ymin=-1.1, ymax=1.1, zmin=-1.1, zmax=1.1)
        ]
    return UncollidedProblem(**arguments)


def solve(grid, xs_map, ray_tracer, method):
    file_name = f"uncollided_ray_trace_method_{ray_tracer}_{method}.h5"
    problem = make_problem(grid, xs_map, ray_tracer, method, with_region=(method == "sweep"))
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
    grid.SetBlockIDFromLogicalVolume(
        RPPLogicalVolume(xmin=-2.0, xmax=2.0, ymin=0.25, ymax=0.75, zmin=-2.0, zmax=2.0), 1, True
    )
    grid.SetOrthogonalBoundaries()
    xs_map = [{"block_ids": [0], "xs": make_xs(0.4)}, {"block_ids": [1], "xs": make_xs(2.5)}]

    identical = {}
    nonzero = True
    for ray_tracer in ("cell_walk", "material_hull"):
        with_region = solve(grid, xs_map, ray_tracer, "sweep")
        without_region = solve(grid, xs_map, ray_tracer, "ray_trace")
        identical[ray_tracer] = bool(np.array_equal(with_region, without_region))
        nonzero = nonzero and bool(np.abs(without_region).max() > 0.0)

    region_rejected = False
    try:
        make_problem(grid, xs_map, "cell_walk", "ray_trace", with_region=True)
    except Exception as error:
        region_rejected = "near-source logical volumes are not used" in str(error)

    if rank == 0:
        print(f"RayTraceMethodCellWalkIdentical={int(identical['cell_walk'])}")
        print(f"RayTraceMethodMaterialHullIdentical={int(identical['material_hull'])}")
        print(f"RayTraceMethodFluxNonzero={int(nonzero)}")
        print(f"RayTraceMethodRegionRejected={int(region_rejected)}")
        sys.stdout.flush()

    if not all(identical.values()):
        raise RuntimeError("method='ray_trace' differs from a near-source region over the mesh.")
    if not region_rejected:
        raise RuntimeError("method='ray_trace' accepted a near-source region.")
