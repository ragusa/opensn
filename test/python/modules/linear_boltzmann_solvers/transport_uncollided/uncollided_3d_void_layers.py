#!/usr/bin/env python3
"""Uncollided flux through a void and a near-void layer.

A box [-1, 1]^3 is made of four layers normal to x: an absorber, a void (sigma_t = 0), a
near-void (sigma_t = 1e-8) and a second absorber. A point source sits in the first absorber and
the whole mesh is ray traced.

The optical depth from the source to a point is the sum over the layers of sigma_t times the
length of the straight path in the layer, which is the length of the path times the fraction of
its x extent that lies in the layer. The flux is then the point kernel
S exp(-tau) / (4 pi r^2), evaluated here independently of the mesh.

The conservation correction of a ray-traced cell scales its flux and its outgoing leakage by
(inflow + source) / (outflow + sigma_t * integral of the flux). In a void this is
inflow / outflow, which is well defined, so the balance must close in every cell. The test
checks the flux against the point kernel in every layer, and tests.json checks the global
balance that the solver reports, production = removal + outflow, to round-off.
"""

import importlib
import math
import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.fieldfunc import FieldFunctionInterpolationPoint
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.math import Vector3
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.solver import UncollidedProblem, UncollidedSolver
    from pyopensn.source import PointSource
    from pyopensn.xs import MultiGroupXS

sys.path.append(os.path.dirname(__file__))
uncollided_utils = importlib.import_module("uncollided_unstructured_utils")
point_value = uncollided_utils.point_value
relative_error = uncollided_utils.relative_error
remove_file = uncollided_utils.remove_file

# Optional argument: the ray tracer of the uncollided problem.
if "ray_tracer" not in globals():
    ray_tracer = "cell_walk"


# Layers as (xmin, xmax, sigma_t).
LAYERS = [(-1.0, -0.25, 0.8), (-0.25, 0.25, 0.0), (0.25, 0.5, 1.0e-8), (0.5, 1.0, 1.5)]
SOURCE = (-0.6127, 0.0713, -0.0389)
STRENGTH = 1.0


def exact_flux(point):
    radius = math.dist(SOURCE, point)
    x0, x1 = sorted((SOURCE[0], point[0]))
    optical_depth = 0.0
    for xmin, xmax, sigma_t in LAYERS:
        overlap = max(0.0, min(x1, xmax) - max(x0, xmin))
        optical_depth += sigma_t * radius * overlap / (x1 - x0)
    return STRENGTH * math.exp(-optical_depth) / (4.0 * math.pi * radius * radius)


if __name__ == "__main__":
    if size != 1:
        sys.exit(f"Expected one process, got {size}.")

    nodes = [-1.0 + 0.125 * i for i in range(17)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes] * 3).Execute()
    xs_map = []
    for block_id, (xmin, xmax, sigma_t) in enumerate(LAYERS):
        grid.SetBlockIDFromLogicalVolume(
            RPPLogicalVolume(xmin=xmin, xmax=xmax, infy=True, infz=True), block_id, True
        )
        xs = MultiGroupXS()
        xs.CreateSimpleOneGroup(sigma_t=sigma_t, c=0.0)
        xs_map.append({"block_ids": [block_id], "xs": xs})

    file_name = "uncollided_3d_void_layers.h5"
    remove_file(file_name)
    problem = UncollidedProblem(
        mesh=grid,
        num_groups=1,
        groupsets=[{"groups_from_to": [0, 0]}],
        xs_map=xs_map,
        point_sources=[PointSource(location=list(SOURCE), strength=[STRENGTH])],
        near_source=[
            RPPLogicalVolume(xmin=-1.1, xmax=1.1, ymin=-1.1, ymax=1.1, zmin=-1.1, zmax=1.1)
        ],
        exact_near_source_leakage=True,
        scattering_order=0,
        ray_tracer=ray_tracer,
    )
    solver = UncollidedSolver(problem=problem, file_name=file_name)
    solver.Initialize()
    solver.Execute()

    # Two points in each layer, away from the source cell.
    sample_points = [
        (-0.31, 0.44, 0.19),
        (-0.87, -0.52, 0.63),
        (-0.06, -0.31, 0.56),
        (0.19, 0.69, -0.44),
        (0.31, 0.19, 0.06),
        (0.44, -0.81, -0.69),
        (0.56, 0.06, -0.19),
        (0.94, 0.81, 0.69),
    ]
    scalar_flux = problem.GetScalarFluxFieldFunction()[0]
    max_error = 0.0
    for point in sample_points:
        value = point_value(scalar_flux, point, FieldFunctionInterpolationPoint, Vector3)
        max_error = max(max_error, relative_error(value, exact_flux(point)))

    if rank == 0:
        print(f"UncollidedVoidLayersMaxRelativeError={max_error:.8e}")
        sys.stdout.flush()

    remove_file(file_name)
    if max_error > 0.05:
        raise RuntimeError(f"Void-layer scalar-flux error is too large: {max_error}")
