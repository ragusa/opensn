#!/usr/bin/env python3
"""Point source on the domain boundary, at a point shared by several cells.

A source on a corner, an edge or a face of a homogeneous box emits only part of its strength
into the box: one eighth, one quarter or one half. The rest leaves the domain directly. Inside
the box the uncollided flux is the point kernel S exp(-sigma_t r) / (4 pi r^2) of the full
strength.

When the source point is shared by several cells, each of them receives the part of the source
emitted into the solid angle it subtends, and the faces between these cells, whose planes
contain the source, carry no current. The argument location selects the case:

- "corner": the corner (0, 0, 0) of a cube of tetrahedra, where several tetrahedra meet;
- "far_corner": the opposite corner (0.032, 0.032, 0.032). Its coordinates are not exactly
  representable, so the centroids of the faces through the source are not exactly in their
  planes;
- "edge": the middle of a boundary edge of a box of hexahedra, shared by two cells.

The test checks the flux against the point kernel at sample points, and the outflow against the
part that leaves directly plus the escape probability of the part emitted into the box, computed
independently by integrating exp(-sigma_t R) over the directions into the box. tests.json also
checks the balance the solver reports.
"""

import importlib
import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.fieldfunc import (
        FieldFunctionInterpolationPoint,
        FieldFunctionInterpolationVolume,
    )
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.math import Vector3
    from pyopensn.mesh import FromFileMeshGenerator, OrthogonalMeshGenerator
    from pyopensn.solver import UncollidedProblem, UncollidedSolver
    from pyopensn.source import PointSource
    from pyopensn.xs import MultiGroupXS

sys.path.append(os.path.dirname(__file__))
uncollided_utils = importlib.import_module("uncollided_unstructured_utils")
mesh_path = uncollided_utils.mesh_path
point_value = uncollided_utils.point_value
relative_error = uncollided_utils.relative_error
remove_file = uncollided_utils.remove_file
uncollided_3d = uncollided_utils.uncollided_3d
uncollided_escape_3d_box = uncollided_utils.uncollided_escape_3d_box
volume_integral = uncollided_utils.volume_integral

# Optional arguments: the source location and the options of the uncollided problem.
if "location" not in globals():
    location = "corner"
if "ray_tracer" not in globals():
    ray_tracer = "cell_walk"
if "method" not in globals():
    method = "sweep"


if __name__ == "__main__":
    if size != 1:
        sys.exit(f"Expected one process, got {size}.")

    if location in ("corner", "far_corner"):
        side = 0.032
        sigma_t = 35.0
        grid = FromFileMeshGenerator(filename=mesh_path("uncollided_cube_coarse.msh")).Execute()
        bounds = [(0.0, side)] * 3
        if location == "corner":
            source = (0.0, 0.0, 0.0)
            sample_points = [(0.0052, 0.0061, 0.0263), (0.0241, 0.0093, 0.0077),
                             (0.0284, 0.0276, 0.0251), (0.0113, 0.0171, 0.0142)]
        else:
            source = (side, side, side)
            sample_points = [(0.0052, 0.0061, 0.0263), (0.0241, 0.0093, 0.0077),
                             (0.0036, 0.0049, 0.0068), (0.0113, 0.0171, 0.0142)]
        emitted_fraction = 0.125
    elif location == "edge":
        sigma_t = 0.7
        nodes = [0.25 * i for i in range(9)]
        grid = OrthogonalMeshGenerator(node_sets=[nodes] * 3).Execute()
        bounds = [(0.0, 2.0)] * 3
        # The vertex (0, 0.5, 0) lies on the boundary edge x = 0, z = 0, where two cells meet.
        source = (0.0, 0.5, 0.0)
        sample_points = [(0.46, 0.11, 0.37), (1.39, 0.43, 0.82), (0.22, 1.58, 1.21),
                         (1.71, 1.66, 1.87)]
        emitted_fraction = 0.25
    else:
        raise ValueError(f"Unknown location {location}")
    grid.SetUniformBlockID(0)

    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=sigma_t, c=0.0)

    arguments = {}
    if method == "sweep":
        # Half of the box around the source is ray traced; the rest is swept.
        near = []
        for coordinate, (lower, upper) in zip(source, bounds):
            middle = 0.5 * (lower + upper)
            margin = 0.01 * (upper - lower)
            if coordinate < middle:
                near.append((lower - margin, middle))
            else:
                near.append((middle, upper + margin))
        arguments["near_source"] = [
            RPPLogicalVolume(
                xmin=near[0][0],
                xmax=near[0][1],
                ymin=near[1][0],
                ymax=near[1][1],
                zmin=near[2][0],
                zmax=near[2][1],
            )
        ]

    file_name = f"uncollided_boundary_shared_source_{location}_{ray_tracer}_{method}.h5"
    remove_file(file_name)
    problem = UncollidedProblem(
        mesh=grid,
        num_groups=1,
        groupsets=[{"groups_from_to": [0, 0]}],
        xs_map=[{"block_ids": [0], "xs": xs}],
        point_sources=[PointSource(location=list(source), strength=[1.0])],
        exact_near_source_leakage=True,
        scattering_order=0,
        ray_tracer=ray_tracer,
        method=method,
        **arguments,
    )
    solver = UncollidedSolver(problem=problem, file_name=file_name)
    solver.Initialize()
    solver.Execute()

    scalar_flux = problem.GetScalarFluxFieldFunction()[0]
    whole_domain = RPPLogicalVolume(infx=True, infy=True, infz=True)
    scalar_integral = volume_integral(scalar_flux, whole_domain, FieldFunctionInterpolationVolume)
    computed_outflow = 1.0 - sigma_t * scalar_integral
    # The escape integral skips the faces that contain the source, so it covers the directions
    # into the box.
    reference_outflow = (1.0 - emitted_fraction) + uncollided_escape_3d_box(sigma_t, source, bounds)
    outflow_error = relative_error(computed_outflow, reference_outflow)

    max_error = 0.0
    for point in sample_points:
        value = point_value(scalar_flux, point, FieldFunctionInterpolationPoint, Vector3)
        exact = uncollided_3d(1.0, sigma_t, source, point)
        max_error = max(max_error, relative_error(value, exact))

    if rank == 0:
        print(f"BoundarySharedSourceOutflowRelativeError={outflow_error:.8e}")
        print(f"BoundarySharedSourceMaxRelativeError={max_error:.8e}")
        sys.stdout.flush()

    remove_file(file_name)
    if outflow_error > 5.0e-3:
        raise RuntimeError(f"Outflow error is too large: {outflow_error}")
    if max_error > 0.15:
        raise RuntimeError(f"Scalar-flux error is too large: {max_error}")
