#!/usr/bin/env python3
"""Point source on reflecting planes.

A reflecting plane through a point source is a plane of symmetry of the whole problem. The image
of the source across it coincides with the source, so no image is traced across it: the source
has the strength of the whole source, of which the fraction 2^-k is emitted into the domain when
it lies on k reflecting planes, and the uncollided flux in the domain is that of the whole
source. The production rate of the balance is the rate emitted into the domain.

The argument case selects the problem:

- "corner": the source at the corner (0, 0, 0) of the box, on the three reflecting planes
  x = 0, y = 0 and z = 0. No image is traced, and the flux is the point kernel of the whole
  source. With compare_vacuum=True the problem is also solved with vacuum planes, which gives
  the same flux: the script prints how many values of the flux differ by at least one bit
  between the two solves, over all ranks.
- "one_plane": the source on x = 0, with x = 0 and y = 0 reflecting: the flux is that of the
  source and of its image across y = 0. Tracing the images across x = 0 doubles it.
- "vertex": as "one_plane", with the source at a vertex of the mesh shared by four cells
  (hexahedra only). The folded paths to its image end at that vertex; the cell walk stops in a
  cell of the source, which tests.json checks through the number of paths walked again.
- "edge": the source on the edge x = y = 0, with x = 0, y = 0 and z = 0 reflecting: the flux is
  that of the source and of its image across z = 0.
- "two_sources": a source on x = 0 and a source inside the box, with x = 0 and y = 0
  reflecting: the first has one image, the second three.

The argument mesh_type is "hexahedra" (a box of two materials, layers normal to x, with the
reference computed through the layers and their mirror images) or "tetrahedra" (a homogeneous
cube, against the point kernel; corner and one_plane only). The arguments ray_tracer and method
select the uncollided method, and conservation_correction=False keeps the projected flux. With
split_source_cells=True on 2 or 4 ranks, the tetrahedra are partitioned by planes that separate
the four cells at the corner (0, 0, 0): they are then on 2 or 3 ranks, as the solver reports.

The script prints the largest relative error at sample points and the sum of the flux at those
points, which does not depend on the number of ranks. The problem with reflecting planes is
solved last, after the line "Solve with reflecting planes", from which tests.json reads the
balance table of the solver: production rate 2^-k per source, balance to round-off.
"""

import importlib
import math
import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank

    def MPIAllReduce(value):
        return MPI.COMM_WORLD.allreduce(value, op=MPI.SUM)

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.fieldfunc import FieldFunctionInterpolationPoint
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.math import Vector3
    from pyopensn.mesh import FromFileMeshGenerator, KBAGraphPartitioner, OrthogonalMeshGenerator
    from pyopensn.solver import UncollidedProblem, UncollidedSolver
    from pyopensn.source import PointSource
    from pyopensn.xs import MultiGroupXS

sys.path.append(os.path.dirname(__file__))
uncollided_utils = importlib.import_module("uncollided_unstructured_utils")
mesh_path = uncollided_utils.mesh_path
point_value = uncollided_utils.point_value
relative_error = uncollided_utils.relative_error
remove_file = uncollided_utils.remove_file

# Optional arguments.
if "case" not in globals():
    case = "corner"
if "mesh_type" not in globals():
    mesh_type = "hexahedra"
if "ray_tracer" not in globals():
    ray_tracer = "material_hull"
if "method" not in globals():
    method = "ray_trace"
if "compare_vacuum" not in globals():
    compare_vacuum = False
if "split_source_cells" not in globals():
    split_source_cells = False

PLANES = {"x": "xmin", "y": "ymin", "z": "zmin"}


def make_xs(sigma_t):
    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=sigma_t, c=0.0)
    return xs


def layered_kernel(source, point, layers):
    """Point kernel of a unit source through layers (lower, upper, sigma_t) normal to x."""
    radius = math.dist(source, point)
    x0, x1 = sorted((source[0], point[0]))
    optical_depth = 0.0
    for lower, upper, sigma_t in layers:
        overlap = max(0.0, min(x1, upper) - max(x0, lower))
        if x1 > x0:
            optical_depth += sigma_t * radius * overlap / (x1 - x0)
        elif lower <= x0 <= upper:
            # A path in a plane x = constant lies in one layer.
            optical_depth = sigma_t * radius
    return math.exp(-optical_depth) / (4.0 * math.pi * radius * radius)


def images(source, planes):
    """The source and its images across the reflecting planes that do not contain it."""
    axes = [axis for axis in planes if source["xyz".index(axis)] != 0.0]
    points = []
    for mask in range(1 << len(axes)):
        point = list(source)
        for bit, axis in enumerate(axes):
            if mask & (1 << bit):
                point["xyz".index(axis)] *= -1.0
        points.append(tuple(point))
    return points


if __name__ == "__main__":
    if mesh_type == "hexahedra":
        # Layers normal to x and their images across x = 0.
        layers = [(0.0, 0.5, 0.7), (0.5, 1.0, 1.5)]
        mirrored_layers = layers + [(-upper, -lower, sigma_t) for lower, upper, sigma_t in layers]
        nodes = [0.125 * i for i in range(9)]
        grid = OrthogonalMeshGenerator(node_sets=[nodes] * 3).Execute()
        xs_map = []
        for block_id, (lower, upper, sigma_t) in enumerate(layers):
            volume = RPPLogicalVolume(xmin=lower, xmax=upper, infy=True, infz=True)
            grid.SetBlockIDFromLogicalVolume(volume, block_id, True)
            xs_map.append({"block_ids": [block_id], "xs": make_xs(sigma_t)})
        side = 1.0
        sample_points = [
            (0.31, 0.44, 0.19),
            (0.87, 0.52, 0.63),
            (0.06, 0.81, 0.31),
            (0.19, 0.69, 0.44),
            (0.56, 0.06, 0.81),
            (0.94, 0.81, 0.69),
        ]

        def kernel(source, point):
            return layered_kernel(source, point, mirrored_layers)

    elif mesh_type == "tetrahedra":
        sigma_t = 35.0
        partitioner = {}
        if split_source_cells:
            # The centroids of the cells at the corner have x and y between 0.0011 and 0.0040.
            if size not in (2, 4):
                raise ValueError("split_source_cells needs 2 or 4 ranks.")
            cuts = {"nx": 2, "xcuts": [0.003], "ny": 1, "nz": 1}
            if size == 4:
                cuts.update(ny=2, ycuts=[0.003])
            partitioner["partitioner"] = KBAGraphPartitioner(**cuts)
        grid = FromFileMeshGenerator(
            filename=mesh_path("uncollided_cube_coarse.msh"), **partitioner
        ).Execute()
        grid.SetUniformBlockID(0)
        xs_map = [{"block_ids": [0], "xs": make_xs(sigma_t)}]
        side = 0.032
        sample_points = [
            (0.0052, 0.0061, 0.0263),
            (0.0241, 0.0093, 0.0077),
            (0.0284, 0.0276, 0.0251),
            (0.0113, 0.0171, 0.0142),
        ]

        def kernel(source, point):
            radius = math.dist(source, point)
            return math.exp(-sigma_t * radius) / (4.0 * math.pi * radius * radius)

    else:
        raise ValueError(f"Unknown mesh_type {mesh_type}")
    grid.SetOrthogonalBoundaries()

    if case == "corner":
        sources = [(0.0, 0.0, 0.0)]
        planes = "xyz"
    elif case == "one_plane":
        sources = [(0.0, 0.3 * side, 0.45 * side)]
        planes = "xy"
    elif case == "vertex":
        if mesh_type != "hexahedra":
            raise ValueError("case vertex needs hexahedra")
        sources = [(0.0, 0.25 * side, 0.5 * side)]
        planes = "xy"
    elif case == "edge":
        sources = [(0.0, 0.0, 0.45 * side)]
        planes = "xyz"
    elif case == "two_sources":
        sources = [(0.0, 0.3 * side, 0.45 * side), (0.55 * side, 0.62 * side, 0.21 * side)]
        planes = "xy"
    else:
        raise ValueError(f"Unknown case {case}")

    def exact(point):
        return sum(kernel(image, point) for source in sources for image in images(source, planes))

    options = dict(
        mesh=grid,
        num_groups=1,
        groupsets=[{"groups_from_to": [0, 0]}],
        xs_map=xs_map,
        point_sources=[PointSource(location=list(source), strength=[1.0]) for source in sources],
        exact_near_source_leakage=True,
        scattering_order=0,
        ray_tracer=ray_tracer,
        method=method,
    )
    if "conservation_correction" in globals():
        options["conservation_correction"] = conservation_correction
    if method == "sweep":
        # The half of the box on the side of each source is ray traced; the rest is swept.
        margin = 0.01 * side
        near_source = []
        for source in sources:
            bounds = {}
            for axis, coordinate in zip("xyz", source):
                if coordinate < 0.5 * side:
                    bounds[f"{axis}min"], bounds[f"{axis}max"] = -margin, 0.5 * side
                else:
                    bounds[f"{axis}min"], bounds[f"{axis}max"] = 0.5 * side, side + margin
            near_source.append(RPPLogicalVolume(**bounds))
        options["near_source"] = near_source

    file_name = f"uncollided_source_on_reflecting_plane_{case}_{mesh_type}_{size}.h5"
    vacuum_file_name = file_name.replace(".h5", "_vacuum.h5")

    def solve(boundary_conditions, output):
        if rank == 0:
            remove_file(output)
        problem = UncollidedProblem(**options, boundary_conditions=boundary_conditions)
        solver = UncollidedSolver(problem=problem, file_name=output)
        solver.Initialize()
        solver.Execute()
        if rank == 0:
            remove_file(output)
        return problem

    if compare_vacuum:
        vacuum_problem = solve([], vacuum_file_name)

    if rank == 0:
        print("Solve with reflecting planes")
        sys.stdout.flush()
    problem = solve([{"name": PLANES[axis], "type": "reflecting"} for axis in planes], file_name)

    num_compared_values = 0
    num_different_values = 0
    if compare_vacuum:
        # The local flux vectors as bytes, eight for each value.
        flux = bytes(problem.GetPhiNewLocal())
        vacuum_flux = bytes(vacuum_problem.GetPhiNewLocal())
        if len(flux) != len(vacuum_flux):
            raise RuntimeError("The two solves have flux vectors of different sizes.")
        num_compared_values = MPIAllReduce(len(flux) // 8)
        num_different_values = MPIAllReduce(
            sum(flux[i:i + 8] != vacuum_flux[i:i + 8] for i in range(0, len(flux), 8))
        )

    scalar_flux = problem.GetScalarFluxFieldFunction()[0]
    max_error = 0.0
    flux_sum = 0.0
    for point in sample_points:
        value = point_value(scalar_flux, point, FieldFunctionInterpolationPoint, Vector3)
        max_error = max(max_error, relative_error(value, exact(point)))
        flux_sum += value

    if rank == 0:
        print(f"SourceOnReflectingPlaneMaxRelativeError={max_error:.8e}")
        print(f"SourceOnReflectingPlaneFluxSum={flux_sum:.14e}")
        if compare_vacuum:
            print(f"SourceOnReflectingPlaneComparedValues={num_compared_values}")
            print(f"SourceOnReflectingPlaneDifferentValues={num_different_values}")
        sys.stdout.flush()

    if max_error > 0.1:
        raise RuntimeError(f"Scalar-flux error is too large: {max_error}")
    if compare_vacuum and (num_compared_values == 0 or num_different_values != 0):
        raise RuntimeError(
            f"{num_different_values} of {num_compared_values} flux values differ from the "
            "solve with vacuum planes."
        )
