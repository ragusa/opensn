#!/usr/bin/env python3
"""Uncollided flux by ray tracing every cell, on one or several MPI ranks.

With method="ray_trace" and ray_tracer="material_hull", every cell is traced against the
material hull of the mesh, which every rank holds, and then corrected for conservation from the
source outward. Each rank traces its own cells; the correction, which couples a cell to the cells
upwind of it, is done for all cells on one rank.

The argument mesh_type selects one of five problems, each with an analytic reference that does
not involve the mesh:

- "hexahedra": layers normal to x of an absorber, a void, a near-void and a second absorber. The
  optical depth to a point is the sum over the layers of sigma_t times the length of the path in
  the layer, which is the length of the path times the fraction of its x extent in the layer.
- "prisms": the same with three layers normal to z on extruded triangles. The P1 moments read back
  by a collided problem with no scattering are compared with phi Omega, which also checks that
  the file written from several ranks is read correctly on several ranks.
- "tetrahedra": a homogeneous cube, against the point kernel.
- "reflecting": a homogeneous box with two reflecting planes, against the point kernel of the
  source and of its three images.
- "polyhedra": a homogeneous box of general polyhedra, against the point kernel.

Each problem prints the largest relative error at sample points and the sum of the fluxes at
those points. The flux of a cell depends on the mesh only, not on how it is partitioned, so the
sum has the same reference value on 1, 2 and 4 ranks, to round-off. tests.json also checks that
the balance the solver reports, production = removal + outflow, closes to round-off in the
problems without reflecting planes, whose image sources are not corrected for conservation.

The optional arguments hull_tracer_search ("segment", "cell_shaft" or "source_directions"),
hull_tracer_backend and hull_tracer_direction_bins select how the tracer searches the hull, and
turn on the statistics of the tracer. The search does not change which triangles a path crosses,
so the flux sum and the number of crossings that the solver reports do not depend on it.
tests.json takes the number of crossings from a run with hull_tracer_backend="brute_force",
which tests every triangle against every path. These runs also write a diagnostics file, for
which the solver gathers the statistics of each cell: the number of cells with paths, the sum
of their paths and the largest number of crossings in a cell must not depend on the search, on
the number of ranks or on the number of threads either.
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
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.fieldfunc import FieldFunctionInterpolationPoint
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.math import Vector3
    from pyopensn.mesh import (
        ExtruderMeshGenerator,
        FromFileMeshGenerator,
        OrthogonalMeshGenerator,
    )
    from pyopensn.solver import (
        DiscreteOrdinatesProblem,
        SteadyStateSourceSolver,
        UncollidedProblem,
        UncollidedSolver,
    )
    from pyopensn.source import PointSource
    from pyopensn.xs import MultiGroupXS

sys.path.append(os.path.dirname(__file__))
uncollided_utils = importlib.import_module("uncollided_unstructured_utils")
extruded_mesh = uncollided_utils.extruded_mesh
mesh_path = uncollided_utils.mesh_path
point_value = uncollided_utils.point_value
relative_error = uncollided_utils.relative_error
uncollided_3d = uncollided_utils.uncollided_3d

# Optional argument: the problem to solve.
if "mesh_type" not in globals():
    mesh_type = "hexahedra"


def make_xs(sigma_t):
    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=sigma_t, c=0.0)
    return xs


def layered_flux(source, point, layers, axis):
    """Point kernel through layers (lower, upper, sigma_t) normal to an axis."""
    radius = math.dist(source, point)
    x0, x1 = sorted((source[axis], point[axis]))
    optical_depth = 0.0
    for lower, upper, sigma_t in layers:
        overlap = max(0.0, min(x1, upper) - max(x0, lower))
        optical_depth += sigma_t * radius * overlap / (x1 - x0)
    return math.exp(-optical_depth) / (4.0 * math.pi * radius * radius)


def set_layers(grid, layers, axis):
    xs_map = []
    for block_id, (lower, upper, sigma_t) in enumerate(layers):
        bounds = {"infx": True, "infy": True, "infz": True}
        name = "xyz"[axis]
        del bounds[f"inf{name}"]
        bounds[f"{name}min"] = lower
        bounds[f"{name}max"] = upper
        grid.SetBlockIDFromLogicalVolume(RPPLogicalVolume(**bounds), block_id, True)
        xs_map.append({"block_ids": [block_id], "xs": make_xs(sigma_t)})
    return xs_map


if __name__ == "__main__":
    boundary_conditions = []
    scattering_order = 0
    if mesh_type == "hexahedra":
        layers = [(-1.0, -0.25, 0.8), (-0.25, 0.25, 0.0), (0.25, 0.5, 1.0e-8), (0.5, 1.0, 1.5)]
        nodes = [-1.0 + 0.125 * i for i in range(17)]
        grid = OrthogonalMeshGenerator(node_sets=[nodes] * 3).Execute()
        xs_map = set_layers(grid, layers, 0)
        source = (-0.6127, 0.0713, -0.0389)
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

        def exact(point):
            return layered_flux(source, point, layers, 0)

    elif mesh_type == "prisms":
        layers = [(0.0, 0.75, 0.8), (0.75, 1.25, 0.0), (1.25, 2.0, 1.5)]
        grid = extruded_mesh(
            ExtruderMeshGenerator, FromFileMeshGenerator, "triangle_mesh2x2.obj", num_layers=16
        )
        xs_map = set_layers(grid, layers, 2)
        source = (0.0441723, -0.0330477, 0.31)
        scattering_order = 1
        sample_points = [
            (0.46, 0.11, 0.57),
            (-0.39, 0.43, 0.12),
            (0.22, -0.58, 0.93),
            (-0.61, -0.27, 1.11),
            (0.33, 0.52, 1.44),
            (-0.18, 0.71, 1.87),
        ]

        def exact(point):
            return layered_flux(source, point, layers, 2)

    elif mesh_type == "tetrahedra":
        sigma_t = 35.0
        grid = FromFileMeshGenerator(filename=mesh_path("uncollided_cube_coarse.msh")).Execute()
        grid.SetUniformBlockID(0)
        xs_map = [{"block_ids": [0], "xs": make_xs(sigma_t)}]
        source = (0.0113, 0.0171, 0.0142)
        sample_points = [
            (0.0052, 0.0061, 0.0263),
            (0.0241, 0.0093, 0.0077),
            (0.0284, 0.0276, 0.0251),
            (0.0036, 0.0289, 0.0048),
            (0.0197, 0.0234, 0.0029),
        ]

        def exact(point):
            return uncollided_3d(1.0, sigma_t, source, point)

    elif mesh_type == "reflecting":
        sigma_t = 0.7
        nodes = [1.0 / 6.0 * i for i in range(13)]
        grid = OrthogonalMeshGenerator(node_sets=[nodes] * 3).Execute()
        grid.SetUniformBlockID(0)
        grid.SetOrthogonalBoundaries()
        xs_map = [{"block_ids": [0], "xs": make_xs(sigma_t)}]
        boundary_conditions = [
            {"name": "xmin", "type": "reflecting"},
            {"name": "ymin", "type": "reflecting"},
        ]
        source = (0.4127, 0.6713, 0.9389)
        sample_points = [
            (0.21, 0.34, 1.19),
            (1.47, 0.22, 0.63),
            (0.96, 1.71, 1.56),
            (1.81, 1.69, 0.44),
            (0.31, 1.19, 0.26),
        ]

        def exact(point):
            return sum(
                uncollided_3d(1.0, sigma_t, (sx * source[0], sy * source[1], source[2]), point)
                for sx in (1.0, -1.0)
                for sy in (1.0, -1.0)
            )

    elif mesh_type == "polyhedra":
        sigma_t = 0.1
        grid = FromFileMeshGenerator(
            filename=mesh_path("kobayashi/kobayashi_dog_leg_pmesh")
        ).Execute()
        xs_map = [{"block_ids": [1, 2, 3], "xs": make_xs(sigma_t)}]
        source = (17.3, 21.9, 19.1)
        sample_points = [
            (7.0, 9.0, 11.0),
            (33.0, 14.0, 22.0),
            (25.0, 58.0, 31.0),
            (48.0, 77.0, 12.0),
            (12.0, 91.0, 49.0),
        ]

        def exact(point):
            return uncollided_3d(1.0, sigma_t, source, point)

    else:
        raise ValueError(f"Unknown mesh_type {mesh_type}")

    file_name = f"uncollided_ray_trace_parallel_{mesh_type}.h5"
    # Options of the material-hull ray tracer given as arguments; any of them turns on the
    # statistics of the tracer.
    tracer_options = {
        name: globals()[name]
        for name in ("hull_tracer_search", "hull_tracer_backend", "hull_tracer_direction_bins")
        if name in globals()
    }
    solver_options = {}
    if tracer_options:
        tracer_options["hull_tracer_statistics"] = True
        solver_options["diagnostics_file"] = file_name.replace(".h5", "_diagnostics.h5")
    if rank == 0:
        for name in (file_name, *solver_options.values()):
            if os.path.exists(name):
                os.remove(name)
    problem = UncollidedProblem(
        mesh=grid,
        num_groups=1,
        groupsets=[{"groups_from_to": [0, 0]}],
        xs_map=xs_map,
        point_sources=[PointSource(location=list(source), strength=[1.0])],
        boundary_conditions=boundary_conditions,
        exact_near_source_leakage=True,
        scattering_order=scattering_order,
        method="ray_trace",
        ray_tracer="material_hull",
        **tracer_options,
    )
    solver = UncollidedSolver(problem=problem, file_name=file_name, **solver_options)
    solver.Initialize()
    solver.Execute()

    scalar_flux = problem.GetScalarFluxFieldFunction()[0]
    max_error = 0.0
    flux_sum = 0.0
    for point in sample_points:
        value = point_value(scalar_flux, point, FieldFunctionInterpolationPoint, Vector3)
        max_error = max(max_error, relative_error(value, exact(point)))
        flux_sum += value

    max_p1_error = 0.0
    if scattering_order > 0:
        # Read the file back through a collided problem without scattering, whose flux is the
        # uncollided flux with all its moments.
        quadrature = GLCProductQuadrature3DXYZ(
            n_polar=4, n_azimuthal=16, scattering_order=scattering_order
        )
        collided = DiscreteOrdinatesProblem(
            mesh=grid,
            num_groups=1,
            groupsets=[{"groups_from_to": [0, 0], "angular_quadrature": quadrature}],
            xs_map=xs_map,
            uncollided_flux=file_name,
        )
        collided_solver = SteadyStateSourceSolver(problem=collided)
        collided_solver.Initialize()
        collided_solver.Execute()
        moments = collided.GetScalarFluxFieldFunction(only_scalar_flux=False)[0]
        for point in sample_points:
            dx, dy, dz = (p - q for p, q in zip(point, source))
            radius = math.sqrt(dx * dx + dy * dy + dz * dz)
            exact_scalar = exact(point)
            exact_p1 = (exact_scalar * dy / radius, exact_scalar * dz / radius,
                        exact_scalar * dx / radius)
            for moment, reference in zip(moments[1:], exact_p1):
                value = point_value(moment, point, FieldFunctionInterpolationPoint, Vector3)
                max_p1_error = max(max_p1_error, abs(value - reference) / exact_scalar)

    if rank == 0:
        print(f"RayTraceParallelMaxRelativeError={max_error:.8e}")
        print(f"RayTraceParallelFluxSum={flux_sum:.14e}")
        if scattering_order > 0:
            print(f"RayTraceParallelMaxP1ScaledError={max_p1_error:.8e}")
        sys.stdout.flush()
        for name in (file_name, *solver_options.values()):
            if os.path.exists(name):
                os.remove(name)

    if max_error > 0.1:
        raise RuntimeError(f"Scalar-flux error is too large: {max_error}")
    if max_p1_error > 0.1:
        raise RuntimeError(f"P1-moment error is too large: {max_p1_error}")
