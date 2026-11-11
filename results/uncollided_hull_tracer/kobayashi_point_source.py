#!/usr/bin/env python3
"""Uncollided flux of a point source in the Kobayashi dog-leg geometry (OpenSn console input).

This is the modification of Kobayashi problem 3 used by Woodsford, Ragusa and Morel, "Sweep-based
uncollided-flux treatment on unstructured grids", Progress in Nuclear Energy 200 (2026) 106524:
the 10 cm cube source is replaced by a point source of strength 1000 at the origin. The model is
one eighth of the benchmark, with the source at a corner of the mesh. The three symmetry planes
through the source are vacuum boundaries, which is exact for the uncollided flux: no path from
the source crosses them.

Total cross sections: 0.1 /cm in the source region and in the shield, 1e-4 /cm in the duct.

Arguments (set with --py name=value):

    mesh_name   "coarse" (4,826 tetrahedra), "fine" (13,909), "xfine" (111,272) or "polyhedra"
                (9,024 polyhedra)
    strategy    "A": bulk sweep, near-source region traced with the cell walk
                "B": bulk sweep, near-source region traced with the material hull
                "C": every cell traced with the material hull
    prefix      prefix of the output files, relative to the working directory
    mesh_dir    directory of the Kobayashi meshes
    lineout     False to skip the flux along the centerline of the duct (default True)

and, passed to the problem when given, the options of the material-hull ray tracer:
hull_tracer_backend, hull_tracer_leaf_size, hull_tracer_search, hull_tracer_direction_bins and
hull_tracer_statistics.

Outputs: <prefix>.h5 (uncollided flux moments), <prefix>_diagnostics.h5 (scale factors and
balance residuals of the cells) and <prefix>_lineout.csv (flux along the centerline of the duct).
"""

import csv
import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    from pyopensn.fieldfunc import FieldFunctionInterpolationPoint
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.math import Vector3
    from pyopensn.mesh import FromFileMeshGenerator
    from pyopensn.solver import UncollidedProblem, UncollidedSolver
    from pyopensn.source import PointSource
    from pyopensn.xs import MultiGroupXS

if "mesh_name" not in globals():
    mesh_name = "coarse"
if "strategy" not in globals():
    strategy = "C"
if "prefix" not in globals():
    prefix = f"kobayashi_{mesh_name}_{strategy}"
if "mesh_dir" not in globals():
    mesh_dir = "../../test/assets/mesh/kobayashi"
if "lineout" not in globals():
    lineout = True
TRACER_OPTIONS = ("hull_tracer_backend", "hull_tracer_leaf_size", "hull_tracer_search",
                  "hull_tracer_direction_bins", "hull_tracer_statistics")

MESH_FILES = {
    "coarse": "kobayashi_dog_leg.msh",
    "fine": "kobayashi_dog_leg_fine.msh",
    "xfine": "kobayashi_dog_leg_xfine.msh",
    "polyhedra": "kobayashi_dog_leg_pmesh",
}
# Block ids of the benchmark regions and their total cross sections (1/cm).
SIGMA_T = {1: 0.1, 2: 1.0e-4, 3: 0.1}
SOURCE_STRENGTH = 1000.0
# Centerline of the duct: straight legs between these points, 150 cm in all.
CENTERLINE = [(5.0, 5.0, 5.0), (5.0, 55.0, 5.0), (35.0, 55.0, 5.0), (35.0, 55.0, 35.0),
              (35.0, 95.0, 35.0)]
NUM_LINEOUT_POINTS = 301


def centerline_points():
    """Points equally spaced in path length along the centerline, with their path length."""
    legs = []
    total = 0.0
    for a, b in zip(CENTERLINE[:-1], CENTERLINE[1:]):
        length = sum((q - p) ** 2 for p, q in zip(a, b)) ** 0.5
        legs.append((total, length, a, b))
        total += length
    points = []
    for n in range(NUM_LINEOUT_POINTS):
        s = total * n / (NUM_LINEOUT_POINTS - 1)
        start, length, a, b = next(leg for leg in reversed(legs) if s >= leg[0])
        t = min(1.0, (s - start) / length)
        points.append((s, tuple(p + t * (q - p) for p, q in zip(a, b))))
    return points


if __name__ == "__main__":
    grid = FromFileMeshGenerator(filename=os.path.join(mesh_dir, MESH_FILES[mesh_name])).Execute()

    xs_map = []
    for block_id, sigma_t in SIGMA_T.items():
        xs = MultiGroupXS()
        xs.CreateSimpleOneGroup(sigma_t=sigma_t, c=0.0)
        xs_map.append({"block_ids": [block_id], "xs": xs})

    options = {
        "A": {"method": "sweep", "ray_tracer": "cell_walk"},
        "B": {"method": "sweep", "ray_tracer": "material_hull"},
        "C": {"method": "ray_trace", "ray_tracer": "material_hull"},
    }[strategy]
    if options["method"] == "sweep":
        # The near-source region is the source region of the benchmark, as in the paper.
        options["near_source"] = [
            RPPLogicalVolume(xmin=-1.0, xmax=10.0, ymin=-1.0, ymax=10.0, zmin=-1.0, zmax=10.0)
        ]

    options.update({name: globals()[name] for name in TRACER_OPTIONS if name in globals()})

    problem = UncollidedProblem(
        mesh=grid,
        num_groups=1,
        groupsets=[{"groups_from_to": [0, 0]}],
        xs_map=xs_map,
        point_sources=[PointSource(location=[0.0, 0.0, 0.0], strength=[SOURCE_STRENGTH])],
        exact_near_source_leakage=True,
        scattering_order=0,
        **options,
    )
    solver = UncollidedSolver(
        problem=problem,
        file_name=f"{prefix}.h5",
        diagnostics_file=f"{prefix}_diagnostics.h5",
        progress_interval=0,
    )
    solver.Initialize()
    solver.Execute()

    rows = []
    if lineout:
        scalar_flux = problem.GetScalarFluxFieldFunction()[0]
        for s, point in centerline_points():
            interpolation = FieldFunctionInterpolationPoint()
            interpolation.SetPointOfInterest(Vector3(*point))
            interpolation.AddFieldFunction(scalar_flux)
            interpolation.Execute()
            rows.append((s, *point, interpolation.GetPointValue()))
    if rank == 0:
        if lineout:
            with open(f"{prefix}_lineout.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["s", "x", "y", "z", "phi"])
                writer.writerows(rows)
        print(f"KobayashiPointSourceDone={mesh_name},{strategy},{size}")
        sys.stdout.flush()
