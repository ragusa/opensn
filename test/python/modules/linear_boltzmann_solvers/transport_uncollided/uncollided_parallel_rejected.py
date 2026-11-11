#!/usr/bin/env python3
"""Verify that uncollided generation rejects multiple MPI ranks."""

import importlib
import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.mesh import ExtruderMeshGenerator, FromFileMeshGenerator
    from pyopensn.solver import UncollidedProblem, UncollidedSolver
    from pyopensn.source import PointSource
    from pyopensn.xs import MultiGroupXS

sys.path.append(os.path.dirname(__file__))
extruded_mesh = importlib.import_module("uncollided_unstructured_utils").extruded_mesh


if __name__ == "__main__":
    if size != 2:
        sys.exit(f"Expected two processes, got {size}.")

    grid = extruded_mesh(
        ExtruderMeshGenerator, FromFileMeshGenerator, "triangle_mesh2x2.obj", num_layers=4
    )
    grid.SetUniformBlockID(0)
    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t=1.0, c=0.0)
    whole_domain = RPPLogicalVolume(
        xmin=-1.01,
        xmax=1.01,
        ymin=-1.01,
        ymax=1.01,
        infz=True,
    )
    rejected = False
    try:
        problem = UncollidedProblem(
            mesh=grid,
            num_groups=1,
            groupsets=[{"groups_from_to": [0, 0]}],
            xs_map=[{"block_ids": [0], "xs": xs}],
            point_sources=[PointSource(location=[0.0, 0.0, 1.05], strength=[1.0])],
            near_source=[whole_domain],
        )
        solver = UncollidedSolver(
            problem=problem,
            file_name="uncollided_parallel_rejected.h5",
        )
        solver.Initialize()
    except ValueError as error:
        rejected = "exactly one MPI rank" in str(error)

    # Several ranks are supported only when every cell is ray traced against the material hull,
    # which needs no cell of another rank. The two other combinations of these options walk
    # through cells or sweep them, and are rejected.
    other_combinations_rejected = True
    for arguments in (
        {"method": "ray_trace", "ray_tracer": "cell_walk"},
        {"method": "sweep", "ray_tracer": "material_hull", "near_source": [whole_domain]},
    ):
        try:
            problem = UncollidedProblem(
                mesh=grid,
                num_groups=1,
                groupsets=[{"groups_from_to": [0, 0]}],
                xs_map=[{"block_ids": [0], "xs": xs}],
                point_sources=[PointSource(location=[0.0, 0.0, 1.05], strength=[1.0])],
                **arguments,
            )
            other_combinations_rejected = False
        except ValueError as error:
            if "exactly one MPI rank" not in str(error):
                other_combinations_rejected = False

    if rank == 0:
        print(f"UncollidedParallelGenerationRejected={int(rejected)}")
        print(f"UncollidedParallelOtherCombinationsRejected={int(other_combinations_rejected)}")
        sys.stdout.flush()

    if not rejected:
        raise RuntimeError("Uncollided generation did not reject multiple MPI ranks")
    if not other_combinations_rejected:
        raise RuntimeError("Uncollided generation accepted an unsupported option on several ranks")
