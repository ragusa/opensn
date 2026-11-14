#!/usr/bin/env python3
"""Verify that reflecting uncollided setups reject face- and vertex-aligned sources inside the
domain, accept a source on a reflecting boundary, and reject a source within 1e-6 of a
reflecting boundary but not on it, where the mesh would place it on the boundary face while its
image would still be traced."""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.solver import UncollidedProblem
    from pyopensn.source import PointSource
    from pyopensn.xs import MultiGroupXS


if __name__ == "__main__":
    if size != 1:
        sys.exit(f"Expected one process, got {size}.")

    grid = OrthogonalMeshGenerator(node_sets=[[-1.0, 0.0, 1.0]] * 3).Execute()
    grid.SetUniformBlockID(0)
    grid.SetOrthogonalBoundaries()

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
        UncollidedProblem(
            mesh=grid,
            num_groups=1,
            groupsets=[{"groups_from_to": [0, 0]}],
            xs_map=[{"block_ids": [0], "xs": xs}],
            boundary_conditions=[{"name": "xmin", "type": "reflecting"}],
            point_sources=[PointSource(location=[0.0, 0.25, 0.3], strength=[1.0])],
            near_source=[whole_domain],
        )
    except Exception as error:
        rejected = "not currently supported for uncollided generation" in str(error)

    # A source on the reflecting plane itself has one cell and no image across the plane.
    on_plane_accepted = True
    try:
        UncollidedProblem(
            mesh=grid,
            num_groups=1,
            groupsets=[{"groups_from_to": [0, 0]}],
            xs_map=[{"block_ids": [0], "xs": xs}],
            boundary_conditions=[{"name": "xmin", "type": "reflecting"}],
            point_sources=[PointSource(location=[-1.0, 0.25, 0.3], strength=[1.0])],
            near_source=[whole_domain],
        )
    except Exception:
        on_plane_accepted = False

    # A source 5e-7 from the reflecting plane is in the band where it is neither on the plane nor
    # clear of the boundary face.
    near_plane_rejected = False
    try:
        UncollidedProblem(
            mesh=grid,
            num_groups=1,
            groupsets=[{"groups_from_to": [0, 0]}],
            xs_map=[{"block_ids": [0], "xs": xs}],
            boundary_conditions=[{"name": "xmin", "type": "reflecting"}],
            point_sources=[PointSource(location=[-1.0 + 5.0e-7, 0.25, 0.3], strength=[1.0])],
            near_source=[whole_domain],
        )
    except Exception as error:
        near_plane_rejected = "from the reflecting boundary" in str(error)

    if rank == 0:
        print(f"UncollidedReflectingFaceOrVertexSourceRejected={int(rejected)}")
        print(f"UncollidedReflectingPlaneSourceAccepted={int(on_plane_accepted)}")
        print(f"UncollidedReflectingNearPlaneSourceRejected={int(near_plane_rejected)}")
        sys.stdout.flush()

    if not rejected:
        raise RuntimeError(
            "Reflecting uncollided generation accepted a point source on a face, edge, or vertex."
        )
    if not on_plane_accepted:
        raise RuntimeError(
            "Reflecting uncollided generation rejected a point source on a reflecting boundary."
        )
    if not near_plane_rejected:
        raise RuntimeError(
            "Reflecting uncollided generation accepted a point source within 1e-6 of a reflecting "
            "boundary."
        )
