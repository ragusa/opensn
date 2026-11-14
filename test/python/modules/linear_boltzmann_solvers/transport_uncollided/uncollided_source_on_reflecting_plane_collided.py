#!/usr/bin/env python3
"""Uncollided and collided flux of a source on reflecting planes: an octant against a quarter.

A point source at the origin of a box of two materials with scattering is modelled twice:

- an octant, [0, 1]^3, with the source at its corner on the reflecting planes x = 0, y = 0 and
  z = 0;
- a quarter, [0, 1]^2 x [-1, 1], with the source on the edge of the reflecting planes x = 0 and
  y = 0, between two cells; its other boundaries are vacuum.

Both are given the strength of the whole source. The quarter is the octant and its mirror image
across z = 0, with the same mesh in each half, so its uncollided and collided fluxes in z > 0
are those of the octant, and every integral over the quarter is twice the one over the octant:
the uncollided production rate (1/4 against 1/8), the uncollided and collided flux integrals,
and the collided production rate, which includes the uncollided rate. The collided balance of
each model closes.

The script prints the octant production rate and balance, and the relative difference of each
integral of the quarter from twice that of the octant.
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.solver import (
        DiscreteOrdinatesProblem,
        SteadyStateSourceSolver,
        UncollidedProblem,
        UncollidedSolver,
    )
    from pyopensn.source import PointSource
    from pyopensn.xs import MultiGroupXS


def flux_integral(problem):
    interpolation = FieldFunctionInterpolationVolume()
    interpolation.SetOperationType("sum")
    interpolation.SetLogicalVolume(RPPLogicalVolume(infx=True, infy=True, infz=True))
    interpolation.AddFieldFunction(problem.GetScalarFluxFieldFunction()[0])
    interpolation.Execute()
    return interpolation.GetValue()


def solve(model):
    nodes = [i / 6.0 for i in range(7)]
    z_nodes = nodes if model == "octant" else [-1.0 + i / 6.0 for i in range(13)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes, z_nodes]).Execute()
    grid.SetOrthogonalBoundaries()
    xs_map = []
    for block_id, (lower, upper, sigma_t) in enumerate([(0.0, 0.5, 0.7), (0.5, 1.0, 1.5)]):
        volume = RPPLogicalVolume(xmin=lower, xmax=upper, infy=True, infz=True)
        grid.SetBlockIDFromLogicalVolume(volume, block_id, True)
        xs = MultiGroupXS()
        xs.CreateSimpleOneGroup(sigma_t=sigma_t, c=0.5)
        xs_map.append({"block_ids": [block_id], "xs": xs})
    planes = ["xmin", "ymin"] + (["zmin"] if model == "octant" else [])
    boundary_conditions = [{"name": name, "type": "reflecting"} for name in planes]

    file_name = f"uncollided_source_on_reflecting_plane_{model}_{size}.h5"
    if rank == 0 and os.path.exists(file_name):
        os.remove(file_name)
    uncollided = UncollidedProblem(
        mesh=grid,
        num_groups=1,
        groupsets=[{"groups_from_to": [0, 0]}],
        xs_map=xs_map,
        boundary_conditions=boundary_conditions,
        point_sources=[PointSource(location=[0.0, 0.0, 0.0], strength=[1.0])],
        exact_near_source_leakage=True,
        scattering_order=0,
        method="ray_trace",
        ray_tracer="material_hull",
    )
    uncollided_solver = UncollidedSolver(problem=uncollided, file_name=file_name)
    uncollided_solver.Initialize()
    uncollided_solver.Execute()
    uncollided_integral = flux_integral(uncollided)

    quadrature = GLCProductQuadrature3DXYZ(n_polar=4, n_azimuthal=8, scattering_order=0)
    collided = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=1,
        groupsets=[
            {
                "groups_from_to": [0, 0],
                "angular_quadrature": quadrature,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-12,
                "l_max_its": 300,
            }
        ],
        xs_map=xs_map,
        boundary_conditions=boundary_conditions,
        uncollided_flux=file_name,
    )
    collided_solver = SteadyStateSourceSolver(problem=collided)
    collided_solver.Initialize()
    collided_solver.Execute()
    balance = collided_solver.ComputeBalanceTable()
    collided_integral = flux_integral(collided)
    if rank == 0 and os.path.exists(file_name):
        os.remove(file_name)
    return {
        "uncollided": uncollided_integral,
        "collided": collided_integral,
        "production": balance["production_rate"],
        "balance": balance["balance"],
    }


if __name__ == "__main__":
    octant = solve("octant")
    quarter = solve("quarter")

    def difference(name):
        return abs(quarter[name] - 2.0 * octant[name]) / abs(2.0 * octant[name])

    if rank == 0:
        print(f"OctantProductionRate={octant['production']:.12e}")
        print(f"OctantBalance={octant['balance'] / octant['production']:.6e}")
        print(f"QuarterBalance={quarter['balance'] / quarter['production']:.6e}")
        for name in ("production", "uncollided", "collided"):
            label = name.capitalize()
            print(f"QuarterOctant{label}Difference={difference(name):.6e}")
        sys.stdout.flush()

    if abs(octant["production"] - 0.125) > 1.0e-12:
        raise RuntimeError(f"Octant production rate {octant['production']} is not 1/8.")
    for name in ("production", "uncollided", "collided"):
        if difference(name) > 1.0e-8:
            raise RuntimeError(f"The quarter {name} integral is not twice that of the octant.")
