# transport_dsa_reflecting_residual.py (2 ranks) fails when ParMETIS returns a non-straight cut

## Summary

The 2-rank variant of `test/python/modules/linear_boltzmann_solvers/transport_steady/transport_dsa_reflecting_residual.py`
fails on unmodified `main` (22d12f739) on my machine, while CI is green on the same commit:

```
check : FloatCompare key="DSA_REFLECTING_RESIDUAL_MAX_REL_ERR" expected=0 actual=0.0355638 abs_tol=2e-05 rel_tol=0 result=FAILED
```

The cause is the mesh partition, not the reflecting-boundary residual:

- The test uses the default partitioner (PETSc `parmetis`). With my dependency stack, ParMETIS splits the 10x10 mesh
  of the `xy` case into two ranks with a **stepped** cut instead of a straight one.
- A stepped cut makes the two ranks depend on each other for half of the directions. The sweep breaks this cycle
  by lagging one inter-rank angular flux.
- WGDSA has no residual for fluxes lagged between partitions (the test docstring states this for the 4-rank 3D
  variant), so 4 iterations leave an error of 3.6e-2 instead of 3e-6.

The 2-rank tolerance (2e-5) therefore holds only if the partitioner happens to return a partition without
inter-rank cycles, which depends on the METIS/ParMETIS/PETSc build.

## Environment

| Item | Value |
|---|---|
| OpenSn | `main` at 22d12f739 (clean worktree, no local changes), `Release`, `-DOPENSN_WITH_PYTHON_MODULE=ON` |
| Compiler | Intel oneAPI DPC++/C++ 2026.1.1 (`icx`/`icpx`) |
| MPI | MPICH 5.0.2 |
| PETSc | 3.26.0, 64-bit indices |
| ParMETIS / METIS | 4.0.3 / 5.1.0, `IDXTYPEWIDTH 64`, `REALTYPEWIDTH 32` |
| Python | 3.13.13 |
| OS / CPU | RHEL 9.8, x86_64 (AMD EPYC 9115) |

The partition is the same with the GCC 14.4, GCC 16.2 and Clang 21 builds of the same dependency versions on this
machine (checked with the standalone probe below), so it is not a compiler effect. I could not inspect the CI
runner's stack; I assume its partitioner build returns a straight cut.

## How to reproduce

```sh
test/run_tests --exe build/python/opensn \
  -t test/python/modules/linear_boltzmann_solvers/transport_steady/transport_dsa_reflecting_residual.py -j 8 -v 1
```

Result here: 3 tests run, 1 failed. The two 3D variants pass (1 rank: 1.154e-05, tolerance 5e-5; 4 ranks, 8
iterations: 4.815e-03, tolerance 5e-2).

Per-case output of the failing 2-rank variant (only `xy` is off):

```
CASE slab classic_richardson min rel_err 5.808257e-08
CASE slab classic_richardson max rel_err 1.164272e-07
CASE slab petsc_richardson min rel_err 6.058744e-07
CASE slab petsc_richardson max rel_err 5.433786e-07
CASE slab petsc_gmres min rel_err 2.432189e-08
CASE slab petsc_gmres max rel_err 2.277085e-08
CASE xy classic_richardson min rel_err 3.556291e-02
CASE xy classic_richardson max rel_err 3.555752e-02
CASE xy petsc_richardson min rel_err 3.556380e-02
CASE xy petsc_richardson max rel_err 3.555723e-02
CASE xy petsc_gmres min rel_err 2.053795e-02
CASE xy petsc_gmres max rel_err 2.053552e-02
CASE sphere classic_richardson min rel_err 3.567132e-06
CASE sphere classic_richardson max rel_err 2.972298e-06
CASE sphere petsc_richardson min rel_err 3.483456e-06
CASE sphere petsc_richardson max rel_err 2.695138e-06
CASE sphere petsc_gmres min rel_err 1.374225e-06
CASE sphere petsc_gmres max rel_err 7.123770e-07
DSA_REFLECTING_RESIDUAL_MAX_REL_ERR 3.556380e-02
```

## Analysis

### 1. The partition

Owner rank of each cell of the `xy` mesh (10x10, 2 ranks), default partitioner, x to the right and y up:

```
  local cells: rank0 50, rank1 50; ghosts: rank0 12, rank1 12
  y 9  1 1 1 1 1 1 0 0 0 0
  y 8  1 1 1 1 1 1 0 0 0 0
  y 7  1 1 1 1 1 1 0 0 0 0
  y 6  1 1 1 1 1 1 0 0 0 0
  y 5  1 1 1 1 1 1 0 0 0 0
  y 4  1 1 1 1 0 0 0 0 0 0
  y 3  1 1 1 1 0 0 0 0 0 0
  y 2  1 1 1 1 0 0 0 0 0 0
  y 1  1 1 1 1 0 0 0 0 0 0
  y 0  1 1 1 1 0 0 0 0 0 0
  edge cut = 12
```

The parts are balanced (50/50) but the edge cut is 12, where a straight cut gives 10.

### 2. Why the step creates a sweep cycle

Take a direction with Omega_x > 0 and Omega_y > 0:

- across the vertical pieces of the interface (x = 0.04 for y < 0.05, x = 0.06 for y > 0.05) the flow goes from
  rank 1 (left) to rank 0 (right);
- across the horizontal piece (y = 0.05, 0.04 < x < 0.06) the flow goes from the rank-0 cells below to the rank-1
  cells above.

So rank 0 needs rank 1 and rank 1 needs rank 0. The same holds for Omega_x < 0, Omega_y < 0. The other two
quadrants have a one-way dependency. A straight cut has no such cycle for any direction.

This is confirmed by the groupset option `allow_cycles=False`: with the default partition, sweep setup stops with

```
RuntimeError: AAH sweep setup failed on rank 0: AAH_SPDS: Cyclic dependencies found in the global sweep graph.
Cycles need to be allowed by the calling application.
```

and with a straight cut the same input runs normally.

### 3. The error follows the partition, not the partitioner

`xy` case only, 2 ranks, 4 iterations, max over the three inner methods of `|phi/1e4 - 1|` (min and max of phi):

| Partition | Shape | Edge cut | Max rel. error |
|---|---|---|---|
| PETSc `parmetis` (default) | stepped | 12 | 3.556e-02 |
| PETSc `ptscotch` | straight, cut in y | 10 | 2.986e-06 |
| `KBAGraphPartitioner(nx=2, xcuts=[0.05])` | straight, cut in x | 10 | 2.986e-06 |
| `KBAGraphPartitioner(ny=2, ycuts=[0.05])` | straight, cut in y | 10 | 2.986e-06 |
| `LinearGraphPartitioner` | straight, cut in y | 10 | 2.986e-06 |

Same case with the default partitioner on other rank counts: 1 rank 2.986e-06, 4 ranks 2.986e-06.

The straight-cut value matches the docstring ("1d2d, 2 ranks, 4 iterations: below 4e-6").

### 4. It is slow convergence, not a wrong answer

`xy` case, 2 ranks, relative error of the domain maximum of phi versus the iteration count:

| Iterations | stepped: classic Richardson | stepped: PETSc Richardson | stepped: GMRES | straight (KBA x): classic Richardson | straight: PETSc Richardson | straight: GMRES |
|---|---|---|---|---|---|---|
| 2 | 4.280e-02 | 4.280e-02 | 1.372e-02 | 2.240e-06 | 4.194e-06 | 4.388e-06 |
| 4 | 3.556e-02 | 3.556e-02 | 2.054e-02 | 2.864e-06 | 1.915e-06 | 4.760e-07 |
| 8 | 9.018e-04 | 9.017e-04 | 1.331e-03 | 9.622e-07 | 7.838e-07 | 2.183e-08 |
| 16 | 1.777e-03 | 1.777e-03 | 8.811e-05 | 1.260e-07 | 2.730e-07 | 7.155e-09 |
| 32 | 8.188e-05 | 8.188e-05 | 4.117e-06 | 1.628e-08 | 6.726e-09 | 4.403e-11 |

With the stepped cut the iteration still converges to the exact solution, but slowly and not monotonically.
For scale, without WGDSA the error after 4 iterations is 0.99996 with either partition.

### 5. The partition comes from ParMETIS itself

A standalone PETSc program (below) that makes the same `MatPartitioning` call as
`PETScGraphPartitioner::Partition` on the same cell graph returns exactly the stepped map of section 1. I get the
same map with four builds of the dependency stack on this machine (oneAPI 2026.1.1, GCC 14.4.0, GCC 16.2.0,
Clang 21.1.8; all ParMETIS 4.0.3, METIS 5.1.0, PETSc 3.26.0, 64-bit indices).

## Possible fixes

**A. Make the test independent of the partitioner (small).** Give the `xy` case of the 2-rank variant an explicit
straight-cut partitioner, for example `KBAGraphPartitioner(nx=2, xcuts=[0.05])`. Measured here: the `xy` case
gives 2.986e-06, so the variant's maximum becomes the sphere case, 3.567e-06, inside the existing 2e-5 tolerance.
The tolerance does not change, and the partition-lag limitation stays covered by the 4-rank 3D variant. The 1D
slab and sphere cases passed here with the default partitioner (5 + 5 cells), but they rely on ParMETIS in the
same way; I have not seen them fail.

**B. Close the gap (feature).** Give WGDSA a residual for angular fluxes lagged across partition interfaces, as
#1228 did for lagged reflecting boundaries. That removes the limitation and should also tighten the 4-rank 3D
variant, which currently needs a 5e-2 tolerance.

A is enough to make the suite deterministic across dependency stacks; B is the real improvement.

## Diagnostic scripts

All numbers above come from these three files.

<details>
<summary><code>xy_partition_diag.py</code>: the <code>xy</code> case alone, with a selectable partitioner (console mode)</summary>

```python
#!/usr/bin/env python3
"""
Diagnostic for transport_dsa_reflecting_residual.py, xy case only (console mode).

Same problem as the "xy" case of the regression test: 10x10 orthogonal mesh, 0.1 cm wide,
c = 0.9999, all faces reflecting, WGDSA on, exactly max_its iterations.

Run on 2 ranks:
  mpiexec -n 2 build/python/opensn -i xy_partition_diag.py --py part=\\"kba_x\\"

Options (--py name=value):
  part          "default" (PETSc parmetis) | "ptscotch" | "kba_x" | "kba_y" | "linear"
  max_its       iteration count (default 4)
  n             cells per side (default 10)
  wgdsa         apply WGDSA (default True)
  allow_cycles  groupset option allow_cycles (default True)

Prints, per rank, the cells it visits (PART lines: local cells first, then ghosts; render
them with show_part.py) and the error of min/max phi per inner method.
"""


def get_option(name, default):
    return globals()[name] if name in globals() else default


def field_stat(problem, op):
    ffi = FieldFunctionInterpolationVolume()
    ffi.SetOperationType(op)
    ffi.SetLogicalVolume(RPPLogicalVolume(infx=True, infy=True, infz=True))
    ffi.AddFieldFunction(problem.GetScalarFluxFieldFunction()[0])
    ffi.Execute()
    return ffi.GetValue()


if __name__ == "__main__":
    part = get_option("part", "default")
    max_its = int(get_option("max_its", 4))
    n = int(get_option("n", 10))
    width = 0.1
    c, phi_exact = 0.9999, 1.0e4

    nodes = [i * width / n for i in range(n + 1)]
    kwargs = {}
    if part == "ptscotch":
        kwargs["partitioner"] = PETScGraphPartitioner(type="ptscotch")
    elif part == "kba_x":
        kwargs["partitioner"] = KBAGraphPartitioner(nx=2, ny=1, xcuts=[width / 2])
    elif part == "kba_y":
        kwargs["partitioner"] = KBAGraphPartitioner(nx=1, ny=2, ycuts=[width / 2])
    elif part == "linear":
        kwargs["partitioner"] = LinearGraphPartitioner()
    grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes], **kwargs).Execute()
    grid.SetUniformBlockID(0)

    # Owner map: the callback visits local cells first, then ghosts.
    seen = []

    def record(centroid, old_id):
        seen.append((int(centroid.x / (width / n)), int(centroid.y / (width / n))))
        return old_id

    grid.SetBlockIDFromFunction(record)
    print(f"PART rank {rank} ncalls {len(seen)} cells " + " ".join(f"{i},{j}" for i, j in seen))

    sigma_t = 0.1 / width
    Q = phi_exact * sigma_t * (1.0 - c)
    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t, c)
    quad = GLCProductQuadrature2DXY(n_polar=4, n_azimuthal=8, scattering_order=0)
    max_err = 0.0
    for method in ("classic_richardson", "petsc_richardson", "petsc_gmres"):
        problem = DiscreteOrdinatesProblem(
            mesh=grid,
            num_groups=1,
            groupsets=[
                {
                    "groups_from_to": (0, 0),
                    "angular_quadrature": quad,
                    "inner_linear_method": method,
                    "l_abs_tol": 1.0e-14,
                    "l_max_its": max_its,
                    "apply_wgdsa": bool(get_option("wgdsa", True)),
                    "wgdsa_l_abs_tol": 1.0e-12,
                    "allow_cycles": bool(get_option("allow_cycles", True)),
                }
            ],
            xs_map=[{"block_ids": [0], "xs": xs}],
            volumetric_sources=[VolumetricSource(block_ids=[0], group_strength=[Q])],
            boundary_conditions=[
                {"name": name, "type": "reflecting"} for name in ("xmin", "xmax", "ymin", "ymax")
            ],
            options={"verbose_inner_iterations": False},
        )
        solver = SteadyStateSourceSolver(problem=problem)
        solver.Initialize()
        solver.Execute()
        for op in ("min", "max"):
            err = abs(field_stat(problem, op) / phi_exact - 1.0)
            max_err = max(max_err, err if err == err else 1.0e300)
            if rank == 0:
                print(f"CASE xy part={part} its={max_its} {method} {op} rel_err {err:.6e}")
    if rank == 0:
        print(f"MAX_REL_ERR part={part} its={max_its} {max_err:.6e}")
```

</details>

<details>
<summary><code>show_part.py</code>: renders the owner map from the output of the script above</summary>

```python
#!/usr/bin/env python3
"""
Render the cell-owner map from the PART lines that xy_partition_diag.py prints (2 ranks).

usage: python3 show_part.py [n] < output_of_xy_partition_diag.log

Each rank lists its local cells first, then its ghost cells. The number of local cells
per rank is not printed, so it is recovered as the only split for which
  - the two local sets are disjoint and cover the n x n mesh, and
  - the ghosts of each rank are exactly the cells of the other rank that share a vertex
    with one of its local cells.
"""
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
calls = {}
for line in sys.stdin:
    if "PART rank" in line:
        w = line.split()
        k = w.index("rank")
        calls[int(w[k + 1])] = [tuple(int(v) for v in c.split(",")) for c in w[k + 5:]]
if sorted(calls) != [0, 1]:
    sys.exit("expected PART lines from exactly 2 ranks")


def halo(local, other):
    return {(i + di, j + dj) for i, j in local for di in (-1, 0, 1) for dj in (-1, 0, 1)} & other


c0, c1 = calls[0], calls[1]
found = []
for L0 in range(len(c0) + 1):
    L1 = n * n - L0
    if L1 < 0 or L1 > len(c1):
        continue
    s0, s1 = set(c0[:L0]), set(c1[:L1])
    if s0 & s1 or len(s0 | s1) != n * n:
        continue
    if set(c0[L0:]) == halo(s0, s1) and set(c1[L1:]) == halo(s1, s0):
        found.append((L0, L1, s0))
if len(found) != 1:
    sys.exit(f"could not recover a unique owner map ({len(found)} candidates)")
L0, L1, s0 = found[0]
print(f"  local cells: rank0 {L0}, rank1 {L1}; ghosts: rank0 {len(c0) - L0}, rank1 {len(c1) - L1}")
cut = 0
for j in reversed(range(n)):
    print("  y%2d  " % j + " ".join("0" if (i, j) in s0 else "1" for i in range(n)))
    for i in range(n):
        cut += i + 1 < n and ((i, j) in s0) != ((i + 1, j) in s0)
        cut += j + 1 < n and ((i, j) in s0) != ((i, j + 1) in s0)
print(f"  edge cut = {cut}")
```

</details>

<details>
<summary><code>part_probe.c</code>: standalone PETSc program making the same partitioner call as OpenSn</summary>

```c
/* Replicates opensn PETScGraphPartitioner::Partition (framework/graphs/petsc_graph_partitioner.cc)
 * on the cell graph that OrthogonalMeshGenerator builds for an n x n 2D mesh:
 * cell id = ix + n*iy, neighbors listed in face order YMIN, XMAX, YMAX, XMIN (no diagonal).
 * usage: part_probe [n] [nparts] [type]
 */
#include <petsc.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  int n = argc > 1 ? atoi(argv[1]) : 10;
  int nparts = argc > 2 ? atoi(argv[2]) : 2;
  const char *type = argc > 3 ? argv[3] : "parmetis";
  PetscInt N = (PetscInt)n * n, *ia, *ja, cnt = 0;
  PetscCall(PetscMalloc((N + 1) * sizeof(PetscInt), &ia));
  PetscCall(PetscMalloc(4 * N * sizeof(PetscInt), &ja));
  for (int iy = 0; iy < n; ++iy)
    for (int ix = 0; ix < n; ++ix)
    {
      ia[ix + n * iy] = cnt;
      if (iy != 0) ja[cnt++] = ix + n * (iy - 1);     /* YMIN */
      if (ix != n - 1) ja[cnt++] = (ix + 1) + n * iy; /* XMAX */
      if (iy != n - 1) ja[cnt++] = ix + n * (iy + 1); /* YMAX */
      if (ix != 0) ja[cnt++] = (ix - 1) + n * iy;     /* XMIN */
    }
  ia[N] = cnt;
  Mat Adj;
  MatPartitioning part;
  IS is;
  PetscCall(MatCreateMPIAdj(PETSC_COMM_SELF, N, N, ia, ja, NULL, &Adj));
  PetscCall(MatPartitioningCreate(MPI_COMM_SELF, &part));
  PetscCall(MatPartitioningSetAdjacency(part, Adj));
  PetscCall(MatPartitioningSetType(part, type));
  PetscCall(MatPartitioningSetNParts(part, nparts));
  PetscCall(MatPartitioningApply(part, &is));
  const PetscInt *pid;
  PetscCall(ISGetIndices(is, &pid));
  int cut = 0;
  for (int iy = n - 1; iy >= 0; --iy)
  {
    printf("  y%2d  ", iy);
    for (int ix = 0; ix < n; ++ix)
    {
      printf("%d ", (int)pid[ix + n * iy]);
      if (ix + 1 < n && pid[ix + n * iy] != pid[ix + 1 + n * iy]) ++cut;
      if (iy + 1 < n && pid[ix + n * iy] != pid[ix + n * (iy + 1)]) ++cut;
    }
    printf("\n");
  }
  printf("  edge cut = %d\n", cut);
  PetscCall(ISRestoreIndices(is, &pid));
  PetscCall(ISDestroy(&is));
  PetscCall(MatPartitioningDestroy(&part));
  PetscCall(MatDestroy(&Adj));
  PetscCall(PetscFinalize());
  return 0;
}
```

</details>

Commands used:

```sh
# owner map and error for each partitioner (section 1 and 3)
for p in default ptscotch kba_x kba_y linear; do
  mpiexec -n 2 build/python/opensn -i xy_partition_diag.py --suppress-color --py part=\"$p\" > out_$p.log
  python3 show_part.py < out_$p.log
  grep MAX_REL_ERR out_$p.log
done

# cycle check (section 2)
mpiexec -n 2 build/python/opensn -i xy_partition_diag.py --py allow_cycles=False
mpiexec -n 2 build/python/opensn -i xy_partition_diag.py --py part=\"kba_x\" --py allow_cycles=False

# iteration sweep (section 4)
for p in default kba_x; do for its in 2 4 8 16 32; do
  mpiexec -n 2 build/python/opensn -i xy_partition_diag.py --py part=\"$p\" --py max_its=$its | grep "max rel_err"
done; done

# without WGDSA, and other rank counts
mpiexec -n 2 build/python/opensn -i xy_partition_diag.py --py wgdsa=False
mpiexec -n 1 build/python/opensn -i xy_partition_diag.py
mpiexec -n 4 build/python/opensn -i xy_partition_diag.py

# standalone probe (section 5); PETSC_DIR is the prefix that holds include/petsc.h and lib/libpetsc.so
mpicc -O2 -o part_probe part_probe.c -I$PETSC_DIR/include -L$PETSC_DIR/lib -Wl,-rpath,$PETSC_DIR/lib -lpetsc
./part_probe 10 2 parmetis
```
