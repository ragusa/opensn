# Searches of the material hull on the Kobayashi point-source problem

The material-hull ray tracer of `UncollidedProblem` finds, for every path from a quadrature
point to a point source, the triangles of the hull that the path crosses. This study compares
the ways of finding them that `hull_tracer_search` selects:

- `segment`: every path searches a bounding volume hierarchy of the hull. `segment_leafN` sets
  the largest number of triangles in a leaf of the hierarchy (`hull_tracer_leaf_size`, 4 by
  default).
- `cell_shaft`: the hierarchy is searched once for all the paths from a cell to the source, and
  each path tests every triangle found.
- `source_directions_N`: the triangles are binned by direction seen from the source, with N bins
  along each side of the six faces of a cube (`hull_tracer_direction_bins`). A path tests the
  triangles of its direction that are not farther from the source than its start point.

The problem is that of `../uncollided_hull_tracer`: a point source at the corner of the
Kobayashi dog-leg geometry, every cell ray traced against the hull (strategy C), on the fine
mesh (13,909 tetrahedra, 13,383 hull triangles) and the xfine mesh (111,272 tetrahedra, 53,532
hull triangles).

## What is measured

- Bound tests per path: tests of a path against a box of the hierarchy, or against the
  rectangle of a triangle in the direction grid.
- Triangle tests per path: full segment-triangle tests.
- Candidate triangles: entries of the candidate sets (one set per cell for `cell_shaft`, one
  grid for `source_directions`).
- Search: time spent finding the crossings, building the candidate sets included, from a run
  with `hull_tracer_statistics=True`. It includes two clock readings per path.
- Building: part of the search time spent building the candidate sets.
- Ray-tracing phase: wall time of the ray-tracing phase of the solver, from a run without the
  statistics. Besides the search it holds the finite element data of the cells, the
  exponentials and the projections, which the search does not change.
- Speed-ups are relative to `segment`.

The counts do not depend on the machine. The flux moments of every configuration are identical,
bit for bit, to those of `segment`; `run_search_study.py` checks this.

## Results

### fine mesh: 2,361,546 paths, 2.19 crossings per path

| Configuration | Bound tests per path | Triangle tests per path | Candidate triangles | Search (s) | Building (s) | Ray-tracing phase (s) | Search speed-up | Phase speed-up |
|---|---|---|---|---|---|---|---|---|
| `segment` | 141.1 | 26.8 | 0 | 2.20 | 0.000 | 2.59 | 1.00 | 1.00 |
| `segment_leaf1` | 169.7 | 18.3 | 0 | 2.20 | 0.000 | 2.58 | 1.00 | 1.00 |
| `segment_leaf2` | 156.9 | 19.6 | 0 | 2.09 | 0.000 | 2.50 | 1.05 | 1.04 |
| `segment_leaf8` | 125.4 | 45.9 | 0 | 2.61 | 0.000 | 3.04 | 0.84 | 0.85 |
| `segment_leaf16` | 110.1 | 88.9 | 0 | 3.85 | 0.000 | 4.25 | 0.57 | 0.61 |
| `cell_shaft` | 0.0 | 125.6 | 1,769,550 | 4.08 | 0.139 | 4.49 | 0.54 | 0.58 |
| `cell_shaft_leaf1` | 0.0 | 125.6 | 1,769,550 | 4.15 | 0.161 | 4.55 | 0.53 | 0.57 |
| `source_directions_16` | 170.5 | 2.3 | 35,447 | 0.75 | 0.007 | 1.18 | 2.93 | 2.20 |
| `source_directions_32` | 68.4 | 2.3 | 53,720 | 0.56 | 0.009 | 0.98 | 3.93 | 2.64 |
| `source_directions_64` | 32.6 | 2.3 | 99,199 | 0.49 | 0.012 | 0.92 | 4.51 | 2.83 |
| `source_directions_128` | 18.7 | 2.3 | 225,310 | 0.47 | 0.021 | 0.89 | 4.70 | 2.93 |
| `source_directions_256` | 12.6 | 2.3 | 618,851 | 0.48 | 0.048 | 0.90 | 4.55 | 2.87 |
| `source_directions_512` | 9.8 | 2.3 | 1,971,829 | 0.60 | 0.137 | 1.02 | 3.64 | 2.53 |
| `source_directions_1024` | 8.5 | 2.3 | 6,939,608 | 1.05 | 0.487 | 1.45 | 2.10 | 1.78 |

### xfine mesh: 18,705,710 paths, 2.20 crossings per path

| Configuration | Bound tests per path | Triangle tests per path | Candidate triangles | Search (s) | Building (s) | Ray-tracing phase (s) | Search speed-up | Phase speed-up |
|---|---|---|---|---|---|---|---|---|
| `segment` | 162.9 | 29.9 | 0 | 19.20 | 0.000 | 22.20 | 1.00 | 1.00 |
| `segment_leaf1` | 194.9 | 18.3 | 0 | 18.48 | 0.000 | 21.66 | 1.04 | 1.02 |
| `segment_leaf2` | 178.1 | 19.8 | 0 | 17.74 | 0.000 | 20.90 | 1.08 | 1.06 |
| `segment_leaf8` | 141.3 | 54.2 | 0 | 23.59 | 0.000 | 26.91 | 0.81 | 0.82 |
| `segment_leaf16` | 127.9 | 86.3 | 0 | 30.79 | 0.000 | 37.22 | 0.62 | 0.60 |
| `cell_shaft` | 0.0 | 133.4 | 14,924,591 | 37.20 | 1.300 | 40.73 | 0.52 | 0.54 |
| `cell_shaft_leaf1` | 0.0 | 133.4 | 14,924,591 | 35.40 | 1.327 | 38.61 | 0.54 | 0.57 |
| `source_directions_16` | 540.5 | 2.3 | 106,134 | 11.17 | 0.024 | 14.71 | 1.72 | 1.51 |
| `source_directions_32` | 187.5 | 2.3 | 138,331 | 6.08 | 0.026 | 9.48 | 3.16 | 2.34 |
| `source_directions_64` | 73.2 | 2.3 | 211,733 | 4.46 | 0.032 | 7.63 | 4.30 | 2.91 |
| `source_directions_128` | 33.8 | 2.3 | 393,779 | 3.77 | 0.046 | 7.00 | 5.10 | 3.17 |
| `source_directions_256` | 18.9 | 2.3 | 899,416 | 3.57 | 0.082 | 6.81 | 5.38 | 3.26 |
| `source_directions_512` | 12.5 | 2.3 | 2,479,102 | 3.66 | 0.198 | 6.94 | 5.25 | 3.20 |
| `source_directions_1024` | 9.7 | 2.3 | 7,908,167 | 4.28 | 0.569 | 7.52 | 4.49 | 2.95 |

## Observations

- With `segment`, a path costs 141 to 163 box tests and 27 to 30 triangle tests for 2.2
  crossings, and the search is more than 80% of the ray-tracing phase. All paths end at the source, which
  lies on three boundary planes, and many run along these planes or along the walls of the
  duct, so a path passes through many boxes that hold no crossing.
- The size of the leaves moves work between box tests and triangle tests without reducing it:
  leaves of 1 to 4 triangles are within 8% of one another, larger leaves are slower.
- `cell_shaft` is about half as fast as `segment`. The shaft from a cell to the source holds
  about 130 triangles on average, for the same reason, and every path of the cell tests them
  all.
- `source_directions` brings the triangle tests down to 2.3 per path, the number of crossings.
  The bound tests fall as the bins get finer, and the time is flat between 128 and 512 bins.
  At 256 bins, the default, the search is 4.6 times faster on the fine mesh and 5.4 times
  faster on the xfine mesh, and the ray-tracing phase 2.9 and 3.3 times faster. The grid takes
  0.05 to 0.08 s to build, and 17 to 25 MB.
- What is left of the ray-tracing phase with `source_directions_256` is no longer mainly the
  search: 3.6 s of search, clock readings included, in a phase of 6.8 s on the xfine mesh.

## Conditions

- Code: branch `uncollided-hull-accel`, the commit that adds this study's scripts.
- Build: Intel oneAPI 2026.1.1 `icpx`, Release, MPICH 5.0.2.
- Machine: 2 x AMD EPYC 9115 (32 cores), load average about 2 during the runs.
- One rank, one thread. Each case ran twice with the statistics and twice without; the faster
  run of each kind is kept.

To reproduce, from this directory with OpenSn built in `build/` and a Python that has h5py:

    python3 run_search_study.py --meshes fine xfine --repeat 2
    python3 summarize.py data/search_study_np1_nt1.csv
