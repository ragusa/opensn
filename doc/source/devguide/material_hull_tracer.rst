.. _material-hull-tracer:

Material-Hull Segment Tracing
=============================

This page describes the classes in ``framework/mesh/raytrace`` that compute the
length of a straight segment in each material of a mesh without visiting the
cells along the segment: ``MaterialHull``, ``SegmentTracer`` and its two
implementations.

Motivation
----------

The optical depth between two points is

.. math::

   \tau_g = \int_0^L \sigma_{t,g}\bigl(\mathbf{x}_0 + s\,\boldsymbol{\Omega}\bigr)\,ds
          = \sum_m \sigma_{t,g,m}\,\ell_m ,

where :math:`\ell_m` is the length of the segment in material :math:`m`. Cross
sections are constant in a material, so a face between two cells of the same
material does not change the integrand. Only the faces across which the material
changes matter, together with the faces on the domain boundary. A mesh can have
millions of cells and only a few material regions, so these faces are few: their
number grows like the two-thirds power of the number of cells.

The lengths :math:`\ell_m` do not depend on the energy group. They are computed
once per segment and combined with the cross sections of every group.

The material hull
-----------------

``MaterialHull::Extract`` loops over the faces of the mesh and keeps a face when

- its two cells have different block ids, or
- it is a domain-boundary face. Its far side is the *exterior*, a material of its
  own with a zero cross section.

A kept face is turned into triangles in the same way the piecewise-linear
spatial discretization defines the surface of a cell: it is split into one
triangle per edge around the face centroid, which is the average of the face
vertices. The tracer therefore sees the faceted geometry of the solver,
including for non-planar faces, and each hull triangle is a side triangle of the
discretization, identified by its cell, face and side.

A face with three vertices is planar, so splitting it does not change the
surface. It is split all the same: the rule is then the same for every face, at
the cost of three times as many triangles on tetrahedral meshes and of about one
fifth more tracing time there.

Each triangle stores its three vertices, the material on each side, the boundary
id of a boundary face, and the cell, face and side it comes from. The normal
:math:`(\mathbf{v}_1 - \mathbf{v}_0) \times (\mathbf{v}_2 - \mathbf{v}_0)` points
from the *back* material, that of the cell that wrote the triangle, to the
*front* material. Materials are indices into the sorted list of block ids; the
index after the last block id is the exterior.

A material region does not have to be convex, and nothing in the tracer assumes
it is: a segment may enter and leave the same region several times.

Parallel meshes
~~~~~~~~~~~~~~~

``Extract`` is collective. A face between two cells is written by the cell with
the lower global id, which is a local cell on exactly one rank; the block id of
the other cell is available there because ghost cells carry their block id. The
triangles of all ranks are gathered on every rank and sorted by (global cell id,
face, side). The hull is then the same on every rank and for every
partitioning, and a rank can trace any segment without communication.

The cost is the memory of one copy of the hull per rank.

Crossing a triangle
-------------------

``IntersectSegmentTriangle`` is the edge-function test of Woop, Benthin and Wald
("Watertight Ray/Triangle Intersection", Journal of Computer Graphics Techniques
2(1), 2013), in double precision. The triangle is sheared into a frame in which
the segment runs along one axis. With :math:`\mathbf{a}`, :math:`\mathbf{b}`,
:math:`\mathbf{c}` the sheared vertices relative to the start point, the edge
functions are

.. math::

   e_{ab} = a_x b_y - a_y b_x, \qquad
   e_{bc} = b_x c_y - b_y c_x, \qquad
   e_{ca} = c_x a_y - c_y a_x ,

and the segment crosses the triangle when the three have the same sign. The
position of the crossing along the segment is
:math:`(e_{bc} a_z + e_{ca} b_z + e_{ab} c_z) / (e_{ab} + e_{bc} + e_{ca})`, a
fraction of the segment length, and the common sign tells whether the segment
runs from the back side to the front side.

A segment that passes through an edge or a vertex of the hull must be counted by
the right triangles: once when it goes through a surface, and either twice or
not at all when it only touches one. Two rules ensure this.

#. *The triangles that share an edge compute the same value for it.* The edge
   function is evaluated from the two end points taken in lexicographic order,
   in a function that is never inlined, and the sign is flipped for the triangle
   that runs along the edge the other way. The two values are then exact
   opposites whatever the floating-point contraction of the build. The same
   holds for the sheared coordinates of a shared vertex.
#. *A zero is resolved as an infinitesimal displacement.* When an edge function
   is exactly zero the segment passes through the edge or one of its vertices.
   The sign is then that of the edge function of a segment displaced by
   :math:`(\varepsilon, \varepsilon^2)` in the sheared frame,

   .. math::

      e(\mathbf{p}, \mathbf{q}) + \varepsilon\,(p_y - q_y) + \varepsilon^2\,(q_x - p_x) ,

   which is never zero for an edge of nonzero sheared length. Every triangle
   uses the same displacement, so the triangles report what one displaced
   segment would cross.

A triangle seen edge-on has a zero sheared area and is not crossed.

Lengths per material
--------------------

``ComputeMaterialLengths`` turns the crossings of a segment into the lengths
:math:`\ell_m`. The material at the end point of the segment is supplied by the
caller. With :math:`L` the length of the segment and :math:`t_k` the distance of
crossing :math:`k` from the start point, which leaves material :math:`b_k` and
enters material :math:`a_k`,

.. math::

   \ell_m = L\,[m = m_\text{end}] + \sum_k t_k \bigl([b_k = m] - [a_k = m]\bigr) .

The sum does not depend on the order of the crossings, so they do not have to be
sorted or paired. It is anchored at the end point for a reason: a crossing at the
start point has :math:`t_k \approx 0` and weighs nothing. A start point that lies
on a hull facet, which is the case of every face quadrature point on a material
interface, therefore needs no special treatment. Whether or not round-off reports
the facet, the lengths are the same.

The same crossings give a check. The quantity

.. math::

   [m = m_\text{end}] + \sum_k \bigl([b_k = m] - [a_k = m]\bigr)

is one for the material at the start point and zero for every other material.
``ComputeMaterialLengths`` returns the start material found this way, or
``SegmentTracer::INVALID_MATERIAL`` when the crossings do not satisfy the
identity, which means a crossing was missed or counted twice. A caller that
knows the material at the start point should compare.

The end point
~~~~~~~~~~~~~

A crossing at the end point has the full weight :math:`L`, so the end point must
not lie on a facet that can be reported. ``MaterialHull::FindEndPointExclusions``
handles this once per end point:

- a point inside a material touches no triangle;
- a point on the domain boundary returns the boundary triangles that contain it.
  They are passed to the tracer as excluded triangles, and the end material is
  that of the cell the point belongs to;
- a point on a facet between two materials has no defined end material, and the
  function throws ``std::invalid_argument``.

Tracers
-------

``SegmentTracer`` is the interface: it is built from a hull, ``Trace`` returns the
crossings of a batch of segments and ``TraceLengths`` their lengths per material.
A tracer is only read after construction, so several threads may trace with it at
the same time.

``BruteForceSegmentTracer`` tests every triangle against every segment. It is the
reference for tests.

``BVHSegmentTracer`` is the default. It builds a binary bounding volume hierarchy
with the binned surface-area heuristic and tests only the triangles in the boxes a
segment passes through. The boxes are padded by more than the round-off of the
box test, and both tracers use the same triangle test, so their crossings are
identical.

``Trace`` adds the work it does to a ``HullTraceCounters``: segments, segment-box
tests, segment-triangle tests and crossings. The counts depend only on the hull and
on the segments, not on the machine, which makes them the measure to compare
search strategies with.

``SegmentTracer::Create`` selects a tracer by name. A tracer built on a
third-party library would be a third implementation of the interface, enabled by
its own CMake option.

Segments that share their end point
-----------------------------------

The paths of an uncollided calculation all end at a source, and those from one
cell start close to one another. Two classes use this. Both find the same
crossings as ``Trace``, with the same triangle test, so a result does not depend
on which is used.

**Candidates of a box.** ``SegmentTracer::CollectCandidates`` returns the
triangles that a segment from a point of a box to a point, the apex, can cross,
and ``TraceCandidates`` tests segments against them. ``BVHSegmentTracer`` searches
its hierarchy once with the *shaft* of the box and the apex, their convex hull
(Haines and Wallace, 1991). The shaft is described by its bounding box and by
the planes through the apex and the edges of the box that are on its outline
seen from the apex, at most six; a box of the hierarchy that lies outside one of
them holds no candidate. The description is conservative, and the shaft is
padded by the margin of the hierarchy. The cost per segment is one triangle test
per candidate, so this pays off only when the shaft meets few triangles. It does
not when the paths run along a surface of the hull, as they do next to a
boundary plane that contains the source.

**Direction grid.** Seen from the apex, a segment that ends there is one
direction, and it can only cross the triangles that cover this direction and are
not farther from the apex than its start point. ``DirectionGridTracer`` bins the
directions on the six faces of a cube centered at the apex. On a face, with
:math:`w` the coordinate along its axis and :math:`(u, v)` the two others
divided by :math:`w`, a triangle clipped to the pyramid of the face projects
onto a convex polygon. The triangle is listed, with the bounding rectangle of
that polygon and its distance to the apex, in every bin the rectangle touches.
The list of a bin is sorted by distance. A segment computes its face and
:math:`(u, v)`, goes through the list of its bin up to the distance of its start
point, and tests the triangles whose rectangle holds :math:`(u, v)`. There is no
search.

The rectangle of a thin or slanted triangle is much larger than the triangle. A
triangle whose three vertices are in front of the apex, :math:`w > 0`, projects
onto the triangle of its projected vertices, its *outline*. When the three
project within eight face widths, the outline is kept, in single precision, and
a direction inside the rectangle is tested against its three edges before the
triangle test. On the Kobayashi meshes this brings the number of triangle tests
per path from three times the number of crossings down to the number of
crossings.

The grid is conservative for three reasons:

- the rectangles and the outlines are enlarged by :math:`10^{-5}`, in the units
  of :math:`(u, v)`, and the pyramid of a face by as much before clipping, so a
  direction on the border of two faces finds the triangle on both. The rounding
  of an outline to single precision is below :math:`10^{-6}`;
- a segment is accepted only when its end point is within :math:`10^{-12}` of
  the size of the hull from the apex, and the triangles within :math:`10^{-6}`
  of that size from the apex are tested by every segment. A crossing farther
  than that is seen from the apex within :math:`10^{-6}` of the direction of
  the start point, which the enlargement covers;
- the listed distance is a lower bound, in single precision, of the distance
  to the triangle.

A segment that does not end at the apex is handed to the tracer the grid was
built from. The memory of the grid is one entry of 28 bytes per triangle and
bin and one outline of 24 bytes per triangle and face, and it is built in one
pass over the triangles followed by a sort.

Use in the uncollided problem
-----------------------------

``UncollidedProblem`` uses the tracer when ``ray_tracer="material_hull"``. The
hull, the tracer, the total cross section of each hull material and the hull
material of each cell are built at the start of ``Execute``, once for all
sources. For each source, ``FindEndPointExclusions`` gives the triangles to
ignore and the material at the source is that of its cell.

``RaytraceLineInto`` computes the optical depth of one path either with
``ComputeCellWalkOpticalDepth`` or with ``ComputeHullOpticalDepth``. Everything
else, the quadrature points, the projection, the exact integration next to the
source and the conservation correction, is shared by the two tracers.

``hull_tracer_search`` selects how ``ComputeHullOpticalDepth`` finds the
crossings of a path that is not folded, which runs from a point of a cell to a
physical source:

- ``"segment"``: ``Trace``, for every path.
- ``"cell_shaft"``: each thread keeps the candidates of the cell and source it
  is working on, collected for the bounding box of the vertices of the cell, and
  calls ``TraceCandidates``. A path whose start point is not in that box searches
  the hierarchy.
- ``"source_directions"``: a ``DirectionGridTracer`` is built for each physical
  source before its cells are traced and released afterwards, so one grid exists
  at a time on each rank.

A path to an image source is folded into pieces that do not end at a common
point, and always uses ``Trace``.

``hull_tracer_statistics`` gathers a ``HullTraceCounters`` and the time of the
searches in each thread; the threads add them to those of the problem when they
are done, and ``ReportHullStatistics`` sums them over the ranks.

``ComputeHullOpticalDepth`` checks every path:

- The start material returned by ``ComputeMaterialLengths`` must be the material
  of the cell the point belongs to. For a point on a face of the cell, the
  material beyond that face is accepted too.
- If the check fails, the start point is displaced into the cell by about
  :math:`10^{-10}` of its distance to the cell centroid, slightly sideways so
  that it also leaves any plane of symmetry, and the path is traced again, up
  to four times. The number of retraced paths is logged. A path that still
  fails is an error.
- A path with a nonzero length in the exterior is an error: a point source must
  see every cell through the mesh.

The path to an image source is folded at the reflecting planes. Its pieces are
traced together and processed from the source end backwards: the material at the
start of a piece is the material at the end of the piece before it. A folded
path touches the reflecting boundaries without crossing them, so their triangles
are excluded for image sources.

The cell walk finds the face through which a path leaves a cell with a triangle
test that has no rule for a path through an edge or a vertex of the mesh. It
can then reject every triangle around the edge, or accept one that leads to a
cell the path only touches, and the search for the next exit fails. When that
happens, ``WalkSegmentWatertight`` walks that piece of the path again from its
start with ``IntersectSegmentTriangle``, in one frame for the whole piece. The
edge is then resolved as for one displaced path, the same for every cell: the
path leaves each convex cell through exactly one side triangle, and a cell it
only touches is crossed with a zero length. These paths occur on regular meshes
for sources at simple coordinates. Their number is logged, and
``uncollided_ray_tracer_equivalence.py`` checks that the two tracers agree to
round-off on them.

One case is left: a path that lies exactly in the plane of a face, as happens
for a source on a plane of symmetry of the mesh. The edges in that plane are
seen end-on, their edge functions are round-off instead of exact zeros, and the
crossings are then not those of one displaced path; the walk can circle around
an edge. ``WalkSegmentWatertight`` detects this, because the walk does not end
in a cell that contains the end point, and the piece is walked again from a
start point moved into its cell by about :math:`10^{-10}` of the distance to
the cell centroid, the displacement ``ComputeHullOpticalDepth`` uses. These
paths are counted in the same log line.

Before this, such a path fell back on ``RayTracer::TraceRay``, which displaces
the point by about one percent of the cell size, and the length of the path in
each cell was off by a similar amount. That fallback remains for meshes with
non-convex cells or faces of more than eight vertices, for which the walk has
no precomputed face data; the paths that take it are counted in a warning.

Ray tracing every cell on several ranks
---------------------------------------

With ``method="ray_trace"`` every cell is traced and corrected, and with the
material-hull tracer this runs on several ranks. The work is split in two.

``TraceNearSourceCell`` computes what a cell needs from ray tracing: the leakage
through its outgoing faces, its projected flux, and the exact integrals of the
cells next to the source. It reads nothing that another cell writes. Each rank
runs it for its own cells, on ``OPENSN_NUM_THREADS`` threads, against its copy
of the hull.

``CorrectCellBalance`` applies the conservation correction to one cell. Its
inputs are the corrected leakage of the cells upwind and a few numbers of the
cell itself: its raw leakages, the integral of its projected flux, its total
cross section and the part of the source emitted into it.
``CorrectRayTracedCells`` gathers these numbers from all ranks on rank 0, where
the cells of the whole mesh are corrected from the source outward: a cell is
ready when the cells upwind of it are done, and among the ready cells the lowest
global id goes first. The scale factors are then sent back to the ranks, which
scale their fluxes. The order depends only on the mesh, so the fluxes, and the
round-off of the totals, do not depend on the partitioning. Rank 0 holds
:math:`O(\text{faces} \times \text{groups})` numbers during the correction,
which is less than the flux moments it gathers to write the output file.

Two more things make the run collective:

- The orientation of a face is decided by the cell with the lower global id. On
  a partition boundary that cell can be a ghost; the rank then decides from the
  ghost's copy of the face, which has the same vertices and therefore the same
  centroid and normal as on the rank that owns it.
- Rank 0 creates and writes the output file. A failure there is rethrown on
  every rank, so that no rank waits for a rank that has stopped.

The default method keeps its own correction loop, because the corrected leakage
of a near-source cell also feeds the sweep of the bulk cells next to it; it
calls the same ``CorrectCellBalance``.

Tests
-----

``test/unit/framework/mesh/material_hull_test.cc`` and ``segment_tracer_test.cc``
run at one, two and four ranks. The reference is a description of the materials
on a grid of boxes that shares no code with the tracer: its triangles are written
down directly and the length of a segment in each material is obtained by
clipping the segment against every box. The tests cover segments through hull
vertices and edges, a non-convex region that is entered more than once, an edge
where three materials meet, start points on a facet, end points at a corner of
the domain, tracing against every cell face, the cell-by-cell walk of
``RayTracer``, and every quadrature point of a regular mesh whose source is
placed so that some paths pass exactly through mesh edges.

The regression tests in
``test/python/modules/linear_boltzmann_solvers/transport_uncollided`` run the
analytic uncollided problems with both tracers against the same reference
values, compare the two tracers and the two tracer backends directly, and run
``method="ray_trace"`` on one, two and four ranks with the same reference
values.

The two searches for segments that share their end point are compared, crossing
by crossing, with ``BruteForceSegmentTracer``, for end points inside the domain,
on a corner, a face and an edge of its boundary, and within :math:`2\times
10^{-7}` of the boundary. The start points include points on the planes of the
hull, points whose segment runs within a plane of the hull or through one of its
vertices, and points at and around the end point. The regression variants with
``hull_tracer_search`` keep the flux checks of the default search and check the
numbers of paths, segments and crossings of a run with
``hull_tracer_backend="brute_force"``.
