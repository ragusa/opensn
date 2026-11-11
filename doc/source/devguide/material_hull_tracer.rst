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
spatial discretization defines the surface of a cell: a face with more than
three vertices is split into one triangle per edge around the face centroid,
which is the average of the face vertices. A face with three vertices is planar,
so it is kept as one triangle. The tracer therefore sees the faceted geometry of
the solver, including for non-planar faces.

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

``SegmentTracer::Create`` selects a tracer by name. A tracer built on a
third-party library would be a third implementation of the interface, enabled by
its own CMake option.

Tests
-----

``test/unit/framework/mesh/material_hull_test.cc`` and ``segment_tracer_test.cc``
run at one, two and four ranks. The reference is a description of the materials
on a grid of boxes that shares no code with the tracer: its triangles are written
down directly and the length of a segment in each material is obtained by
clipping the segment against every box. The tests cover segments through hull
vertices and edges, a non-convex region that is entered more than once, an edge
where three materials meet, start points on a facet, end points at a corner of
the domain, tracing against every cell face, and the cell-by-cell walk of
``RayTracer``.
