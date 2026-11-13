// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/logical_volume/logical_volume.h"
#include "framework/mesh/raytrace/material_hull.h"
#include "framework/mesh/raytrace/raytracer.h"
#include "framework/mesh/raytrace/direction_grid_tracer.h"
#include "framework/mesh/raytrace/segment_tracer.h"
#include "framework/mesh/mesh/cell.h"
#include "framework/data_types/dense_matrix.h"
#include "framework/data_types/vector.h"
#include "framework/data_types/vector3.h"
#include "hdf5.h"
#include <array>
#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

namespace opensn
{

class UncollidedSolver;

struct UncollidedMatrices
{
  DenseMatrix<double> intV_shapeJ_omega_gradshapeI;
  std::vector<DenseMatrix<double>> intS_omega_n_shapeI_shapeJ;
};

class UncollidedProblem : public LBSProblem
{
public:
  ~UncollidedProblem() override;

  void ClearBoundaries() override {}

  /// Generation steps run by UncollidedSolver: locate the point sources, add their images in the
  /// reflecting planes, and compute and write the uncollided flux moments.
  void BuildSourcePoints();
  void AddReflectedSourcePoints();
  /// @p diagnostics_file_name, when not empty, names a second HDF5 file that receives, for each
  /// physical source s, the global ids of the ray-traced cells ("cell ids s"), their scale
  /// factors and relative balance residuals after the conservation correction ("alpha s" and
  /// "balance residual s", indexed [cell * num_groups + group] in the order of the cell ids),
  /// and the same residuals for the swept cells ("bulk cell ids s", "bulk balance residual s").
  void Execute(const std::string& file_name,
               unsigned int progress_interval,
               const std::string& diagnostics_file_name = "");

protected:
  /// Factory-only constructor.
  explicit UncollidedProblem(const InputParameters& params);

  /// Uncollided-generation rules.
  void CheckConfigurationErrors(std::vector<std::string>& errors) const override;

  /// True when a point source lies on a boundary face of one of its cells. Collective.
  bool IsOnDomainBoundary(const PointSource& point_source) const;

  struct ReflectionPlane
  {
    std::uint64_t boundary_id = 0;
    Vector3 normal;
    double offset = 0.0;
  };

  struct SourcePoint
  {
    struct Subscriber
    {
      std::uint32_t cell_local_id = 0;
      double volume_weight = 1.0;
    };

    Vector3 location;
    std::vector<double> strength;
    std::vector<Subscriber> subscribers;
    std::shared_ptr<LogicalVolume> near_source_logvol;
    std::vector<ReflectionPlane> reflection_planes;
    /// Material-hull ray tracer only: the hull material at the source, and the sorted hull
    /// triangles that a path ending at the source ignores. For an image source these are the
    /// material of its physical source, where its folded path ends, and the triangles at that
    /// source and on the reflecting boundaries.
    std::uint32_t hull_material = 0;
    std::vector<std::uint32_t> hull_excluded_triangles;
  };

  /// How the optical depth between a point and a source is computed.
  enum class RayTracerType
  {
    /// Walk from cell to cell along the path.
    CELL_WALK,
    /// Intersect the path with the material hull of the mesh.
    MATERIAL_HULL
  };

  /// How the material-hull ray tracer searches the hull for the triangles a path can cross.
  enum class HullSearch
  {
    /// Every path searches the hierarchy of the tracer.
    SEGMENT,
    /// The hierarchy is searched once for the paths from a cell to a source.
    CELL_SHAFT,
    /// The triangles are binned by direction seen from the source, and a path tests those of
    /// its direction.
    SOURCE_DIRECTIONS
  };

  /// The search that a value of hull_tracer_search stands for. "auto" depends on the method:
  /// SOURCE_DIRECTIONS when every cell is ray traced ("ray_trace"), SEGMENT otherwise.
  static HullSearch ResolveHullSearch(const std::string& search, const std::string& method);

  enum class CellRegion
  {
    NEAR_SOURCE,
    BULK
  };

  struct Moment
  {
    unsigned int ell;
    int m;
  };

  void PrintSimHeader() override;

  void InitializeSpatialDiscretization() override;

  static Vector3 ComputeOmega(const Vector3& point0, const Vector3& point1)
  {
    double norm = (point1 - point0).Norm();
    return norm == 0. ? Vector3(0., 0., 0.) : (point1 - point0).Normalized();
  }

  void BuildSweepOrdering(const SourcePoint& source_point);

  /// Returns the planes of the reflecting boundaries and appends an error for each invalid
  /// boundary condition.
  std::vector<ReflectionPlane> ComputeReflectionPlanes(std::vector<std::string>& errors) const;

  /// Stores the reflection planes once the configuration has been validated.
  void BuildRuntimeData() override;

  /// Ray-traced data of a near-source cell, before its conservation correction.
  struct RawNearSourceCell
  {
    /// Leakage through each face, indexed [face][group]. Tracing fills the outgoing faces; the
    /// conservation pass scales them and copies the leakage of the upwind cell to the incoming
    /// faces.
    std::vector<std::vector<double>> leakage;
    /// Flux at the cell nodes, indexed [group](node).
    std::vector<Vector<double>> phi;
    /// Upwind face terms of the bulk cells next to this cell, by bulk degree of freedom and then
    /// by group.
    std::unordered_map<size_t, std::vector<double>> bulk_rhs;
    /// For each face towards the bulk region, the share of a uniform face flux that goes to each
    /// bulk degree of freedom.
    std::vector<std::vector<std::pair<size_t, double>>> bulk_face_rhs_weights;
    bool is_source_cell = false;
    bool is_exact_neighbor = false;
    /// True when the cell was integrated exactly over direction.
    bool integrated_exactly = false;
  };

  /// Totals of the conservation correction over the ray-traced cells of a source.
  struct ConservationSummary
  {
    /// Cells, and cell-group pairs, corrected by more than 1%.
    size_t mismatched_cell_count = 0;
    size_t mismatched_cell_group_count = 0;
    /// Uncorrected removal of the pairs corrected by more than 1%, and that removal weighted by
    /// the relative correction.
    double original_outgoing_sum = 0.0;
    double outgoing_change_sum = 0.0;
    double max_relative_outgoing_change = 0.0;

    /// Number of cells corrected.
    size_t num_cells = 0;
    /// Smallest and largest scale factor over the cells and groups.
    double min_alpha = std::numeric_limits<double>::max();
    double max_alpha = 0.0;
    /// Upper bounds of the bands of |alpha - 1| in which the cell-group pairs are counted. A
    /// factor far from one marks a cell whose flux the quadratures do not resolve, such as a cell
    /// cut by the shadow of a material edge.
    static constexpr std::array<double, 4> alpha_bands = {1.0e-3, 1.0e-2, 1.0e-1, 0.5};
    /// Number of cell-group pairs in each band; the last entry counts the pairs above all bands.
    std::array<size_t, 5> alpha_histogram = {0, 0, 0, 0, 0};
    /// Number of cell-group pairs whose outgoing faces share the source equally.
    size_t leakage_fallback_count = 0;
    /// Largest |inflow + source - outflow - removal| after the correction, relative to
    /// inflow + source.
    double max_balance_residual = 0.0;
    /// The pair with the largest |alpha - 1|.
    double largest_change = 0.0;
    std::uint64_t largest_change_cell = 0;
    size_t largest_change_group = 0;
    double largest_change_alpha = 1.0;
    /// When set, the global id of every cell and the scale factor and relative balance residual
    /// of every cell and group are kept, in the order in which the cells are corrected.
    bool keep_cells = false;
    std::vector<std::uint64_t> cell_ids;
    std::vector<double> cell_alphas;
    std::vector<double> cell_residuals;
  };

  /// Per-cell results of one physical source for the diagnostics file.
  struct SourceDiagnostics
  {
    std::vector<std::uint64_t> cell_ids;
    std::vector<double> alphas;
    std::vector<double> residuals;
    std::vector<std::uint64_t> bulk_cell_ids;
    std::vector<double> bulk_residuals;
  };

  /// Wall time of rank 0 in each phase of Execute, in seconds.
  struct PhaseTimings
  {
    double material_hull = 0.0;
    double sweep_ordering = 0.0;
    double ray_tracing = 0.0;
    double conservation = 0.0;
    double bulk_sweep = 0.0;
    double image_sources = 0.0;
    double moments = 0.0;
    double output = 0.0;
  };

  /// Work and time of the material-hull ray tracer, for monitoring.
  struct HullStatistics
  {
    /// Paths between a point and a source whose optical depth was computed.
    std::uint64_t paths = 0;
    HullTraceCounters counters;
    /// Time spent searching the hull for the crossings of the paths, in nanoseconds and summed
    /// over the threads. Measured only with hull_tracer_statistics.
    std::uint64_t search_time = 0;
    /// Part of the search time spent collecting candidate sets.
    std::uint64_t collection_time = 0;

    HullStatistics& operator+=(const HullStatistics& other)
    {
      paths += other.paths;
      counters += other.counters;
      search_time += other.search_time;
      collection_time += other.collection_time;
      return *this;
    }

    /// The statistics by name, in the order they are reported.
    std::array<std::pair<const char*, std::uint64_t*>, 10> Entries()
    {
      return {{{"paths", &paths},
               {"segments", &counters.segments},
               {"box tests", &counters.box_tests},
               {"triangle tests", &counters.triangle_tests},
               {"crossings", &counters.crossings},
               {"candidate sets", &counters.candidate_sets},
               {"candidate box tests", &counters.candidate_box_tests},
               {"candidate triangles", &counters.candidate_triangles},
               {"search time", &search_time},
               {"collection time", &collection_time}}};
    }
  };

  /// Candidate triangles of the paths from one cell to one source, kept by each thread.
  struct HullCandidateCache
  {
    /// The hull, cell and source the candidates are for.
    std::uint64_t hull_generation = 0;
    std::uint64_t cell_local_id = 0;
    Vector3 apex;
    /// Box that holds the cell.
    Vector3 lower;
    Vector3 upper;
    std::vector<std::uint32_t> candidates;
  };

  /**
   * Returns the candidate triangles of the paths from a cell to a physical source, collecting
   * them when the calling thread last asked for another cell or source.
   */
  const HullCandidateCache& GetHullCandidates(const Cell& cell, const SourcePoint& source_point);

  /// The statistics that the calling thread gathered since its last flush.
  static HullStatistics& ThreadHullStatistics();

  /// Adds the statistics of the calling thread to those of the problem and resets them. Each
  /// thread that traces paths calls this when it is done. Thread-safe.
  void FlushHullThreadStatistics();

  /// Adds what the statistics of the calling thread gained since @p before to those of a local
  /// cell, when the statistics of the cells are kept. A cell is traced by one thread at a time.
  void AddHullCellStatistics(size_t cell_local_id, HullStatistics before);

  /// Sums the statistics over the ranks and reports them from rank 0. When the statistics of
  /// the cells are kept, gathers them on rank 0. Collective.
  void ReportHullStatistics();

  /// Writes the diagnostics file from rank 0. Collective.
  void WriteDiagnostics(const std::string& file_name) const;

  /**
   * Conservation correction of one ray-traced cell, Eqs. (24)-(25) of Woodsford et al. (2026).
   *
   * In each group, the leakage out of the cell and the removal in it are scaled by
   * alpha = (source + inflow) / (outflow + sigma_t * integral of the projected flux), so that
   * the cell balance closes. When the denominator vanishes although the numerator does not,
   * alpha is one and the outgoing faces share the numerator equally.
   *
   * @param orientations Orientation of each face with respect to the source.
   * @param vacuum_boundary_faces True for the boundary faces that are not reflecting.
   * @param sigma_t Total cross section of the cell, by group.
   * @param projected_integral Integral of the projected flux over the cell, by group.
   * @param source In: rate at which the point source emits into the cell, by group. Out: that
   *        rate plus the leakage into the cell.
   * @param leakage Indexed [face][group]. In: the corrected leakage of the upwind cell on the
   *        incoming faces and the ray-traced leakage on the outgoing faces. Out: the outgoing
   *        faces hold the corrected leakage.
   * @param alpha Out: scale factor to apply to the flux of the cell, by group.
   * @param leakage_fallback Out: nonzero for the groups in which the outgoing faces share the
   *        source equally.
   * @param summary Totals to which this cell is added.
   * @param out_flow Total to which the corrected leakage through the vacuum boundary faces is
   *        added.
   */
  void CorrectCellBalance(std::uint64_t cell_global_id,
                          const std::vector<FaceOrientation>& orientations,
                          const std::vector<bool>& vacuum_boundary_faces,
                          const std::vector<double>& sigma_t,
                          const std::vector<double>& projected_integral,
                          std::vector<double>& source,
                          std::vector<std::vector<double>>& leakage,
                          std::vector<double>& alpha,
                          std::vector<char>& leakage_fallback,
                          ConservationSummary& summary,
                          double& out_flow) const;

  /**
   * Ray traces the near-source region of a source and enforces conservation cell by cell.
   *
   * The cells are first traced independently, on OPENSN_NUM_THREADS threads, and then corrected
   * one after the other in sweep order. The result does not depend on the number of threads.
   */
  void RaytraceNearSourceRegion(const SourcePoint& source_point);

  /**
   * Conservation correction of the ray-traced cells of all ranks, for method = "ray_trace".
   *
   * The correction of a cell needs the corrected leakage of the cells upwind of it, which can be
   * on other ranks. What each cell contributes, a few numbers per face and group, is gathered on
   * rank 0, which corrects the cells of the whole mesh from the source outward with
   * CorrectCellBalance and sends each rank the scale factors of its cells. The order depends
   * only on the mesh, so the result does not depend on the partitioning. Collective.
   *
   * Rank 0 adds the totals of all cells to @p summary, @p source_angle_fraction and the two
   * counts, sets @p num_cells to the number of cells of all ranks, and adds the corrected
   * leakage through the vacuum boundary to out_flow_.
   */
  void CorrectRayTracedCells(const SourcePoint& source_point,
                             std::vector<RawNearSourceCell>& raw_cells,
                             ConservationSummary& summary,
                             double& source_angle_fraction,
                             size_t& exact_neighbor_count,
                             size_t& exact_neighbor_fallback_count,
                             size_t& num_cells);

  /**
   * Ray traces one near-source cell: the leakage through its outgoing faces, its flux projected
   * on the finite element basis, and the upwind face terms of its bulk neighbors, all before
   * the conservation correction. It reads nothing that the tracing or the correction of another
   * cell writes, so several threads may trace different cells, each with its own ray tracer.
   */
  void TraceNearSourceCell(RayTracer& ray_tracer,
                           const SourcePoint& source_point,
                           size_t c,
                           bool is_exact_neighbor,
                           RawNearSourceCell& raw);

  void ProjectReflectedImageSources(unsigned int progress_interval);

  /// Uncollided flux at @p qp_xyz, a point of @p cell. @p face_index is the face of the cell
  /// the point lies on, or -1 for a point inside the cell.
  std::vector<double> RaytraceLine(RayTracer& ray_tracer,
                                   const Cell& cell,
                                   const Vector3& qp_xyz,
                                   const SourcePoint& source_point,
                                   int face_index = -1,
                                   double tolerance = 1.0e-12);

  /// Zero-allocation variant: writes into pre-allocated @p phi_out and reuses the three
  /// scratch vectors. All four buffers must be sized to @p num_groups_ before the first call
  /// (they are reset internally each call, so no manual clearing is needed between calls).
  /// On return, @p scratch_mfp holds the optical depth from the source to @p qp_xyz per group.
  /// The optical depth is computed by the ray tracer selected with the "ray_tracer" parameter.
  void RaytraceLineInto(RayTracer& ray_tracer,
                        const Cell& cell,
                        const Vector3& qp_xyz,
                        const SourcePoint& source_point,
                        std::vector<double>& phi_out,
                        std::vector<std::pair<size_t, double>>& scratch_segs,
                        std::vector<double>& scratch_bp,
                        std::vector<double>& scratch_mfp,
                        int face_index = -1,
                        double tolerance = 1.0e-12);

  /// Number of displaced start points tried for a path whose crossings are not consistent.
  static constexpr size_t num_start_displacements = 4;

  /**
   * Returns @p point displaced into its cell: by @p attempt times 1e-10 of @p to_centroid, the
   * vector from the point to the cell centroid, plus 1e-11 of its length in a direction that
   * differs for each attempt, 1 to num_start_displacements. A path that passes through an edge
   * within round-off, or lies in the plane of a face, has no consistent set of crossings; the
   * path from the displaced point does, and its lengths differ by about 1e-10 of the cell size.
   */
  static Vector3
  DisplaceStartPoint(const Vector3& point, const Vector3& to_centroid, size_t attempt);

  /**
   * Returns the part of the path from @p point towards a source between the fractions @p begin
   * and @p end of the unfolded path, which runs along @p unfolded_direction, folded back into
   * the domain by reflection in the planes it has crossed. For a physical source there is no
   * plane and the path is returned unchanged.
   */
  static std::pair<Vector3, Vector3> FoldSegment(const Vector3& point,
                                                 const Vector3& unfolded_direction,
                                                 double begin,
                                                 double end,
                                                 const std::vector<ReflectionPlane>& planes);

  /**
   * Optical depth between a point of a cell and a source, by walking from cell to cell.
   *
   * @p breakpoints are the fractions of the unfolded path at which it crosses a reflecting
   * plane, with 0 and 1 at the ends. The path through each cell is accumulated in
   * @p scratch_segs.
   */
  void ComputeCellWalkOpticalDepth(RayTracer& ray_tracer,
                                   const Cell& cell,
                                   const Vector3& qp_xyz,
                                   const SourcePoint& source_point,
                                   std::vector<std::pair<size_t, double>>& scratch_segs,
                                   const std::vector<double>& breakpoints,
                                   std::vector<double>& optical_depth,
                                   double tolerance);

  /**
   * Walks a straight segment from cell to cell and appends the length of the segment in each
   * cell to @p segments, which is cleared first.
   *
   * The cell walk normally finds the face through which a path leaves a cell with a triangle
   * test that has no rule for a path through an edge or a vertex: it can then reject every
   * triangle around the edge, or accept one that leads to a cell the path only touches. This walk
   * uses IntersectSegmentTriangle in one frame for the whole segment, which resolves an edge or
   * a vertex as if the segment were displaced by an infinitesimal amount, the same for every
   * cell. The segment then leaves each convex cell through exactly one side triangle, and a
   * cell it only touches is crossed with a zero length.
   *
   * Requires the precomputed face data of use_fast_trace_ (convex cells). Returns false, with
   * @p segments unspecified, when the walk does not end in a cell that contains @p segment_end
   * or on a boundary face at @p segment_end. This happens for a segment that lies in the plane
   * of a face, whose edges in that plane are seen end-on; the caller then displaces the start
   * point. Thread-safe.
   */
  bool WalkSegmentWatertight(size_t start_cell,
                             const Vector3& segment_start,
                             const Vector3& segment_end,
                             std::vector<std::pair<size_t, double>>& segments,
                             size_t& end_cell,
                             double tolerance) const;

  /**
   * Optical depth between a point of a cell and a source, from the lengths of the path in each
   * material. The path is intersected with the material hull only, so its cost does not grow
   * with the number of cells it crosses.
   *
   * The crossings must be consistent with the material of the cell; for a point on a face
   * (@p face_index >= 0) the material beyond the face is accepted too. When they are not, which
   * can only happen when the path passes through a hull edge or vertex within round-off, the
   * start point is displaced into the cell by about 1e-10 of its distance to the cell centroid
   * and the path is traced again. Thread-safe.
   */
  void ComputeHullOpticalDepth(const Cell& cell,
                               const Vector3& qp_xyz,
                               const SourcePoint& source_point,
                               const std::vector<double>& breakpoints,
                               int face_index,
                               std::vector<double>& optical_depth,
                               double tolerance);

  /// Builds the material hull, its tracer and the hull data of the source points. Collective.
  void BuildMaterialHull();

  /// Builds the direction grid of a physical source, with hull_tracer_search =
  /// "source_directions". Not thread-safe.
  void BuildHullDirectionGrid(const SourcePoint& source_point);

  /// Integrals of the uncollided flux of a point source over a cell.
  struct ExactCellIntegrals
  {
    /// Volume integral of the scalar flux over the cell, per group.
    std::vector<double> volume_integral;
    /// Outgoing leakage through each cell face, indexed [face][group]; zero for other faces.
    std::vector<std::vector<double>> face_leakage;
  };

  /**
   * Integrates the uncollided flux of a point source over a cell, and its leakage through each
   * outgoing face, exactly over direction.
   *
   * Every direction from the source through the cell leaves it through an outgoing face. Along
   * it the flux is S exp(-tau(r)) / (4 pi r^2), so with tau_in the optical depth from the source
   * to where the ray enters the cell and Delta the chord through the homogeneous cell, the leakage
   * integrand is S exp(-tau_in - sigma_t Delta) / (4 pi) and the volume integrand is
   * S exp(-tau_in) (1 - exp(-sigma_t Delta)) / (4 pi sigma_t). The optical depth comes from ray
   * tracing, so heterogeneous media are handled. For the cell containing the source, the ray
   * starts in the cell, so tau_in = 0 and Delta is the distance to the face.
   *
   * The directions through each PWL side triangle of each outgoing face are integrated with a
   * composite Gauss-Legendre rule in variables that keep the integrands smooth however close the
   * source is to a face, edge, or vertex, at points that lie on the face.
   *
   * Returns std::nullopt when a ray from the source could cross the cell more than once: for a
   * cell that does not contain the source, when it is not convex; and for the cell containing
   * the source, when a face is not directly visible from it. Also returns std::nullopt when the
   * source is on the inner side of a face that is not outgoing.
   */
  std::optional<ExactCellIntegrals>
  IntegrateCellExactly(RayTracer& ray_tracer, const Cell& cell, const SourcePoint& source_point);

  /**
   * Returns the fraction of the full solid angle that a cell subtends at a point on or in it.
   *
   * The signed solid angles of the PWL side triangles of the cell's faces are summed, so the
   * result is exact for cells of any shape, including non-convex ones. It is 1 for a point
   * inside the cell, 1/2 on a planar face, and smaller on an edge or vertex.
   */
  double SubtendedAngleFraction(const Cell& cell, const Vector3& point) const;

  /// True when every vertex of the cell lies on the inner side of the plane of every face.
  bool IsConvex(const Cell& cell) const;

  void SweepBulkRegion(const SourcePoint& source_point);

  UncollidedMatrices ComputeUncollidedIntegrals(const Cell& cell, const Vector3& pt_loc);

  void UpdateBalance(const SourcePoint& source_point);

  void AccumulateMoments(const Vector3& pt_loc);

  void WriteFluxMoments(hid_t file);

  void FinalizeBalance(hid_t file);

  bool IsReflectingBoundary(std::uint64_t boundary_id) const
  {
    return reflecting_boundary_ids_.count(boundary_id) != 0;
  }

  /// Near source region logical volumes.
  std::vector<std::shared_ptr<LogicalVolume>> near_source_logvols_;
  std::vector<SourcePoint> source_points_;
  std::vector<SourcePoint> reflected_source_points_;
  /// Boundary conditions as (name, type).
  std::vector<std::pair<std::string, std::string>> boundary_conditions_;
  std::vector<ReflectionPlane> reflection_planes_;
  std::set<std::uint64_t> reflecting_boundary_ids_;
  /// Cell face orientations for the cells in the local cell graph.
  std::vector<std::vector<FaceOrientation>> cell_face_orientations_;
  /// Near source uncollided sweep-plane local subgrid.
  std::vector<size_t> near_spls_;
  /// Bulk region uncollided sweep-plane local subgrid.
  std::vector<size_t> bulk_spls_;
  std::vector<CellRegion> cell_regions_;

  std::vector<double> destination_phi_;
  /// Bounding-box diagonal for each local cell, pre-computed once for RayTracer tolerance scaling.
  std::vector<double> cell_sizes_;
  /// False when the mesh contains any geometrically non-convex cell; true for standard hex/tet
  /// meshes.
  bool all_cells_convex_ = true;

  /// Precomputed face vertex data -- bypasses the std::map-backed VertexHandler in the hot path.
  /// Supports faces with up to max_sides vertices (covers common polyhedral mesh faces).
  struct FaceVertData
  {
    static constexpr uint32_t max_sides = 8;
    Vector3 verts[max_sides]; // inline vertex positions (no tree traversal)
    Vector3 centroid;
    uint64_t neighbor_id = 0;
    uint32_t num_sides = 0;
    uint32_t pad = 0;
  };
  std::vector<FaceVertData>
    all_face_verts_; // flat: [cell_0_face_0, cell_0_face_1, ..., cell_1_face_0, ...]
  std::vector<uint32_t> cell_face_offsets_; // [num_cells] first face index in all_face_verts_
  std::vector<uint32_t> cell_num_faces_;    // [num_cells] number of faces per cell
  /// O(1) global-to-local cell-ID lookup; replaces two O(log N) std::map calls per while-loop step.
  std::unordered_map<uint64_t, uint32_t> global_to_local_id_;
  /// True when all_cells_convex_ and all faces have <= FaceVertData::max_sides vertices.
  bool use_fast_trace_ = false;

  RayTracerType ray_tracer_type_ = RayTracerType::CELL_WALK;
  /// Name of the SegmentTracer implementation used by the material-hull ray tracer.
  std::string hull_tracer_backend_;
  /// True when every cell is ray traced and corrected for conservation and no cell is swept
  /// (method = "ray_trace").
  bool ray_trace_all_cells_ = false;
  HullSearch hull_search_ = HullSearch::SEGMENT;
  /// Number of bins along each side of a face of the direction grid.
  unsigned int hull_direction_bins_ = 0;
  /// Largest number of triangles in a leaf of the hierarchy of the "bvh" tracer.
  unsigned int hull_leaf_size_ = 0;
  /// True when the work of the material-hull ray tracer is timed and reported.
  bool hull_tracer_statistics_ = false;
  /// Identifies the hull built by the last BuildMaterialHull among all hulls of the process.
  std::uint64_t hull_generation_ = 0;
  std::shared_ptr<const MaterialHull> hull_;
  std::shared_ptr<const SegmentTracer> hull_tracer_;
  /// The triangles of the hull by direction seen from the physical source being traced, with
  /// hull_tracer_search = "source_directions".
  std::unique_ptr<const DirectionGridTracer> hull_direction_grid_;

  /// Total cross section of each hull material, indexed [material][group]. The last material is
  /// the exterior, a void.
  std::vector<std::vector<double>> hull_sigma_t_;
  /// Hull material of each local cell.
  std::vector<std::uint32_t> cell_hull_materials_;
  /// Number of paths traced again from a displaced start point.
  std::atomic<size_t> hull_retraced_segments_ = 0;
  /// Statistics of the material-hull ray tracer: those of this rank while paths are traced, those
  /// of all ranks after ReportHullStatistics.
  HullStatistics hull_statistics_;
  std::mutex hull_statistics_mutex_;
  /// Statistics of the paths traced for each local cell, summed over the sources. Kept when
  /// the statistics are on and a diagnostics file is written; empty otherwise.
  std::vector<HullStatistics> hull_cell_statistics_;
  /// On rank 0 after ReportHullStatistics: the global ids of the cells of all ranks, in
  /// increasing order, and their statistics, indexed [cell * number of entries + entry].
  std::vector<std::uint64_t> hull_cell_ids_;
  std::vector<std::uint64_t> hull_cell_values_;
  /// Number of path pieces that passed through an edge or a vertex of the mesh and were walked
  /// again with WalkSegmentWatertight.
  std::atomic<size_t> cell_walk_edge_paths_ = 0;
  /// Number of those that lay in the plane of a face and were walked from a displaced start
  /// point.
  std::atomic<size_t> cell_walk_displaced_paths_ = 0;
  /// Number of times the cell walk found no exit from a cell and continued from a point
  /// displaced by the mesh ray tracer, which makes the optical depth of that path approximate.
  /// This remains for meshes with non-convex cells or faces of more than
  /// FaceVertData::max_sides vertices.
  std::atomic<size_t> cell_walk_nudged_paths_ = 0;

  PhaseTimings timings_;
  /// True while Execute collects the data of the diagnostics file.
  bool write_diagnostics_ = false;
  std::vector<SourceDiagnostics> source_diagnostics_;

  unsigned int ell_max_ = 0;
  /// Integrate the leakages and volume integrals of the cells that share a vertex with a point
  /// source's cell exactly over direction instead of with spatial quadrature.
  bool exact_near_source_leakage_ = false;
  std::vector<Moment> moments_;
  std::vector<std::string> moment_names_;
  std::vector<std::vector<double>> accumulated_moments_;

  double production_ = 0.;
  double physical_removal_ = 0.;
  double reflected_removal_ = 0.;
  double out_flow_ = 0.;

public:
  static InputParameters GetInputParameters();
  static std::shared_ptr<UncollidedProblem> Create(const ParameterBlock& params);
};

} // namespace opensn
