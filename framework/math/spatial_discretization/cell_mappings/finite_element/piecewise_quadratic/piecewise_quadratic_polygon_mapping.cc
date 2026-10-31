// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/piecewise_quadratic_polygon_mapping.h"
#include "framework/math/spatial_discretization/finite_element/finite_element_data.h"
#include "framework/mesh/mesh/mesh.h"
#include "framework/mesh/mesh/cell.h"
#include <stdexcept>

namespace opensn
{

namespace
{

std::unique_ptr<SubmeshTying>
MakeTying(const Mesh& grid, const Cell& cell, unsigned int degree, const TyingWeightProvider& w)
{
  if (cell.GetType() != CellType::POLYGON)
    throw std::invalid_argument("PieceWiseQuadraticPolygonMapping: the cell is not a polygon");
  return std::make_unique<SubmeshTying>(grid, cell, degree, w);
}

} // namespace

PieceWiseQuadraticPolygonMapping::PieceWiseQuadraticPolygonMapping(
  const Cell& poly_cell,
  std::shared_ptr<Mesh> ref_grid,
  const TriangleQuadrature& volume_quadrature,
  const LineQuadrature& surface_quadrature)
  : PieceWiseQuadraticPolygonMapping(poly_cell,
                                     ref_grid,
                                     volume_quadrature,
                                     surface_quadrature,
                                     2,
                                     std::make_shared<LeastSquaresTyingWeights>(2))
{
}

PieceWiseQuadraticPolygonMapping::PieceWiseQuadraticPolygonMapping(
  const Cell& poly_cell,
  std::shared_ptr<Mesh> ref_grid,
  const TriangleQuadrature& volume_quadrature,
  const LineQuadrature& surface_quadrature,
  unsigned int degree,
  std::shared_ptr<const TyingWeightProvider> weights)
  : PieceWiseQuadraticPolygonMapping(poly_cell,
                                     ref_grid,
                                     volume_quadrature,
                                     surface_quadrature,
                                     weights,
                                     MakeTying(*ref_grid, poly_cell, degree, *weights))
{
}

PieceWiseQuadraticPolygonMapping::PieceWiseQuadraticPolygonMapping(
  const Cell& poly_cell,
  std::shared_ptr<Mesh> ref_grid,
  const TriangleQuadrature& volume_quadrature,
  const LineQuadrature& surface_quadrature,
  std::shared_ptr<const TyingWeightProvider> weights,
  std::unique_ptr<SubmeshTying> tying)
  : CellMapping(
      ref_grid, poly_cell, tying->GetNumDofs(), tying->GetDofLocations(), tying->GetFaceDofs()),
    weights_(std::move(weights)),
    tying_(std::move(tying)),
    volume_quadrature_(volume_quadrature),
    surface_quadrature_(surface_quadrature)
{
  // Side triangles (v0, v1, x_c), exactly as in PieceWiseLinearPolygonMapping
  const Vector3& vc = poly_cell.centroid;
  const size_t num_faces = poly_cell.faces.size();
  if (tying_->GetNumSides() != num_faces)
    throw std::logic_error("PieceWiseQuadraticPolygonMapping: side enumeration mismatch");
  sides_.reserve(num_faces);

  for (size_t f = 0; f < num_faces; ++f)
  {
    const CellFace& face = poly_cell.faces[f];
    if (tying_->GetSideFace(f) != f)
      throw std::logic_error("PieceWiseQuadraticPolygonMapping: side enumeration mismatch");

    const auto& v0 = grid_->GlobalVertex(face.vertex_ids[0]);
    const auto& v1 = grid_->GlobalVertex(face.vertex_ids[1]);
    const Vector3 v01 = v1 - v0;
    const Vector3 v02 = vc - v0;

    SideData side;
    side.v0 = v0;
    side.detJ = v01.x * v02.y - v02.x * v01.y;
    if (side.detJ <= 0.0)
      throw std::invalid_argument(
        "PieceWiseQuadraticPolygonMapping: non-positive side-triangle Jacobian on cell " +
        std::to_string(poly_cell.global_id) + ", face " + std::to_string(f));
    side.detJ_surf = v01.Norm();
    side.normal = face.normal;

    side.J.SetIJ(0, 0, v01.x);
    side.J.SetIJ(1, 0, v01.y);
    side.J.SetIJ(0, 1, v02.x);
    side.J.SetIJ(1, 1, v02.y);
    side.J.SetIJ(2, 2, 0.0);

    side.Jinv.SetIJ(0, 0, v02.y / side.detJ);
    side.Jinv.SetIJ(1, 0, -v01.y / side.detJ);
    side.Jinv.SetIJ(0, 1, -v02.x / side.detJ);
    side.Jinv.SetIJ(1, 1, v01.x / side.detJ);
    side.Jinv.SetIJ(2, 2, 0.0);

    side.JTinv.SetIJ(0, 0, v02.y / side.detJ);
    side.JTinv.SetIJ(1, 0, -v02.x / side.detJ);
    side.JTinv.SetIJ(0, 1, -v01.y / side.detJ);
    side.JTinv.SetIJ(1, 1, v01.x / side.detJ);
    side.JTinv.SetIJ(2, 2, 0.0);

    sides_.push_back(side);
  }
}

void
PieceWiseQuadraticPolygonMapping::EvaluateInSide(size_t s,
                                                 const Vector3& xi,
                                                 std::vector<double>& values,
                                                 std::vector<Vector3>& grads) const
{
  const auto& ref = tying_->GetReferenceElement();
  const auto& ref_nodes = tying_->GetSideReferenceNodes();
  const auto& Ts = tying_->GetSideTying(s);
  const size_t N = num_nodes_;

  std::vector<double> ref_values;
  std::vector<Vector3> ref_grads;
  ref.Shape(Vector3{xi.x, xi.y, 0.0}, ref_values);
  ref.Grad(Vector3{xi.x, xi.y, 0.0}, ref_grads);

  values.assign(N, 0.0);
  grads.assign(N, Vector3{0.0, 0.0, 0.0});
  const auto& JTinv = sides_[s].JTinv;
  for (size_t k = 0; k < ref_nodes.size(); ++k)
  {
    const size_t t = ref_nodes[k];
    // The zeta derivative plays no role on the zeta = 0 face
    const Vector3 g = JTinv * Vector3{ref_grads[t].x, ref_grads[t].y, 0.0};
    for (size_t j = 0; j < N; ++j)
    {
      const double c = Ts(k, j);
      if (c == 0.0)
        continue;
      values[j] += c * ref_values[t];
      grads[j] += g * c;
    }
  }
}

bool
PieceWiseQuadraticPolygonMapping::LocateSide(const Vector3& xyz, size_t& side, Vector3& xi) const
{
  for (size_t s = 0; s < sides_.size(); ++s)
  {
    const Vector3 ref = sides_[s].Jinv * (xyz - sides_[s].v0);
    if ((ref.x >= -1.0e-12) and (ref.y >= -1.0e-12) and ((ref.x + ref.y) <= (1.0 + 1.0e-12)))
    {
      side = s;
      xi = Vector3{ref.x, ref.y, 0.0};
      return true;
    }
  }
  return false;
}

double
PieceWiseQuadraticPolygonMapping::ShapeValue(size_t i, const Vector3& xyz) const
{
  size_t s;
  Vector3 xi;
  if (not LocateSide(xyz, s, xi))
    return 0.0;
  std::vector<double> values;
  std::vector<Vector3> grads;
  EvaluateInSide(s, xi, values, grads);
  return values[i];
}

Vector3
PieceWiseQuadraticPolygonMapping::GradShapeValue(size_t i, const Vector3& xyz) const
{
  size_t s;
  Vector3 xi;
  if (not LocateSide(xyz, s, xi))
    return {0.0, 0.0, 0.0};
  std::vector<double> values;
  std::vector<Vector3> grads;
  EvaluateInSide(s, xi, values, grads);
  return grads[i];
}

void
PieceWiseQuadraticPolygonMapping::ShapeValues(const Vector3& xyz,
                                              Vector<double>& shape_values) const
{
  shape_values.Resize(num_nodes_, 0.0);
  shape_values.Set(0.0);
  size_t s;
  Vector3 xi;
  if (not LocateSide(xyz, s, xi))
    return;
  std::vector<double> values;
  std::vector<Vector3> grads;
  EvaluateInSide(s, xi, values, grads);
  for (size_t i = 0; i < num_nodes_; ++i)
    shape_values(i) = values[i];
}

void
PieceWiseQuadraticPolygonMapping::GradShapeValues(const Vector3& xyz,
                                                  std::vector<Vector3>& gradshape_values) const
{
  gradshape_values.assign(num_nodes_, Vector3{0.0, 0.0, 0.0});
  size_t s;
  Vector3 xi;
  if (not LocateSide(xyz, s, xi))
    return;
  std::vector<double> values;
  EvaluateInSide(s, xi, values, gradshape_values);
}

VolumetricFiniteElementData
PieceWiseQuadraticPolygonMapping::MakeVolumetricFiniteElementData() const
{
  const size_t num_vol_qpoints = volume_quadrature_.qpoints.size();
  const size_t ttl_num_vol_qpoints = sides_.size() * num_vol_qpoints;

  std::vector<unsigned int> V_quadrature_point_indices;
  std::vector<Vector3> V_qpoints_xyz;
  std::vector<std::vector<double>> V_shape_value(num_nodes_);
  std::vector<std::vector<Vector3>> V_shape_grad(num_nodes_);
  std::vector<double> V_JxW;

  V_quadrature_point_indices.reserve(ttl_num_vol_qpoints);
  for (unsigned int qp = 0; qp < ttl_num_vol_qpoints; ++qp)
    V_quadrature_point_indices.push_back(qp);

  for (size_t i = 0; i < num_nodes_; ++i)
  {
    V_shape_value[i].reserve(ttl_num_vol_qpoints);
    V_shape_grad[i].reserve(ttl_num_vol_qpoints);
  }
  V_JxW.reserve(ttl_num_vol_qpoints);
  V_qpoints_xyz.reserve(ttl_num_vol_qpoints);

  std::vector<double> values;
  std::vector<Vector3> grads;
  for (size_t s = 0; s < sides_.size(); ++s)
  {
    const auto& side = sides_[s];
    for (size_t qp = 0; qp < num_vol_qpoints; ++qp)
    {
      const auto& q = volume_quadrature_.qpoints[qp];
      const Vector3 xi{q.x, q.y, 0.0};
      EvaluateInSide(s, xi, values, grads);
      for (size_t i = 0; i < num_nodes_; ++i)
      {
        V_shape_value[i].push_back(values[i]);
        V_shape_grad[i].push_back(grads[i]);
      }
      V_JxW.push_back(side.detJ * volume_quadrature_.weights[qp]);
      V_qpoints_xyz.push_back(side.v0 + side.J * xi);
    }
  }

  return {V_quadrature_point_indices,
          V_qpoints_xyz,
          V_shape_value,
          V_shape_grad,
          V_JxW,
          face_node_mappings_,
          num_nodes_};
}

SurfaceFiniteElementData
PieceWiseQuadraticPolygonMapping::MakeSurfaceFiniteElementData(size_t face_index) const
{
  const size_t s = face_index;
  const auto& side = sides_[s];
  const size_t num_srf_qpoints = surface_quadrature_.qpoints.size();

  std::vector<unsigned int> F_quadrature_point_indices;
  std::vector<Vector3> F_qpoints_xyz;
  std::vector<std::vector<double>> F_shape_value(num_nodes_);
  std::vector<std::vector<Vector3>> F_shape_grad(num_nodes_);
  std::vector<double> F_JxW;
  std::vector<Vector3> F_normals;

  F_quadrature_point_indices.reserve(num_srf_qpoints);
  for (unsigned int qp = 0; qp < num_srf_qpoints; ++qp)
    F_quadrature_point_indices.push_back(qp);

  F_normals.assign(num_srf_qpoints, side.normal);

  for (size_t i = 0; i < num_nodes_; ++i)
  {
    F_shape_value[i].reserve(num_srf_qpoints);
    F_shape_grad[i].reserve(num_srf_qpoints);
  }
  F_JxW.reserve(num_srf_qpoints);
  F_qpoints_xyz.reserve(num_srf_qpoints);

  std::vector<double> values;
  std::vector<Vector3> grads;
  for (size_t qp = 0; qp < num_srf_qpoints; ++qp)
  {
    // Line quadrature points are (xi, 0): the eta = 0 edge of the side triangle
    const Vector3 xi{surface_quadrature_.qpoints[qp].x, 0.0, 0.0};
    EvaluateInSide(s, xi, values, grads);
    for (size_t i = 0; i < num_nodes_; ++i)
    {
      F_shape_value[i].push_back(values[i]);
      F_shape_grad[i].push_back(grads[i]);
    }
    F_JxW.push_back(side.detJ_surf * surface_quadrature_.weights[qp]);
    F_qpoints_xyz.push_back(side.v0 + side.J * xi);
  }

  return {F_quadrature_point_indices,
          F_qpoints_xyz,
          F_shape_value,
          F_shape_grad,
          F_JxW,
          F_normals,
          face_node_mappings_,
          face_node_mappings_[s].size()};
}

} // namespace opensn
