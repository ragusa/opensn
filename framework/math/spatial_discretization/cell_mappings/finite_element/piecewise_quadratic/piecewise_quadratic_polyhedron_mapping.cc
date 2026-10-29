// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/spatial_discretization/cell_mappings/finite_element/piecewise_quadratic/piecewise_quadratic_polyhedron_mapping.h"
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
  if (cell.GetType() != CellType::POLYHEDRON)
    throw std::invalid_argument(
      "PieceWiseQuadraticPolyhedronMapping: the cell is not a polyhedron");
  return std::make_unique<SubmeshTying>(grid, cell, degree, w);
}

} // namespace

PieceWiseQuadraticPolyhedronMapping::PieceWiseQuadraticPolyhedronMapping(
  const Cell& polyh_cell,
  std::shared_ptr<Mesh> ref_grid,
  const TetrahedraQuadrature& volume_quadrature,
  const TriangleQuadrature& surface_quadrature)
  : PieceWiseQuadraticPolyhedronMapping(polyh_cell,
                                        ref_grid,
                                        volume_quadrature,
                                        surface_quadrature,
                                        2,
                                        std::make_shared<LeastSquaresTyingWeights>(2))
{
}

PieceWiseQuadraticPolyhedronMapping::PieceWiseQuadraticPolyhedronMapping(
  const Cell& polyh_cell,
  std::shared_ptr<Mesh> ref_grid,
  const TetrahedraQuadrature& volume_quadrature,
  const TriangleQuadrature& surface_quadrature,
  unsigned int degree,
  std::shared_ptr<const TyingWeightProvider> weights)
  : PieceWiseQuadraticPolyhedronMapping(polyh_cell,
                                        ref_grid,
                                        volume_quadrature,
                                        surface_quadrature,
                                        weights,
                                        MakeTying(*ref_grid, polyh_cell, degree, *weights))
{
}

PieceWiseQuadraticPolyhedronMapping::PieceWiseQuadraticPolyhedronMapping(
  const Cell& polyh_cell,
  std::shared_ptr<Mesh> ref_grid,
  const TetrahedraQuadrature& volume_quadrature,
  const TriangleQuadrature& surface_quadrature,
  std::shared_ptr<const TyingWeightProvider> weights,
  std::unique_ptr<SubmeshTying> tying)
  : CellMapping(
      ref_grid, polyh_cell, tying->GetNumDofs(), tying->GetDofLocations(), tying->GetFaceDofs()),
    weights_(std::move(weights)),
    tying_(std::move(tying)),
    volume_quadrature_(volume_quadrature),
    surface_quadrature_(surface_quadrature)
{
  // Side tetrahedra, exactly as in PieceWiseLinearPolyhedronMapping
  const Vector3& vcc = polyh_cell.centroid;
  const size_t num_faces = polyh_cell.faces.size();
  face_sides_.resize(num_faces);
  face_normals_.reserve(num_faces);
  sides_.reserve(tying_->GetNumSides());

  for (size_t f = 0; f < num_faces; ++f)
  {
    const CellFace& face = polyh_cell.faces[f];
    face_normals_.push_back(face.normal);
    const Vector3& vfc = face.centroid;

    const size_t num_edges = face.vertex_ids.size();
    for (size_t e = 0; e < num_edges; ++e)
    {
      const size_t ep1 = (e < (num_edges - 1)) ? e + 1 : 0;
      const auto& v0 = grid_->GlobalVertex(face.vertex_ids[e]);
      const auto& v1 = vfc;
      const auto& v2 = grid_->GlobalVertex(face.vertex_ids[ep1]);
      const auto& v3 = vcc;

      SideData side;
      side.v0 = v0;

      const Vector3 v01 = v1 - v0;
      const Vector3 v02 = v2 - v0;
      const Vector3 v03 = v3 - v0;

      // Surface Jacobian: rotate into the face frame
      Vector3 normal = face.normal * -1.0;
      Vector3 tangent = v02.Cross(normal);
      tangent = tangent / tangent.Norm();
      const Vector3 binorm = v02 / v02.Norm();

      Matrix3x3 R;
      R.SetColJVec(0, tangent);
      R.SetColJVec(1, binorm);
      R.SetColJVec(2, normal);
      const Matrix3x3 Rinv = R.Inverse();
      const Vector3 v01N = Rinv * v01;
      const Vector3 v02N = Rinv * v02;
      side.detJ_surf = v01N.x * v02N.y - v01N.y * v02N.x;

      Matrix3x3 J;
      J.SetColJVec(0, v01);
      J.SetColJVec(1, v02);
      J.SetColJVec(2, v03);
      side.J = J;
      side.detJ = J.Det();
      if (side.detJ <= 0.0)
        throw std::invalid_argument(
          "PieceWiseQuadraticPolyhedronMapping: non-positive side-tetrahedron Jacobian on cell " +
          std::to_string(polyh_cell.global_id) + ", face " + std::to_string(f) + ", edge " +
          std::to_string(e));
      side.Jinv = J.Inverse();
      side.JTinv = J.Transpose().Inverse();

      const size_t s = sides_.size();
      if (tying_->GetSideFace(s) != f or tying_->GetSideEdge(s) != e)
        throw std::logic_error("PieceWiseQuadraticPolyhedronMapping: side enumeration mismatch");
      face_sides_[f].push_back(s);
      sides_.push_back(side);
    }
  }
}

void
PieceWiseQuadraticPolyhedronMapping::EvaluateInSide(size_t s,
                                                    const Vector3& xi,
                                                    std::vector<double>& values,
                                                    std::vector<Vector3>& grads) const
{
  const auto& ref = tying_->GetReferenceElement();
  const auto& Ts = tying_->GetSideTying(s);
  const size_t n_tet = ref.GetNumNodes();
  const size_t N = num_nodes_;

  std::vector<double> tet_values;
  std::vector<Vector3> tet_grads;
  ref.Shape(xi, tet_values);
  ref.Grad(xi, tet_grads);

  values.assign(N, 0.0);
  grads.assign(N, Vector3{0.0, 0.0, 0.0});
  const auto& JTinv = sides_[s].JTinv;
  for (size_t t = 0; t < n_tet; ++t)
  {
    const Vector3 g = JTinv * tet_grads[t];
    for (size_t j = 0; j < N; ++j)
    {
      const double c = Ts(t, j);
      if (c == 0.0)
        continue;
      values[j] += c * tet_values[t];
      grads[j] += g * c;
    }
  }
}

bool
PieceWiseQuadraticPolyhedronMapping::LocateSide(const Vector3& xyz, size_t& side, Vector3& xi) const
{
  for (size_t s = 0; s < sides_.size(); ++s)
  {
    const Vector3 ref = sides_[s].Jinv * (xyz - sides_[s].v0);
    if ((ref.x >= -1.0e-12) and (ref.y >= -1.0e-12) and (ref.z >= -1.0e-12) and
        ((ref.x + ref.y + ref.z) <= (1.0 + 1.0e-12)))
    {
      side = s;
      xi = ref;
      return true;
    }
  }
  return false;
}

double
PieceWiseQuadraticPolyhedronMapping::ShapeValue(size_t i, const Vector3& xyz) const
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
PieceWiseQuadraticPolyhedronMapping::GradShapeValue(size_t i, const Vector3& xyz) const
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
PieceWiseQuadraticPolyhedronMapping::ShapeValues(const Vector3& xyz,
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
PieceWiseQuadraticPolyhedronMapping::GradShapeValues(const Vector3& xyz,
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
PieceWiseQuadraticPolyhedronMapping::MakeVolumetricFiniteElementData() const
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
      const auto& xi = volume_quadrature_.qpoints[qp];
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
PieceWiseQuadraticPolyhedronMapping::MakeSurfaceFiniteElementData(size_t face_index) const
{
  const size_t f = face_index;
  const size_t num_srf_qpoints = surface_quadrature_.qpoints.size();
  const auto& fsides = face_sides_[f];
  const size_t ttl_num_face_qpoints = fsides.size() * num_srf_qpoints;

  std::vector<unsigned int> F_quadrature_point_indices;
  std::vector<Vector3> F_qpoints_xyz;
  std::vector<std::vector<double>> F_shape_value(num_nodes_);
  std::vector<std::vector<Vector3>> F_shape_grad(num_nodes_);
  std::vector<double> F_JxW;
  std::vector<Vector3> F_normals;

  F_quadrature_point_indices.reserve(ttl_num_face_qpoints);
  for (unsigned int qp = 0; qp < ttl_num_face_qpoints; ++qp)
    F_quadrature_point_indices.push_back(qp);

  F_normals.assign(ttl_num_face_qpoints, face_normals_[f]);

  for (size_t i = 0; i < num_nodes_; ++i)
  {
    F_shape_value[i].reserve(ttl_num_face_qpoints);
    F_shape_grad[i].reserve(ttl_num_face_qpoints);
  }
  F_JxW.reserve(ttl_num_face_qpoints);
  F_qpoints_xyz.reserve(ttl_num_face_qpoints);

  std::vector<double> values;
  std::vector<Vector3> grads;
  for (const auto s : fsides)
  {
    const auto& side = sides_[s];
    for (size_t qp = 0; qp < num_srf_qpoints; ++qp)
    {
      // Triangle quadrature points are (xi, eta, 0): the zeta = 0 face of the side tet
      const auto& xi = surface_quadrature_.qpoints[qp];
      EvaluateInSide(s, xi, values, grads);
      for (size_t i = 0; i < num_nodes_; ++i)
      {
        F_shape_value[i].push_back(values[i]);
        F_shape_grad[i].push_back(grads[i]);
      }
      F_JxW.push_back(side.detJ_surf * surface_quadrature_.weights[qp]);
      F_qpoints_xyz.push_back(side.v0 + side.J * xi);
    }
  }

  return {F_quadrature_point_indices,
          F_qpoints_xyz,
          F_shape_value,
          F_shape_grad,
          F_JxW,
          F_normals,
          face_node_mappings_,
          face_node_mappings_[f].size()};
}

} // namespace opensn
