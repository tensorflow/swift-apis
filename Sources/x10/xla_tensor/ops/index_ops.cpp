// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/index_ops.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/arithmetic_ir_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/expand.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/index_get.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/index_put.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/permute.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/scalar.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/xla_lower_util.h"

namespace swift_xla {
namespace {

// Wraps index tensors once into the [0, dim_size) interval, where dim_size is
// the size of the current indexed dimension.
std::vector<XLATensor> WrapIndicesOnce(const XLATensor& base,
                                       absl::Span<const XLATensor> indices,
                                       int start_dim) {
  std::vector<XLATensor> canonical_indices;
  auto base_shape_ref = base.shape();
  XLA_CHECK_LE(indices.size(), base_shape_ref.get().rank());
  for (size_t dim_idx = 0; dim_idx < indices.size(); ++dim_idx) {
    const XLATensor& dim_index = indices[dim_idx];
    int64_t dim_size = base_shape_ref.get().dimensions(dim_idx + start_dim);
    XLATensor wrapped_dim_index = XLATensor::Create(
        dim_index.GetIrValue() +
            XLATensor::GetIrValueForScalar(dim_size, dim_index.shape(),
                                           base.GetDevice()),
        base.GetDevice());
    XLATensor wrap_cond =
        XLATensor::lt(indices[dim_idx], at::Scalar(int64_t(0)));
    canonical_indices.push_back(
        XLATensor::where(wrap_cond, wrapped_dim_index, dim_index));
  }
  return canonical_indices;
}

ir::NodePtr IndexFillOp(const ir::Value& buffer, xla::int64 dim,
                        const ir::Value& index, const ir::Value& value) {
  auto lower_fn = [dim](const ir::Node& node,
                        ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_base = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_index = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_value = loctx->GetOutputOp(node.operand(2));
    return node.ReturnOp(CreateIndexFill(xla_base, dim, xla_index, xla_value),
                         loctx);
  };
  auto lower_for_shape_fn =
      [dim](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateIndexFill(operands[0], dim, operands[1], operands[2]);
  };
  ir::Value index_rank1 = EnsureRank1(index);
  return ir::ops::GenericOp(
      ir::OpKind(at::aten::index_fill), {buffer, index_rank1, value},
      [&]() {
        return ir::ops::InferOutputShape(
            {buffer.shape(), index_rank1.shape(), value.shape()},
            lower_for_shape_fn);
      },
      std::move(lower_fn), /*num_outputs=*/1, xla::util::MHash(dim));
}

ir::NodePtr IndexAddOp(const ir::Value& buffer, xla::int64 dim,
                       const ir::Value& index, const ir::Value& source) {
  auto lower_fn = [dim](const ir::Node& node,
                        ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_base = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_index = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_source = loctx->GetOutputOp(node.operand(2));
    return node.ReturnOp(CreateIndexAdd(xla_base, dim, xla_index, xla_source),
                         loctx);
  };
  auto lower_for_shape_fn =
      [dim](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateIndexAdd(operands[0], dim, operands[1], operands[2]);
  };
  ir::Value index_rank1 = EnsureRank1(index);
  return ir::ops::GenericOp(
      ir::OpKind(at::aten::index_add), {buffer, index_rank1, source},
      [&]() {
        return ir::ops::InferOutputShape(
            {buffer.shape(), index_rank1.shape(), source.shape()},
            lower_for_shape_fn);
      },
      std::move(lower_fn));
}

ir::NodePtr IndexCopyOp(const ir::Value& buffer, xla::int64 dim,
                        const ir::Value& index, const ir::Value& source) {
  auto lower_fn = [dim](const ir::Node& node,
                        ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_base = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_index = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_source = loctx->GetOutputOp(node.operand(2));
    return node.ReturnOp(CreateIndexCopy(xla_base, dim, xla_index, xla_source),
                         loctx);
  };
  auto lower_for_shape_fn =
      [dim](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateIndexCopy(operands[0], dim, operands[1], operands[2]);
  };
  ir::Value index_rank1 = EnsureRank1(index);
  return ir::ops::GenericOp(
      ir::OpKind(at::aten::index_copy), {buffer, index_rank1, source},
      [&]() {
        return ir::ops::InferOutputShape(
            {buffer.shape(), index_rank1.shape(), source.shape()},
            lower_for_shape_fn);
      },
      std::move(lower_fn));
}

}  // namespace

ir::Value EnsureRank1(const ir::Value& index) {
  XLA_CHECK_LE(index->shape().rank(), 1);
  return index->shape().rank() == 0
             ? ir::MakeNode<ir::ops::Expand>(index, std::vector<xla::int64>{1})
             : index;
}

XLATensor IndexByTensors(const XLATensor& base,
                         absl::Span<const XLATensor> indices,
                         xla::int64 start_dim) {
  if (indices.empty()) {
    return base;
  }
  auto canonical_indices = WrapIndicesOnce(base, indices, start_dim);
  xla::int64 indices_rank = canonical_indices.front().shape().get().rank();
  // Stack the indices to allow the whole multi-indexing to be dispatched with a
  // single gather.
  XLATensor indices_nd = XLATensor::stack(canonical_indices, indices_rank);
  return XLATensor::Create(
      ir::MakeNode<ir::ops::IndexGet>(base.GetIrValue(),
                                      indices_nd.GetIrValue(), start_dim),
      base.GetDevice(), base.dtype());
}

ir::Value IndexPutByTensors(const XLATensor& base,
                            absl::Span<const XLATensor> indices,
                            xla::int64 start_dim, const XLATensor& values,
                            bool accumulate,
                            absl::Span<const xla::int64> result_permutation) {
  if (indices.empty()) {
    return base.GetIrValue();
  }
  auto canonical_indices = WrapIndicesOnce(base, indices, start_dim);
  xla::int64 indices_rank = canonical_indices.front().shape().get().rank();
  // Stack the indices to allow the whole multi-indexing to be dispatched with a
  // single scatter.
  XLATensor indices_nd = XLATensor::stack(canonical_indices, indices_rank);
  return ir::MakeNode<ir::ops::Permute>(
      ir::MakeNode<ir::ops::IndexPut>(base.GetIrValue(),
                                      indices_nd.GetIrValue(), start_dim,
                                      values.GetIrValue(), accumulate),
      xla::util::ToVector<xla::int64>(result_permutation));
}

ir::NodePtr IndexFill(const XLATensor& base, xla::int64 dim,
                      const XLATensor& index, at::Scalar value) {
  XLA_CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Fill index is expected to be of scalar type Long";
  XLA_CHECK_LE(index.shape().get().rank(), 1)
      << "Fill index is supposed to be a vector";
  return IndexFillOp(
      base.GetIrValue(), dim, index.GetIrValue(),
      XLATensor::GetIrValueForScalar(value, base.shape().get().element_type(),
                                     base.GetDevice()));
}

ir::NodePtr IndexFill(const XLATensor& base, xla::int64 dim,
                      const XLATensor& index, const XLATensor& value) {
  XLA_CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Fill index is expected to be of scalar type Long";
  XLA_CHECK_LE(index.shape().get().rank(), 1)
      << "Fill index is supposed to be a vector";
  XLA_CHECK_EQ(value.shape().get().rank(), 0)
      << "Fill only supports a 0-dimensional value tensor";
  return IndexFillOp(base.GetIrValue(), dim, index.GetIrValue(),
                     value.GetIrValue());
}

ir::Value IndexAdd(const XLATensor& base, xla::int64 dim,
                   const XLATensor& index, const XLATensor& source) {
  XLA_CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Add index is expected to be of scalar type Long";
  XLA_CHECK_LE(index.shape().get().rank(), 1)
      << "Add index is supposed to be a vector";
  return IndexAddOp(base.GetIrValue(), dim, index.GetIrValue(),
                    source.GetIrValue());
}

ir::Value IndexCopy(const XLATensor& base, xla::int64 dim,
                    const XLATensor& index, const XLATensor& source) {
  XLA_CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Copy index is expected to be of scalar type Long";
  XLA_CHECK_LE(index.shape().get().rank(), 1)
      << "Copy index is supposed to be a vector";
  return IndexCopyOp(base.GetIrValue(), dim, index.GetIrValue(),
                     source.GetIrValue());
}

}  // namespace swift_xla
