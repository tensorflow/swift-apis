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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_one_hot.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp BuildOneHot(xla::XlaOp indices, xla::XlaOp on_value,
                       xla::XlaOp off_value, xla::int64 depth,
                       xla::int64 axis) {
  xla::XlaBuilder* builder = indices.builder();
  xla::Shape indices_shape = XlaHelpers::ShapeOfXlaOp(indices);
  std::vector<xla::int64> broadcast_dims(indices_shape.dimensions().size());
  if (axis < 0) axis = axis + broadcast_dims.size() + 1;
  std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
  std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);

  std::vector<xla::int64> output_dimensions(indices_shape.dimensions().size() +
                                            1);
  output_dimensions.assign(indices_shape.dimensions().begin(),
                           indices_shape.dimensions().end());
  output_dimensions.insert(output_dimensions.begin() + axis, depth);
  xla::Shape iota_shape = xla::ShapeUtil::MakeShape(
      indices_shape.element_type(), output_dimensions);

  return xla::Select(
      xla::Eq(indices, xla::Iota(builder, iota_shape, axis), broadcast_dims),
      xla::Broadcast(on_value, output_dimensions),
      xla::Broadcast(off_value, output_dimensions));
}

xla::Shape NodeOutputShape(const Value& indices, const Value& on_value,
                           const Value& off_value, xla::int64 depth,
                           xla::int64 axis) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    CHECK_EQ(operands.size(), 3)
        << "Unexpected number of operands: " << operands.size();
    return BuildOneHot(operands[0], operands[1], operands[2], depth, axis);
  };
  return InferOutputShape(
      {indices.shape(), on_value.shape(), off_value.shape()},
      lower_for_shape_fn);
}

}  // namespace

TfOneHot::TfOneHot(const Value& indices, const Value& on_value,
                   const Value& off_value, xla::int64 depth, xla::int64 axis)
    : Node(ir::OpKind(at::aten::tf_one_hot), {indices, on_value, off_value},
           [&] {
             return NodeOutputShape(indices, on_value, off_value, depth, axis);
           },
           1, xla::util::MHash(depth, axis)),
      depth_(depth),
      axis_(axis) {}

NodePtr TfOneHot::Clone(OpList operands) const {
  return MakeNode<TfOneHot>(operands.at(0), operands.at(1), operands.at(2),
                            depth_, axis_);
}

XlaOpVector TfOneHot::Lower(LoweringContext* loctx) const {
  xla::XlaOp indices = loctx->GetOutputOp(operand(0));
  xla::XlaOp on_value = loctx->GetOutputOp(operand(1));
  xla::XlaOp off_value = loctx->GetOutputOp(operand(2));
  xla::XlaOp output = BuildOneHot(indices, on_value, off_value, depth_, axis_);
  return ReturnOp(output, loctx);
}

std::string TfOneHot::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", depth=" << depth_ << ", axis=" << axis_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
