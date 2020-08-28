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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/dynamic_slice.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

std::vector<Value> ConcatArguments(const Value& base,
                                   absl::Span<const Value> start_indices) {
  std::vector<Value> out;
  out.reserve(start_indices.size() + 1);
  out.push_back(base);
  for (auto& tmp : start_indices) out.push_back(tmp);
  return out;
}

xla::Shape NodeOutputShape(const Value& base,
                           absl::Span<const Value> start_indices,
                           absl::Span<const xla::int64> slice_shape) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::DynamicSlice(operands[0], operands.subspan(1), slice_shape);
  };
  std::vector<xla::Shape> argument_shapes;
  argument_shapes.reserve(1 + start_indices.size());
  argument_shapes.push_back(base.shape());
  for (auto& value : start_indices) argument_shapes.push_back(value.shape());
  return InferOutputShape(argument_shapes, lower_for_shape_fn);
}

}  // namespace

DynamicSlice::DynamicSlice(const Value& base,
                           absl::Span<const Value> start_indices,
                           absl::Span<const xla::int64> slice_shape)
    : Node(
          ir::OpKind(at::aten::xla_dynamic_slice), ConcatArguments(base, start_indices),
          [&]() { return NodeOutputShape(base, start_indices, slice_shape); },
          /*num_outputs=*/1, xla::util::MHash(slice_shape)), slice_shapes_(slice_shape.begin(),
                                                        slice_shape.end()) {}

NodePtr DynamicSlice::Clone(OpList operands) const {
  return MakeNode<DynamicSlice>(operands.at(0), operands.subspan(1), slice_shapes_);
}

XlaOpVector DynamicSlice::Lower(LoweringContext* loctx) const {
  xla::XlaOp base = loctx->GetOutputOp(operand(0));
  std::vector<xla::XlaOp> start_indices;
  size_t count = operands().size() - 1;
  for (size_t i = 0; i < count; ++i) {
    start_indices.push_back(loctx->GetOutputOp(operand(1 + i)));
  }
  xla::XlaOp output = xla::DynamicSlice(base, start_indices, slice_shapes_);
  return ReturnOp(output, loctx);
}

std::string DynamicSlice::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", base_indices=("
     << absl::StrJoin(slice_shapes_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
