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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/dynamic_update_slice.h"

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

std::vector<Value> ConcatArguments(const Value& base, const Value& update,
                                   absl::Span<const Value> start_indices) {
  std::vector<Value> out;
  out.reserve(start_indices.size() + 2);
  out.push_back(base);
  out.push_back(update);
  for (auto& tmp : start_indices) out.push_back(tmp);
  return out;
}

xla::Shape NodeOutputShape(const Value& base, const Value& update,
                           absl::Span<const Value> start_indices) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::DynamicUpdateSlice(operands[0], operands[1],
                                   operands.subspan(2));
  };
  std::vector<xla::Shape> argument_shapes;
  argument_shapes.reserve(2 + start_indices.size());
  argument_shapes.push_back(base.shape());
  argument_shapes.push_back(update.shape());
  for (auto& value : start_indices) argument_shapes.push_back(value.shape());
  return InferOutputShape(argument_shapes, lower_for_shape_fn);
}

}  // namespace

DynamicUpdateSlice::DynamicUpdateSlice(const Value& base, const Value& update,
                                       absl::Span<const Value> start_indices)
    : Node(
          ir::OpKind(at::aten::xla_dynamic_update_slice), ConcatArguments(base, update, start_indices),
          [&]() { return NodeOutputShape(base, update, start_indices); },
          /*num_outputs=*/1, 0x28935648) {}

NodePtr DynamicUpdateSlice::Clone(OpList operands) const {
  return MakeNode<DynamicUpdateSlice>(operands.at(0), operands.at(1),
                                      operands.subspan(2));
}

XlaOpVector DynamicUpdateSlice::Lower(LoweringContext* loctx) const {
  xla::XlaOp base = loctx->GetOutputOp(operand(0));
  xla::XlaOp update = loctx->GetOutputOp(operand(1));
  std::vector<xla::XlaOp> start_indices;
  size_t count = operands().size() - 2;
  for (size_t i = 0; i < count; ++i) {
    start_indices.push_back(loctx->GetOutputOp(operand(2 + i)));
  }
  xla::XlaOp output = xla::DynamicUpdateSlice(base, update, start_indices);
  return ReturnOp(output, loctx);
}

std::string DynamicUpdateSlice::ToString() const {
  std::stringstream ss;
  ss << Node::ToString();
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
