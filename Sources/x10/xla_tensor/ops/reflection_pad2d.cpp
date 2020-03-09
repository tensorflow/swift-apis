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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/reflection_pad2d.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> padding) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildReflectionPad2d(operands[0], padding);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

ReflectionPad2d::ReflectionPad2d(const Value& input,
                                 std::vector<xla::int64> padding)
    : Node(OpKind(at::aten::reflection_pad2d), {input},
           [&]() { return NodeOutputShape(input, padding); },
           /*num_outputs=*/1, xla::util::MHash(padding)),
      padding_(std::move(padding)) {}

NodePtr ReflectionPad2d::Clone(OpList operands) const {
  return MakeNode<ReflectionPad2d>(operands.at(0), padding_);
}

XlaOpVector ReflectionPad2d::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildReflectionPad2d(input, padding_);
  return ReturnOp(output, loctx);
}

std::string ReflectionPad2d::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", padding=(" << absl::StrJoin(padding_, ", ")
     << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
