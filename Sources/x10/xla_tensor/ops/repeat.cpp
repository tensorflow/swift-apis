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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/repeat.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> repeats) {
  auto lower_for_shape_fn =
      [repeats](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return BuildRepeat(operands[0], repeats);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Repeat::Repeat(const Value& input, std::vector<xla::int64> repeats)
    : Node(ir::OpKind(at::aten::repeat), {input},
           [&]() { return NodeOutputShape(input, repeats); },
           /*num_outputs=*/1, xla::util::MHash(repeats)),
      repeats_(std::move(repeats)) {}

NodePtr Repeat::Clone(OpList operands) const {
  return MakeNode<Repeat>(operands.at(0), repeats_);
}

XlaOpVector Repeat::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildRepeat(input, repeats_);
  return ReturnOp(output, loctx);
}

std::string Repeat::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", repeats=(" << absl::StrJoin(repeats_, ", ")
     << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
