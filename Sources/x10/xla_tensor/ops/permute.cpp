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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/permute.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> dims) {
  auto lower_for_shape_fn =
      [dims](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return xla::Transpose(operands[0], dims);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Permute::Permute(const Value& input, std::vector<xla::int64> dims)
    : Node(ir::OpKind(at::aten::permute), {input},
           [&]() { return NodeOutputShape(input, dims); },
           /*num_outputs=*/1, xla::util::MHash(dims)),
      dims_(std::move(dims)) {}

NodePtr Permute::Clone(OpList operands) const {
  return MakeNode<Permute>(operands.at(0), dims_);
}

XlaOpVector Permute::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = xla::Transpose(input, dims_);
  return ReturnOp(output, loctx);
}

std::string Permute::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dims=(" << absl::StrJoin(dims_, ", ") << ")";
  return ss.str();
}

xla::Shape Permute::MakePermuteShape(const xla::Shape& source_shape,
                                     absl::Span<const xla::int64> permutation) {
  return XlaHelpers::GetDynamicReshape(
      source_shape,
      XlaHelpers::Permute(permutation, source_shape.dimensions()));
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
