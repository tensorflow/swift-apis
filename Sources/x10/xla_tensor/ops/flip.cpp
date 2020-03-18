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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/flip.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace swift_xla {
namespace ir {
namespace ops {

Flip::Flip(const Value& input, std::vector<xla::int64> dims)
    : Node(ir::OpKind(at::aten::flip), {input}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash(dims)),
      dims_(std::move(dims)) {}

NodePtr Flip::Clone(OpList operands) const {
  return MakeNode<Flip>(operands.at(0), dims_);
}

XlaOpVector Flip::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = xla::Rev(input, dims_);
  return ReturnOp(output, loctx);
}

std::string Flip::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dims=(" << absl::StrJoin(dims_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
