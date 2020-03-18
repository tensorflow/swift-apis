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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tril.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/matrix.h"

namespace swift_xla {
namespace ir {
namespace ops {

Tril::Tril(const Value& input, xla::int64 diagonal)
    : Node(ir::OpKind(at::aten::tril), {input}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash(diagonal)),
      diagonal_(diagonal) {}

NodePtr Tril::Clone(OpList operands) const {
  return MakeNode<Tril>(operands.at(0), diagonal_);
}

XlaOpVector Tril::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildTril(input, diagonal_);
  return ReturnOp(output, loctx);
}

std::string Tril::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", diagonal=" << diagonal_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
