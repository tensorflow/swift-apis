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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/masked_scatter.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/xla_lower_util.h"

namespace swift_xla {
namespace ir {
namespace ops {

MaskedScatter::MaskedScatter(const Value& input, const Value& mask,
                             const Value& source)
    : Node(ir::OpKind(at::aten::masked_scatter), {input, mask, source},
           input.shape(),
           /*num_outputs=*/1) {}

NodePtr MaskedScatter::Clone(OpList operands) const {
  return MakeNode<MaskedScatter>(operands.at(0), operands.at(1),
                                 operands.at(2));
}

XlaOpVector MaskedScatter::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp mask = loctx->GetOutputOp(operand(1));
  xla::XlaOp source = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildMaskedScatter(input, mask, source), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
