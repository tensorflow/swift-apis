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

#include "xla_tensor/ops/put.h"

#include "xla_client/util.h"
#include "xla_tensor/lowering_context.h"
#include "xla_tensor/xla_lower_util.h"

namespace swift_xla {
namespace ir {
namespace ops {

Put::Put(const Value& input, const Value& index, const Value& source,
         bool accumulate)
    : Node(ir::OpKind(at::aten::put), {input, index, source}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash(accumulate)),
      accumulate_(accumulate) {}

NodePtr Put::Clone(OpList operands) const {
  return MakeNode<Put>(operands.at(0), operands.at(1), operands.at(2),
                       accumulate_);
}

XlaOpVector Put::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  xla::XlaOp source = loctx->GetOutputOp(operand(2));
  return ReturnOp(CreatePut(input, index, source, accumulate_), loctx);
}

std::string Put::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", accumulate=" << accumulate_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
