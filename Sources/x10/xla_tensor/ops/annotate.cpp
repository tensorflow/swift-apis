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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/annotate.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"

namespace swift_xla {
namespace ir {
namespace ops {

Annotate::Annotate(const Value& input, std::string annotation)
    : Node(ir::OpKind(at::aten::annotate), {input}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash()),
      annotation_(annotation) {}

NodePtr Annotate::Clone(OpList operands) const {
  return MakeNode<Annotate>(operands.at(0), annotation_);
}

XlaOpVector Annotate::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(input, loctx);
}

std::string Annotate::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", annotation=" << annotation_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla

