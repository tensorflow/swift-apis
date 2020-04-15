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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/leaky_relu_backward.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"

namespace swift_xla {
namespace ir {
namespace ops {

LeakyReluBackward::LeakyReluBackward(const Value& grad_output,
                                     const Value& input, double negative_slope)
    : Node(ir::OpKind(at::aten::leaky_relu_backward), {grad_output, input},
           input.shape(),
           /*num_outputs=*/1, xla::util::MHash(negative_slope)),
      negative_slope_(negative_slope) {}

NodePtr LeakyReluBackward::Clone(OpList operands) const {
  return MakeNode<LeakyReluBackward>(operands.at(0), operands.at(1),
                                     negative_slope_);
}

XlaOpVector LeakyReluBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      BuildLeakyReluBackward(grad_output, input, negative_slope_);
  return ReturnOp(output, loctx);
}

std::string LeakyReluBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", negative_slope=" << negative_slope_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
