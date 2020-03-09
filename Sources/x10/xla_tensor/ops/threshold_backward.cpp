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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/threshold_backward.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"

namespace swift_xla {
namespace ir {
namespace ops {

ThresholdBackward::ThresholdBackward(const Value& grad_output,
                                     const Value& input, float threshold)
    : Node(ir::OpKind(at::aten::threshold_backward), {grad_output, input},
           input.shape(), /*num_outputs=*/1, xla::util::MHash(threshold)),
      threshold_(threshold) {}

NodePtr ThresholdBackward::Clone(OpList operands) const {
  return MakeNode<ThresholdBackward>(operands.at(0), operands.at(1),
                                     threshold_);
}

XlaOpVector ThresholdBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildThreshold(input, grad_output, threshold_, 0);
  return ReturnOp(output, loctx);
}

std::string ThresholdBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", threshold=" << threshold_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
