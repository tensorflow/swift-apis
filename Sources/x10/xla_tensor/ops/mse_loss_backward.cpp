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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/mse_loss_backward.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/mse_loss.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& grad_output, const Value& input,
                           const Value& target, ReductionMode reduction) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMseLossBackward(operands[0], operands[1], operands[2],
                                reduction);
  };
  return InferOutputShape({grad_output.shape(), input.shape(), target.shape()},
                          lower_for_shape_fn);
}

}  // namespace

MseLossBackward::MseLossBackward(const Value& grad_output, const Value& input,
                                 const Value& target, ReductionMode reduction)
    : Node(ir::OpKind(at::aten::mse_loss_backward),
           {grad_output, input, target},
           [&]() {
             return NodeOutputShape(grad_output, input, target, reduction);
           },
           /*num_outputs=*/1,
           xla::util::MHash(xla::util::GetEnumValue(reduction))),
      reduction_(reduction) {}

NodePtr MseLossBackward::Clone(OpList operands) const {
  return MakeNode<MseLossBackward>(operands.at(0), operands.at(1),
                                   operands.at(2), reduction_);
}

XlaOpVector MseLossBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp target = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildMseLossBackward(grad_output, input, target, reduction_),
                  loctx);
}

std::string MseLossBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", reduction=" << xla::util::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
