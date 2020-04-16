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

#include "xla_tensor/ops/native_batch_norm_backward.h"

#include "xla_client/debug_macros.h"
#include "xla_client/util.h"
#include "xla_tensor/batch_norm.h"
#include "xla_tensor/lowering_context.h"
#include "xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& grad_out, const Value& input,
                           const Value& weight, const Value& save_mean,
                           const Value& save_invstd, bool training) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    BatchNormGrads xla_outputs =
        BuildBatchNormBackward(operands[0], operands[1], operands[2],
                               operands[3], operands[4], training, 0.5);
    return xla::Tuple(operands[0].builder(),
                      {xla_outputs.grad_input, xla_outputs.grad_weight,
                       xla_outputs.grad_bias});
  };
  return InferOutputShape({grad_out.shape(), input.shape(), weight.shape(),
                           save_mean.shape(), save_invstd.shape()},
                          lower_for_shape_fn);
}

}  // namespace

NativeBatchNormBackward::NativeBatchNormBackward(
    const Value& grad_out, const Value& input, const Value& weight,
    const Value& save_mean, const Value& save_invstd, bool training, double eps)
    : Node(ir::OpKind(at::aten::native_batch_norm_backward),
           {grad_out, input, weight, save_mean, save_invstd},
           [&]() {
             return NodeOutputShape(grad_out, input, weight, save_mean,
                                    save_invstd, training);
           },
           /*num_outputs=*/3, xla::util::MHash(training, eps)),
      training_(training),
      eps_(eps) {}

NodePtr NativeBatchNormBackward::Clone(OpList operands) const {
  return MakeNode<NativeBatchNormBackward>(operands.at(0), operands.at(1),
                                           operands.at(2), operands.at(3),
                                           operands.at(4), training_, eps_);
}

XlaOpVector NativeBatchNormBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_out = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp weight = loctx->GetOutputOp(operand(2));
  xla::XlaOp save_mean = loctx->GetOutputOp(operand(3));
  xla::XlaOp save_invstd = loctx->GetOutputOp(operand(4));
  BatchNormGrads grads = BuildBatchNormBackward(
      grad_out, input, weight, save_mean, save_invstd, training_, eps_);
  return ReturnOps({std::move(grads.grad_input), std::move(grads.grad_weight),
                    std::move(grads.grad_bias)},
                   loctx);
}

std::string NativeBatchNormBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", training=" << training_ << ", eps=" << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
