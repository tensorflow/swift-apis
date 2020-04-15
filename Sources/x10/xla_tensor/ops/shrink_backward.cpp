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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/shrink_backward.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/scalar.h"

namespace swift_xla {
namespace ir {
namespace ops {

ShrinkBackward::ShrinkBackward(OpKind kind, const Value& grad_output,
                               const Value& input, at::Scalar lambda)
    : Node(kind, {grad_output, input}, input.shape(), /*num_outputs=*/1,
           ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string ShrinkBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

NodePtr ShrinkBackward::Clone(OpList operands) const {
  return MakeNode<ShrinkBackward>(op(), operands.at(0), operands.at(1),
                                  lambda_);
}

XlaOpVector ShrinkBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildShrinkBackward(grad_output, input, lambda_), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
