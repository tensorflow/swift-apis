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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/masked_fill.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/scalar.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"

namespace swift_xla {
namespace ir {
namespace ops {

MaskedFill::MaskedFill(const Value& input, const Value& mask, at::Scalar value)
    : Node(OpKind(at::aten::masked_fill), {input, mask}, input.shape(),
           /*num_outputs=*/1, ScalarHash(value)),
      value_(std::move(value)) {}

NodePtr MaskedFill::Clone(OpList operands) const {
  return MakeNode<MaskedFill>(operands.at(0), operands.at(1), value_);
}

XlaOpVector MaskedFill::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp mask = loctx->GetOutputOp(operand(1));
  xla::XlaOp zero = xla::Zero(loctx->builder(), XlaHelpers::TypeOfXlaOp(mask));
  xla::XlaOp mask_pred = xla::Ne(mask, zero);
  // Input shape is the same as output shape.
  const xla::Shape& input_shape = shape();
  xla::XlaOp value =
      xla::Broadcast(XlaHelpers::ScalarValue(value_, input_shape.element_type(),
                                             input.builder()),
                     input_shape.dimensions());
  return ReturnOp(xla::Select(mask_pred, value, input), loctx);
}

std::string MaskedFill::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
