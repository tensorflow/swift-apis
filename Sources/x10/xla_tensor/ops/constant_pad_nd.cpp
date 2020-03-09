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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/constant_pad_nd.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/scalar.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> pad) {
  xla::PrimitiveType input_element_type = input.shape().element_type();
  auto lower_for_shape_fn =
      [input_element_type,
       pad](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp xla_input = operands[0];
    return xla::Pad(xla_input,
                    xla::Zero(xla_input.builder(), input_element_type),
                    XlaHelpers::MakeXlaPaddingConfigFromNdPadding(pad));
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

ConstantPadNd::ConstantPadNd(const Value& input, std::vector<xla::int64> pad,
                             at::Scalar value)
    : Node(ir::OpKind(at::aten::constant_pad_nd), OpList{input},
           [&]() { return NodeOutputShape(input, pad); },
           /*num_outputs=*/1, xla::util::MHash(pad, ScalarHash(value))),
      pad_(std::move(pad)),
      value_(value) {}

NodePtr ConstantPadNd::Clone(OpList operands) const {
  return MakeNode<ConstantPadNd>(operands.at(0), pad_, value_);
}

XlaOpVector ConstantPadNd::Lower(LoweringContext* loctx) const {
  Output input = operand(0);
  xla::XlaOp xla_input = loctx->GetOutputOp(input);
  xla::XlaOp xla_output = xla::Pad(
      xla_input,
      XlaHelpers::ScalarValue(value_, input.node->shape().element_type(),
                              loctx->builder()),
      XlaHelpers::MakeXlaPaddingConfigFromNdPadding(pad_));
  return ReturnOp(xla_output, loctx);
}

std::string ConstantPadNd::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", pad=[" << absl::StrJoin(pad_, ", ") << "]"
     << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
