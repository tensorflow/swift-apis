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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/cast.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/convert_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, xla::PrimitiveType type) {
  xla::Shape shape = input.shape();
  shape.set_element_type(type);
  return shape;
}

}  // namespace

Cast::Cast(const Value& input, xla::PrimitiveType type)
    : Node(xla_cast, {input}, NodeOutputShape(input, type),
           /*num_outputs=*/1, xla::util::MHash(static_cast<int>(type))),
      type_(type) {}

Cast::Cast(const Value& input, at::ScalarType dtype)
    : Node(xla_cast, {input},
           NodeOutputShape(input,
                           MakeXlaPrimitiveType(dtype, /*device=*/nullptr)),
           /*num_outputs=*/1, xla::util::MHash(101, static_cast<int>(dtype))),
      type_(MakeXlaPrimitiveType(dtype, /*device=*/nullptr)),
      dtype_(dtype) {}

NodePtr Cast::Clone(OpList operands) const {
  return dtype_ ? MakeNode<Cast>(operands.at(0), *dtype_)
                : MakeNode<Cast>(operands.at(0), type_);
}

XlaOpVector Cast::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::PrimitiveType raw_type =
      dtype_ ? TensorTypeToRawXlaType(*dtype_) : type_;
  xla::XlaOp output =
      ConvertToRaw(input, input_shape.element_type(), type_, raw_type,
                   /*device=*/nullptr);
  return ReturnOp(output, loctx);
}

std::string Cast::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", type=" << xla::primitive_util::LowercasePrimitiveTypeName(type_);
  if (dtype_) {
    ss << ", dtype=" << *dtype_;
  }
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
