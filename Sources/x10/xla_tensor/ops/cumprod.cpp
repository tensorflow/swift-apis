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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/cumprod.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/convert_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerCumProd(xla::XlaOp input, xla::int64 dim,
                        c10::optional<at::ScalarType> dtype, bool exclusive,
                        bool reverse) {
  xla::XlaOp casted_input = CastToScalarType(input, dtype);
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(casted_input);
  xla::XlaOp init = XlaHelpers::ScalarValue<float>(
      1, input_shape.element_type(), casted_input.builder());
  xla::XlaComputation reducer =
      XlaHelpers::CreateMulComputation(input_shape.element_type());
  return BuildCumulativeComputation(casted_input, dim, reducer, init, exclusive,
                                    reverse);
}

xla::Shape NodeOutputShape(const Value& input,
                           c10::optional<at::ScalarType> dtype) {
  if (dtype) {
    return xla::ShapeUtil::ChangeElementType(
        input.shape(), MakeXlaPrimitiveType(*dtype, /*device=*/nullptr));
  }
  return input.shape();
}

}  // namespace

CumProd::CumProd(const Value& input, xla::int64 dim,
                 c10::optional<at::ScalarType> dtype, bool exclusive,
                 bool reverse)
    : Node(ir::OpKind(at::aten::cumprod), {input},
           [&]() { return NodeOutputShape(input, dtype); },
           /*num_outputs=*/1,
           xla::util::MHash(dim, OptionalOr<int>(dtype, -1), exclusive,
                            reverse)),
      dim_(dim),
      dtype_(dtype),
      exclusive_(exclusive),
      reverse_(reverse) {}

NodePtr CumProd::Clone(OpList operands) const {
  return MakeNode<CumProd>(operands.at(0), dim_, dtype_, exclusive_, reverse_);
}

XlaOpVector CumProd::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerCumProd(input, dim_, dtype_, exclusive_, reverse_),
                  loctx);
}

std::string CumProd::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_
     << ", dtype=" << OptionalOr<int>(dtype_, -1)
     << ", exclusive=" << exclusive_ << ", reverse=" << reverse_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
