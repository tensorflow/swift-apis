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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/masked_select.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/xla_lower_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input) {
  const xla::Shape& input_shape = input.shape();
  xla::int64 input_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::PrimitiveType size_type = GetShapeDimensionType(/*device=*/nullptr);
  xla::Shape result_shape =
      xla::ShapeUtil::MakeShape(input_shape.element_type(), {input_elements});
  result_shape.set_dynamic_dimension(0, true);
  return xla::ShapeUtil::MakeTupleShape(
      {result_shape, xla::ShapeUtil::MakeShape(size_type, {})});
}

}  // namespace

MaskedSelect::MaskedSelect(const Value& input, const Value& mask)
    : Node(ir::OpKind(at::aten::masked_select), {input, mask},
           NodeOutputShape(input),
           /*num_outputs=*/2) {}

NodePtr MaskedSelect::Clone(OpList operands) const {
  return MakeNode<MaskedSelect>(operands.at(0), operands.at(1));
}

XlaOpVector MaskedSelect::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp mask = loctx->GetOutputOp(operand(1));
  return ReturnOps(BuildMaskedSelect(input, mask), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
