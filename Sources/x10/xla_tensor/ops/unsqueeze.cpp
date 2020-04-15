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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/unsqueeze.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, int dim) {
  const xla::Shape& shape = input.shape();
  auto dimensions = BuildUnsqueezeDimensions(shape.dimensions(), dim);
  return xla::ShapeUtil::MakeShape(shape.element_type(), dimensions);
}

}  // namespace

Unsqueeze::Unsqueeze(const Value& input, int dim)
    : Node(ir::OpKind(at::aten::unsqueeze), {input},
           [&]() { return NodeOutputShape(input, dim); },
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr Unsqueeze::Clone(OpList operands) const {
  return MakeNode<Unsqueeze>(operands.at(0), dim_);
}

XlaOpVector Unsqueeze::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildUnsqueeze(input, dim_);
  return ReturnOp(output, loctx);
}

std::string Unsqueeze::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
