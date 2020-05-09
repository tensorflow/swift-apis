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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/scatter_add.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/convert_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/xla_lower_util.h"

namespace swift_xla {
namespace ir {
namespace ops {

ScatterAdd::ScatterAdd(const Value& input, const Value& index, const Value& src,
                       xla::int64 dim)
    : Node(ir::OpKind(at::aten::scatter_add), {input, index, src},
           input.shape(),
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr ScatterAdd::Clone(OpList operands) const {
  return MakeNode<ScatterAdd>(operands.at(0), operands.at(1), operands.at(2),
                              dim_);
}

XlaOpVector ScatterAdd::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  xla::XlaOp src = loctx->GetOutputOp(operand(2));
  return ReturnOp(CreateScatter(loctx->device(), input, index, src, dim_,
                                NumericAddCombiner()),
                  loctx);
}

std::string ScatterAdd::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
