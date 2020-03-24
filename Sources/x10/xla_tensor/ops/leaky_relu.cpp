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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/leaky_relu.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"

namespace swift_xla {
namespace ir {
namespace ops {

LeakyRelu::LeakyRelu(const Value& input, double negative_slope)
    : Node(ir::OpKind(at::aten::leaky_relu), {input}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash(negative_slope)),
      negative_slope_(negative_slope) {}

NodePtr LeakyRelu::Clone(OpList operands) const {
  return MakeNode<LeakyRelu>(operands.at(0), negative_slope_);
}

XlaOpVector LeakyRelu::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildLeakyRelu(input, negative_slope_);
  return ReturnOp(output, loctx);
}

std::string LeakyRelu::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", negative_slope=" << negative_slope_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
