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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/linear_interpolation.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {

LinearInterpolation::LinearInterpolation(const Value& value,
                                         const Value& new_value, double alpha)
    : Node(xla_moving_average, {value, new_value}, value.shape(),
           /*num_outputs=*/1, xla::util::MHash(alpha)),
      alpha_(alpha) {}

NodePtr LinearInterpolation::Clone(OpList operands) const {
  return MakeNode<LinearInterpolation>(operands.at(0), operands.at(1), alpha_);
}

XlaOpVector LinearInterpolation::Lower(LoweringContext* loctx) const {
  xla::XlaOp value = loctx->GetOutputOp(operand(0));
  xla::XlaOp new_value = loctx->GetOutputOp(operand(1));
  return ReturnOp(XlaHelpers::LinearInterpolation(value, new_value, alpha_),
                  loctx);
}

std::string LinearInterpolation::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", alpha=" << alpha_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
