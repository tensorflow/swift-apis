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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/rrelu_with_noise.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/scalar.h"

namespace swift_xla {
namespace ir {
namespace ops {

RreluWithNoise::RreluWithNoise(const Value& input, const Value& seed,
                               at::Scalar lower, at::Scalar upper,
                               bool training)
    : Node(ir::OpKind(at::aten::rrelu_with_noise), {input, seed},
           xla::ShapeUtil::MakeTupleShape({input.shape(), input.shape()}),
           /*num_outputs=*/2,
           xla::util::MHash(ScalarHash(lower), ScalarHash(upper), training)),
      lower_(std::move(lower)),
      upper_(std::move(upper)),
      training_(training) {}

NodePtr RreluWithNoise::Clone(OpList operands) const {
  return MakeNode<RreluWithNoise>(operands.at(0), operands.at(1), lower_,
                                  upper_, training_);
}

XlaOpVector RreluWithNoise::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp rng_seed = loctx->GetOutputOp(operand(1));
  return ReturnOps(BuildRrelu(input, lower_, upper_, training_, rng_seed),
                   loctx);
}

std::string RreluWithNoise::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lower=" << lower_ << ", upper=" << upper_
     << ", training=" << training_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
