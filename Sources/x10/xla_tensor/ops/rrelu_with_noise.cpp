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

#include "xla_tensor/ops/rrelu_with_noise.h"

#include "xla_client/util.h"
#include "xla_tensor/elementwise.h"
#include "xla_tensor/helpers.h"
#include "xla_tensor/lowering_context.h"
#include "xla_tensor/ops/scalar.h"

namespace swift_xla {
namespace ir {
namespace ops {

RreluWithNoise::RreluWithNoise(const Value& input, at::Scalar lower,
                               at::Scalar upper, bool training,
                               xla::uint64 seed)
    : Node(ir::OpKind(at::aten::rrelu_with_noise), {input},
           xla::ShapeUtil::MakeTupleShape({input.shape(), input.shape()}),
           /*num_outputs=*/2,
           xla::util::MHash(ScalarHash(lower), ScalarHash(upper), training,
                            seed)),
      lower_(std::move(lower)),
      upper_(std::move(upper)),
      training_(training),
      seed_(seed) {}

NodePtr RreluWithNoise::Clone(OpList operands) const {
  return MakeNode<RreluWithNoise>(operands.at(0), lower_, upper_, training_,
                                  seed_);
}

XlaOpVector RreluWithNoise::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp rng_seed =
      XlaHelpers::ScalarValue(seed_, xla::PrimitiveType::U64, input.builder());
  return ReturnOps(BuildRrelu(input, lower_, upper_, training_, rng_seed),
                   loctx);
}

std::string RreluWithNoise::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lower=" << lower_ << ", upper=" << upper_
     << ", training=" << training_ << ", seed=" << seed_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
