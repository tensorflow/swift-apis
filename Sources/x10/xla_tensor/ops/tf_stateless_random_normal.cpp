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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_stateless_random_normal.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"

namespace swift_xla {
namespace ir {
namespace ops {

TfStatelessRandomNormal::TfStatelessRandomNormal(xla::Shape shape,
                                                 const Value& seeds,
                                                 BitGeneratorType generator)
    : Node(ir::OpKind(at::aten::tf_stateless_random_normal), {seeds}, shape,
           /*num_outputs=*/1,
           xla::util::MHash(shape.ToString(), static_cast<int>(generator))),
      generator_(generator) {}

NodePtr TfStatelessRandomNormal::Clone(OpList operands) const {
  return MakeNode<TfStatelessRandomNormal>(shape(), operands.at(0),
                                           generator());
}

XlaOpVector TfStatelessRandomNormal::Lower(LoweringContext* loctx) const {
  xla::XlaOp seeds = loctx->GetOutputOp(operand(0));
  xla::XlaOp seed0 = xla::Reshape(xla::Slice(seeds, {0}, {1}, {1}), {});
  xla::XlaOp seed1 = xla::Reshape(xla::Slice(seeds, {1}, {2}, {1}), {});
  xla::XlaOp initial_state =
      xla::ConstantR0WithType(loctx->builder(), xla::U64, 0);
  xla::XlaOp key =
      ConvertElementType(seed0, xla::U64) |
      ShiftLeft(ConvertElementType(seed1, xla::U64),
                ConstantR0WithType(loctx->builder(), xla::U64, 32));
  xla::XlaOp normal =
      xla::NormalFloatingPointDistribution(
          key, initial_state, GetBitGenerator(generator()), shape())
          .value;
  return ReturnOp(normal, loctx);
}

std::string TfStatelessRandomNormal::ToString() const {
  std::stringstream ss;
  ss << Node::ToString();
  switch (generator()) {
    case BitGeneratorType::PHILOX:
      ss << ", generator=PHILOX";
      break;
    case BitGeneratorType::THREE_FRY:
      ss << ", generator=THREE_FRY";
      break;
  }
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
