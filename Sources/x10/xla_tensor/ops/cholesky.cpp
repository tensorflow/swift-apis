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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/cholesky.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace swift_xla {
namespace ir {
namespace ops {

Cholesky::Cholesky(const Value& input, bool lower)
    : Node(ir::OpKind(at::aten::cholesky), {input}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash(lower)),
      lower_(lower) {}

NodePtr Cholesky::Clone(OpList operands) const {
  return MakeNode<Cholesky>(operands.at(0), lower_);
}

XlaOpVector Cholesky::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      xla::Triangle(xla::Cholesky(input, /*lower=*/lower_), /*lower=*/lower_);
  return ReturnOp(output, loctx);
}

std::string Cholesky::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lower=" << lower_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
