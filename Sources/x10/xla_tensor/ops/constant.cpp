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

#include "xla_tensor/ops/constant.h"

#include <algorithm>
#include <sstream>

#include "xla_tensor/lowering_context.h"

namespace swift_xla {
namespace ir {
namespace ops {

Constant::Constant(xla::Literal value)
    : Node(OpKind(at::prim::Constant), value.shape(), /*num_outputs=*/1,
           value.Hash()),
      value_(std::move(value)) {}

std::string Constant::ToString() const {
  // The Literal to string conversion produces \n separated content, which we do
  // not want. It can also produce giant strings, but that's a different issue.
  std::string value_as_string = value_.ToStringWithoutShape();
  std::replace(value_as_string.begin(), value_as_string.end(), '\n', ';');
  std::stringstream ss;
  ss << Node::ToString() << ", value=" << value_as_string;
  return ss.str();
}

NodePtr Constant::Clone(OpList operands) const {
  return MakeNode<Constant>(value_.Clone());
}

XlaOpVector Constant::Lower(LoweringContext* loctx) const {
  return ReturnOp(xla::ConstantLiteral(loctx->builder(), value_), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
