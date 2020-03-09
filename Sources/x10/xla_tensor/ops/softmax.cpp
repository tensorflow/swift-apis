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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/softmax.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/convert_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/softmax_builder.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerSoftmax(xla::XlaOp input, xla::int64 dim,
                        const c10::optional<at::ScalarType>& dtype) {
  xla::XlaOp result = BuildSoftmax(input, dim);
  return CastToScalarType(result, dtype);
}

xla::Shape NodeOutputShape(const Value& input,
                           const c10::optional<at::ScalarType>& dtype) {
  if (dtype) {
    return xla::ShapeUtil::ChangeElementType(
        input.shape(), MakeXlaPrimitiveType(*dtype, /*device=*/nullptr));
  }
  return input.shape();
}

}  // namespace

Softmax::Softmax(const Value& input, xla::int64 dim,
                 c10::optional<at::ScalarType> dtype)
    : Node(ir::OpKind(at::aten::softmax), {input},
           [&]() { return NodeOutputShape(input, dtype); },
           /*num_outputs=*/1,
           xla::util::MHash(dim, OptionalOr<int>(dtype, -1))),
      dim_(dim),
      dtype_(dtype) {}

NodePtr Softmax::Clone(OpList operands) const {
  return MakeNode<Softmax>(operands.at(0), dim_, dtype_);
}

XlaOpVector Softmax::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerSoftmax(input, dim_, dtype_), loctx);
}

std::string Softmax::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_
     << ", dtype=" << OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
