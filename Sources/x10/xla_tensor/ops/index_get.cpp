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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/index_get.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/xla_lower_util.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& base, const Value& indices,
                           xla::int64 start_dim) {
  auto lower_for_shape_fn =
      [start_dim](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2);
    return CreateIndex(operands[0], operands[1], start_dim);
  };
  return InferOutputShape({base.shape(), indices.shape()}, lower_for_shape_fn);
}

}  // namespace

IndexGet::IndexGet(const ir::Value& base, const ir::Value& indices,
                   xla::int64 start_dim)
    : Node(OpKind(at::aten::index), {base, indices},
           [&]() { return NodeOutputShape(base, indices, start_dim); },
           /*num_outputs=*/1, xla::util::MHash(start_dim)),
      start_dim_(start_dim) {}

std::string IndexGet::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", start_dim=" << start_dim_;
  return ss.str();
}

NodePtr IndexGet::Clone(OpList operands) const {
  return MakeNode<IndexGet>(operands.at(0), operands.at(1), start_dim_);
}

XlaOpVector IndexGet::Lower(LoweringContext* loctx) const {
  xla::XlaOp base = loctx->GetOutputOp(operand(0));
  xla::XlaOp indices = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = CreateIndex(base, indices, start_dim_);
  return ReturnOp(output, loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
