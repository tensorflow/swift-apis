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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/gather.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, const Value& index,
                           xla::int64 dim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::TorchGather(operands[0], operands[1], dim,
                            IsSparseGather(operands[0], operands[1], dim));
  };
  return InferOutputShape({input.shape(), index.shape()}, lower_for_shape_fn);
}

}  // namespace

Gather::Gather(const Value& input, xla::int64 dim, const Value& index)
    : Node(ir::OpKind(at::aten::gather), {input, index},
           [&]() { return NodeOutputShape(input, index, dim); },
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr Gather::Clone(OpList operands) const {
  return MakeNode<Gather>(operands.at(0), dim_, operands.at(1));
}

XlaOpVector Gather::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  return ReturnOp(
      xla::TorchGather(input, index, dim_, IsSparseGather(input, index, dim_)),
      loctx);
}

std::string Gather::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
