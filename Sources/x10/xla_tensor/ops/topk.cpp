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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/topk.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/xla_lower_util.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, xla::int64 k, xla::int64 dim,
                           bool largest, bool sorted) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(operands[0].builder(),
                      CreateTopK(operands[0], k, dim, largest, sorted));
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

TopK::TopK(const Value& input, xla::int64 k, xla::int64 dim, bool largest,
           bool sorted)
    : Node(ir::OpKind(at::aten::topk), {input},
           [&]() { return NodeOutputShape(input, k, dim, largest, sorted); },
           /*num_outputs=*/2, xla::util::MHash(k, dim, largest, sorted)),
      k_(k),
      dim_(dim),
      largest_(largest),
      sorted_(sorted) {}

NodePtr TopK::Clone(OpList operands) const {
  return MakeNode<TopK>(operands.at(0), k_, dim_, largest_, sorted_);
}

XlaOpVector TopK::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(CreateTopK(input, k_, dim_, largest_, sorted_), loctx);
}

std::string TopK::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", k=" << k_ << ", dim=" << dim_
     << ", largest=" << largest_ << ", sorted=" << sorted_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
