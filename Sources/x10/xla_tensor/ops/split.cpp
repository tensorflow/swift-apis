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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/split.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           const std::vector<xla::int64>& split_sizes,
                           xla::int64 dim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(operands[0].builder(),
                      BuildSplit(operands[0], split_sizes, dim));
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Split::Split(const Value& input, std::vector<xla::int64> split_sizes,
             xla::int64 dim)
    : Node(ir::OpKind(at::aten::split), {input},
           [&]() { return NodeOutputShape(input, split_sizes, dim); },
           ComputeSplitCount(input.shape().dimensions(dim), split_sizes),
           xla::util::MHash(split_sizes, dim)),
      split_sizes_(std::move(split_sizes)),
      dim_(dim) {}

NodePtr Split::Clone(OpList operands) const {
  return MakeNode<Split>(operands.at(0), split_sizes_, dim_);
}

XlaOpVector Split::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  const auto outputs = BuildSplit(input, split_sizes_, dim_);
  return ReturnOps(outputs, loctx);
}

std::string Split::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", split_sizes=("
     << absl::StrJoin(split_sizes_, ", ") << "), dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
