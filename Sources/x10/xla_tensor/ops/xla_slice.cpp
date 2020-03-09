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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_slice.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& operand,
                           absl::Span<const xla::int64> start_indices,
                           absl::Span<const xla::int64> limit_indices,
                           absl::Span<const xla::int64> strides) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return xla::Slice(operands[0], start_indices, limit_indices, strides);
  };
  return InferOutputShape({operand.shape()}, lower_for_shape_fn);
}

}  // namespace

XlaSlice::XlaSlice(const Value& operand, std::vector<xla::int64> start_indices,
                   std::vector<xla::int64> limit_indices,
                   std::vector<xla::int64> strides)
    : Node(ir::OpKind(at::aten::xla_slice), {operand},
           [&]() {
             return NodeOutputShape(operand, start_indices, limit_indices,
                                    strides);
           },
           /*num_outputs=*/1,
           xla::util::MHash(start_indices, limit_indices, strides)),
      start_indices_(std::move(start_indices)),
      limit_indices_(std::move(limit_indices)),
      strides_(std::move(strides)) {}

NodePtr XlaSlice::Clone(OpList operands) const {
  return MakeNode<XlaSlice>(operands.at(0), start_indices_, limit_indices_,
                            strides_);
}

XlaOpVector XlaSlice::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      xla::Slice(input, start_indices_, limit_indices_, strides_);
  return ReturnOp(output, loctx);
}

std::string XlaSlice::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", start_indices=["
     << absl::StrJoin(start_indices_, ", ") << "], limit_indices=["
     << absl::StrJoin(limit_indices_, ", ") << "], strides=["
     << absl::StrJoin(strides_, ", ") << "]";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
