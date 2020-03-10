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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/generic_slice.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> base_indices,
                           absl::Span<const xla::int64> sizes) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildSlice(operands[0], base_indices, sizes);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

GenericSlice::GenericSlice(const Value& input,
                           absl::Span<const xla::int64> base_indices,
                           absl::Span<const xla::int64> sizes)
    : Node(xla_generic_slice, {input},
           [&]() { return NodeOutputShape(input, base_indices, sizes); },
           /*num_outputs=*/1, xla::util::MHash(base_indices, sizes)),
      base_indices_(base_indices.begin(), base_indices.end()),
      sizes_(sizes.begin(), sizes.end()) {}

NodePtr GenericSlice::Clone(OpList operands) const {
  return MakeNode<GenericSlice>(operands.at(0), base_indices_, sizes_);
}

XlaOpVector GenericSlice::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildSlice(input, base_indices_, sizes_);
  return ReturnOp(output, loctx);
}

std::string GenericSlice::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", base_indices=("
     << absl::StrJoin(base_indices_, ", ") << "), sizes=("
     << absl::StrJoin(sizes_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
