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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/std.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           std::vector<xla::int64>& dimensions,
                           bool keep_reduced_dimensions, bool unbiased) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildStdDeviation(operands[0], dimensions, keep_reduced_dimensions,
                             unbiased);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Std::Std(const Value& input, std::vector<xla::int64> dimensions,
         bool keep_reduced_dimensions, bool unbiased)
    : Node(ir::OpKind(at::aten::std), {input},
           [&]() {
             return NodeOutputShape(input, dimensions, keep_reduced_dimensions,
                                    unbiased);
           },
           /*num_outputs=*/1,
           xla::util::MHash(dimensions, keep_reduced_dimensions, unbiased)),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions),
      unbiased_(unbiased) {}

NodePtr Std::Clone(OpList operands) const {
  return MakeNode<Std>(operands.at(0), dimensions_, keep_reduced_dimensions_,
                       unbiased_);
}

XlaOpVector Std::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildStdDeviation(input, dimensions_,
                                    keep_reduced_dimensions_, unbiased_),
                  loctx);
}

std::string Std::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=(" << absl::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", unbiased=" << unbiased_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
