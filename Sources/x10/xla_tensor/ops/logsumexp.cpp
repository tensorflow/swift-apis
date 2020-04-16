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

#include "xla_tensor/ops/logsumexp.h"

#include "absl/strings/str_join.h"
#include "xla_client/util.h"
#include "xla_tensor/helpers.h"
#include "xla_tensor/lowering_context.h"
#include "xla_tensor/ops/infer_output_shape.h"
#include "xla_tensor/reduction.h"
#include "xla_tensor/tensor_util.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           std::vector<xla::int64>& dimensions,
                           bool keep_reduced_dimensions) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildLogsumexp(operands[0], dimensions, keep_reduced_dimensions);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Logsumexp::Logsumexp(const Value& input, std::vector<xla::int64> dimensions,
                     bool keep_reduced_dimensions)
    : Node(ir::OpKind(at::aten::logsumexp), {input},
           [&]() {
             return NodeOutputShape(input, dimensions, keep_reduced_dimensions);
           },
           /*num_outputs=*/1,
           xla::util::MHash(dimensions, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions) {}

NodePtr Logsumexp::Clone(OpList operands) const {
  return MakeNode<Logsumexp>(operands.at(0), dimensions_,
                             keep_reduced_dimensions_);
}

XlaOpVector Logsumexp::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildLogsumexp(input, dimensions_, keep_reduced_dimensions_),
                  loctx);
}

std::string Logsumexp::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=(" << absl::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
