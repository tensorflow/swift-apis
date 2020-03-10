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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/view.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> output_sizes) {
  const xla::Shape& input_shape = input.shape();
  auto info = XlaHelpers::GetDynamicReshapeInfo(input_shape, output_sizes);
  if (info) {
    return std::move(info->output_shape);
  }
  const auto complete_output_sizes =
      GetCompleteShape(output_sizes, input_shape.dimensions());
  return xla::ShapeUtil::MakeShape(input_shape.element_type(),
                                   complete_output_sizes);
}

}  // namespace

View::View(const Value& input, std::vector<xla::int64> output_size)
    : Node(ir::OpKind(at::aten::view), {input},
           NodeOutputShape(input, output_size),
           /*num_outputs=*/1, xla::util::MHash(output_size)),
      output_size_(std::move(output_size)) {}

XlaOpVector View::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildView(input, output_size_);
  return ReturnOp(output, loctx);
}

std::string View::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
