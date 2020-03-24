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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/upsample_nearest2d_backward.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/resize_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {

UpsampleNearestBackward::UpsampleNearestBackward(
    const Value& input, std::vector<xla::int64> output_size,
    std::vector<xla::int64> input_size)
    : Node(ir::OpKind(at::aten::upsample_nearest2d_backward), {input},
           [&]() {
             return resize::GetBackwardOutputShape2d(input.shape(), input_size);
           },
           /*num_outputs=*/1, xla::util::MHash(output_size, input_size)),
      output_size_(std::move(output_size)),
      input_size_(std::move(input_size)) {}

NodePtr UpsampleNearestBackward::Clone(OpList operands) const {
  return MakeNode<UpsampleNearestBackward>(operands.at(0), output_size_,
                                           input_size_);
}

XlaOpVector UpsampleNearestBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = resize::LowerBackward2d(
      "ResizeNearestGrad", input, shape(),
      /*align_corners=*/false, /*half_pixel_centers=*/false);
  return ReturnOp(output, loctx);
}

std::string UpsampleNearestBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << "), input_size=("
     << absl::StrJoin(input_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
