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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/upsample_bilinear2d_backward.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/resize_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {

UpsampleBilinearBackward::UpsampleBilinearBackward(
    const Value& input, std::vector<xla::int64> output_size,
    std::vector<xla::int64> input_size, bool align_corners)
    : Node(ir::OpKind(at::aten::upsample_bilinear2d_backward), {input},
           [&]() {
             return resize::GetBackwardOutputShape2d(input.shape(), input_size);
           },
           /*num_outputs=*/1,
           xla::util::MHash(output_size, input_size, align_corners)),
      output_size_(std::move(output_size)),
      input_size_(std::move(input_size)),
      align_corners_(align_corners) {}

NodePtr UpsampleBilinearBackward::Clone(OpList operands) const {
  return MakeNode<UpsampleBilinearBackward>(operands.at(0), output_size_,
                                            input_size_, align_corners_);
}

XlaOpVector UpsampleBilinearBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = resize::LowerBackward2d(
      "ResizeBilinearGrad", input, shape(), align_corners_,
      /*half_pixel_centers=*/!align_corners_);
  return ReturnOp(output, loctx);
}

std::string UpsampleBilinearBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << "), input_size=("
     << absl::StrJoin(input_size_, ", ")
     << "), align_corners=" << align_corners_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
