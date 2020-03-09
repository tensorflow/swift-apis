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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/as_strided_view_update.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerAsStridedViewUpdate(xla::XlaOp target, xla::XlaOp input,
                                    absl::Span<const xla::int64> size,
                                    xla::int64 storage_offset) {
  xla::int64 input_element_count =
      xla::ShapeUtil::ElementsIn(XlaHelpers::ShapeOfXlaOp(input));
  xla::int64 slice_size = xla::util::Multiply<xla::int64>(size);
  if (input_element_count == slice_size) {
    XLA_CHECK_EQ(storage_offset, 0);
    return XlaHelpers::DynamicReshape(input, size);
  }
  xla::XlaOp r1_target = XlaHelpers::DynamicReshape(target, {slice_size});
  xla::XlaOp r1_input =
      XlaHelpers::DynamicReshape(input, {input_element_count});
  xla::XlaOp r1_result = BuildUnselect(r1_target, r1_input, 0, storage_offset,
                                       storage_offset + input_element_count, 1);
  return xla::Reshape(r1_result, size);
}

xla::Shape NodeOutputShape(const Value& target, const Value& input,
                           absl::Span<const xla::int64> size,
                           xla::int64 storage_offset) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerAsStridedViewUpdate(operands[0], operands[1], size,
                                    storage_offset);
  };
  return InferOutputShape({target.shape(), input.shape()}, lower_for_shape_fn);
}

}  // namespace

AsStridedViewUpdate::AsStridedViewUpdate(const Value& target,
                                         const Value& input,
                                         std::vector<xla::int64> size,
                                         xla::int64 storage_offset)
    : Node(xla_as_strided_view_update, {target, input},
           [&]() {
             return NodeOutputShape(target, input, size, storage_offset);
           },
           /*num_outputs=*/1, xla::util::MHash(size, storage_offset)),
      size_(std::move(size)),
      storage_offset_(storage_offset) {}

std::string AsStridedViewUpdate::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=[" << absl::StrJoin(size_, ", ")
     << "], storage_offset=" << storage_offset_;
  return ss.str();
}

NodePtr AsStridedViewUpdate::Clone(OpList operands) const {
  return MakeNode<AsStridedViewUpdate>(operands.at(0), operands.at(1), size_,
                                       storage_offset_);
}

XlaOpVector AsStridedViewUpdate::Lower(LoweringContext* loctx) const {
  xla::XlaOp target = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  return ReturnOp(
      LowerAsStridedViewUpdate(target, input, size_, storage_offset_), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
