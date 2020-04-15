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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_unsorted_segment_sum.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/segment_reduction_ops.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerTfUnsortedSegmentSum(xla::XlaOp data, xla::XlaOp indices,
                                     xla::int64 num_segments) {
  const xla::Shape& data_shape = XlaHelpers::ShapeOfXlaOp(data);
  xla::XlaOp init_value = xla::Zero(data.builder(), data_shape.element_type());
  auto combine = [](xla::XlaOp a, xla::XlaOp b) { return a + b; };
  return UnsortedSegmentReduce(data, indices, init_value, num_segments,
                               combine);
}

xla::Shape NodeOutputShape(const Value& data, const Value& indices,
                           xla::int64 num_segments) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    CHECK_EQ(operands.size(), 2)
        << "Unexpected number of operands: " << operands.size();
    return LowerTfUnsortedSegmentSum(operands[0], operands[1], num_segments);
  };
  return InferOutputShape({data.shape(), indices.shape()}, lower_for_shape_fn);
}

}  // namespace

TfUnsortedSegmentSum::TfUnsortedSegmentSum(const Value& data,
                                           const Value& indices,
                                           xla::int64 num_segments)
    : Node(ir::OpKind(at::aten::tf_unsorted_segment_sum), {data, indices},
           [&]() { return NodeOutputShape(data, indices, num_segments); },
           /*num_outputs=*/1, xla::util::MHash(num_segments)),
      num_segments_(num_segments) {}

NodePtr TfUnsortedSegmentSum::Clone(OpList operands) const {
  return MakeNode<TfUnsortedSegmentSum>(operands.at(0), operands.at(1),
                                        num_segments_);
}

XlaOpVector TfUnsortedSegmentSum::Lower(LoweringContext* loctx) const {
  xla::XlaOp data = loctx->GetOutputOp(operand(0));
  xla::XlaOp indices = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = LowerTfUnsortedSegmentSum(data, indices, num_segments_);
  return ReturnOp(output, loctx);
}

std::string TfUnsortedSegmentSum::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", num_segments=" << num_segments_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
