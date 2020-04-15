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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/select.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {

Select::Select(const Value& input, xla::int64 dim, xla::int64 start,
               xla::int64 end, xla::int64 stride)
    : Node(xla_select, {input},
           [&]() {
             return MakeSelectShape(input.shape(), dim, start, end, stride);
           },
           /*num_outputs=*/1, xla::util::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

NodePtr Select::Clone(OpList operands) const {
  return MakeNode<Select>(operands.at(0), dim_, start_, end_, stride_);
}

XlaOpVector Select::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = xla::SliceInDim(input, start_, end_,
                                      GetStride(start_, end_, stride_), dim_);
  return ReturnOp(output, loctx);
}

std::string Select::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

xla::Shape Select::MakeSelectShape(const xla::Shape& shape, xla::int64 dim,
                                   xla::int64 start, xla::int64 end,
                                   xla::int64 stride) {
  xla::int64 effective_stride = GetStride(start, end, stride);
  xla::Shape select_shape(shape);
  select_shape.set_dimensions(
      dim, (end - start + effective_stride - 1) / effective_stride);
  return select_shape;
}

xla::int64 Select::GetStride(xla::int64 start, xla::int64 end,
                             xla::int64 stride) {
  if (stride == 0) {
    XLA_CHECK_EQ(start, end);
    stride = 1;
  }
  return stride;
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
