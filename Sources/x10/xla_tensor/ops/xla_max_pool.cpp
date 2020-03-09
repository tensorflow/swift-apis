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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_max_pool.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/pooling.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

// Infers the output shape of the max pooling operation.
xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> kernel_size,
                           std::vector<xla::int64> strides,
                           xla::Padding padding,
                           const xla::TensorFormat& data_format) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return xla::MaxPool(operands[0], kernel_size, strides, padding,
                        data_format);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

XlaMaxPool::XlaMaxPool(const Value& input, std::vector<xla::int64> kernel_size,
                       std::vector<xla::int64> strides, xla::Padding padding,
                       xla::TensorFormat data_format)
    : Node(ir::OpKind(at::aten::xla_max_pool), {input},
           [&]() {
             return NodeOutputShape(input, kernel_size, strides, padding,
                                    data_format);
           },
           /*num_outputs=*/1,
           xla::util::MHash(kernel_size, strides, static_cast<int>(padding),
                            DataFormatToList(data_format))),
      kernel_size_(std::move(kernel_size)),
      strides_(std::move(strides)),
      padding_(padding),
      data_format_(std::move(data_format)) {}

NodePtr XlaMaxPool::Clone(OpList operands) const {
  return MakeNode<XlaMaxPool>(operands.at(0), kernel_size_, strides_, padding_,
                              data_format_);
}

XlaOpVector XlaMaxPool::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      xla::MaxPool(input, kernel_size_, strides_, padding_, data_format_);
  return ReturnOp(output, loctx);
}

std::string XlaMaxPool::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", kernel_size=["
     << absl::StrJoin(kernel_size_, ", ") << "], strides=["
     << absl::StrJoin(strides_, ", ")
     << "], padding=" << static_cast<int>(padding_) << ", data_format=["
     << absl::StrJoin(DataFormatToList(data_format_), "]");
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
