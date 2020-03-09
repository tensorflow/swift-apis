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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_max_pool_grad.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/pooling.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp BuildXlaMaxPoolGrad(xla::XlaOp input, xla::XlaOp out_backprop,
                               absl::Span<const xla::int64> kernel_size,
                               absl::Span<const xla::int64> strides,
                               xla::Padding padding) {
  xla::Shape out_backprop_shape = XlaHelpers::ShapeOfXlaOp(out_backprop);
  xla::PrimitiveType element_type = out_backprop_shape.element_type();
  xla::XlaBuilder* builder = out_backprop.builder();
  xla::XlaOp init_value = XlaHelpers::ScalarValue(0, element_type, builder);
  auto select = xla::CreateScalarGeComputation(element_type, builder);
  auto scatter = xla::CreateScalarAddComputation(element_type, builder);
  return xla::SelectAndScatter(input, select, kernel_size, strides, padding,
                               out_backprop, init_value, scatter);
}

// Infers the output shape of the max pooling gradient operation.
xla::Shape NodeOutputShape(const Value& input, const Value& out_backprop,
                           absl::Span<const xla::int64> kernel_size,
                           std::vector<xla::int64> strides,
                           xla::Padding padding) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2)
        << "Unexpected number of operands: " << operands.size();
    return BuildXlaMaxPoolGrad(operands[0], operands[1], kernel_size, strides,
                               padding);
  };
  return InferOutputShape({input.shape(), out_backprop.shape()},
                          lower_for_shape_fn);
}

}  // namespace

XlaMaxPoolGrad::XlaMaxPoolGrad(const Value& input, const Value& out_backprop,
                               std::vector<xla::int64> kernel_size,
                               std::vector<xla::int64> strides,
                               xla::Padding padding)
    : Node(ir::OpKind(at::aten::xla_max_pool_grad), {input, out_backprop},
           [&]() {
             return NodeOutputShape(input, out_backprop, kernel_size, strides,
                                    padding);
           },
           /*num_outputs=*/1,
           xla::util::MHash(kernel_size, strides, static_cast<int>(padding))),
      kernel_size_(std::move(kernel_size)),
      strides_(std::move(strides)),
      padding_(padding) {}

NodePtr XlaMaxPoolGrad::Clone(OpList operands) const {
  return MakeNode<XlaMaxPoolGrad>(operands.at(0), operands.at(1), kernel_size_,
                                  strides_, padding_);
}

XlaOpVector XlaMaxPoolGrad::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp out_backprop = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildXlaMaxPoolGrad(input, out_backprop, kernel_size_,
                                          strides_, padding_);
  return ReturnOp(output, loctx);
}

std::string XlaMaxPoolGrad::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", kernel_size=["
     << absl::StrJoin(kernel_size_, ", ") << "], strides=["
     << absl::StrJoin(strides_, ", ")
     << "], padding=" << static_cast<int>(padding_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
