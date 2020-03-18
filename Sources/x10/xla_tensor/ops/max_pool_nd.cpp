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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/max_pool_nd.h"

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
xla::Shape NodeOutputShape(const Value& input, xla::int64 spatial_dim_count,
                           absl::Span<const xla::int64> kernel_size,
                           absl::Span<const xla::int64> stride,
                           absl::Span<const xla::int64> padding,
                           bool ceil_mode) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return BuildMaxPoolNd(operands[0], spatial_dim_count, kernel_size, stride,
                          padding, ceil_mode);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

c10::Symbol MaxPoolNdSymbol(xla::int64 spatial_dim_count) {
  switch (spatial_dim_count) {
    case 1:
      return at::aten::max_pool1d;
    case 2:
      return at::aten::max_pool2d;
    case 3:
      return at::aten::max_pool3d;
    default:
      XLA_ERROR() << "Invalid number of spatial dimensions: "
                  << spatial_dim_count;
  }
}

}  // namespace

MaxPoolNd::MaxPoolNd(const Value& input, xla::int64 spatial_dim_count,
                     std::vector<xla::int64> kernel_size,
                     std::vector<xla::int64> stride,
                     std::vector<xla::int64> padding, bool ceil_mode)
    : Node(ir::OpKind(MaxPoolNdSymbol(spatial_dim_count)), {input},
           [&]() {
             return NodeOutputShape(input, spatial_dim_count, kernel_size,
                                    stride, padding, ceil_mode);
           },
           /*num_outputs=*/1,
           xla::util::MHash(spatial_dim_count, kernel_size, stride, padding,
                            ceil_mode)),
      spatial_dim_count_(spatial_dim_count),
      kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      ceil_mode_(ceil_mode) {}

NodePtr MaxPoolNd::Clone(OpList operands) const {
  return MakeNode<MaxPoolNd>(operands.at(0), spatial_dim_count_, kernel_size_,
                             stride_, padding_, ceil_mode_);
}

XlaOpVector MaxPoolNd::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildMaxPoolNd(input, spatial_dim_count_, kernel_size_,
                                     stride_, padding_, ceil_mode_);
  return ReturnOp(output, loctx);
}

std::string MaxPoolNd::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", spatial_dim_count=" << spatial_dim_count_
     << ", kernel_size=(" << absl::StrJoin(kernel_size_, ", ") << "), stride=("
     << absl::StrJoin(stride_, ", ") << "), padding=("
     << absl::StrJoin(padding_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
