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

#include "xla_tensor/ops/xla_avg_pool_grad.h"

#include "xla_tensor/lowering_context.h"
#include "xla_tensor/ops/infer_output_shape.h"
#include "xla_tensor/pooling.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

// Infers the output shape of the average pooling operation.
xla::Shape NodeOutputShape(
    const Value& out_backprop, absl::Span<const xla::int64> gradients_size,
    absl::Span<const xla::int64> kernel_size,
    absl::Span<const xla::int64> stride,
    absl::Span<const std::pair<xla::int64, xla::int64>> spatial_padding,
    const xla::TensorFormat& data_format, const bool counts_include_padding) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return xla::AvgPoolGrad(operands[0], gradients_size, kernel_size, stride,
                            spatial_padding, data_format,
                            counts_include_padding);
  };
  return InferOutputShape({out_backprop.shape()}, lower_for_shape_fn);
}

}  // namespace

XlaAvgPoolGrad::XlaAvgPoolGrad(
    const Value& out_backprop, std::vector<xla::int64> gradients_size,
    std::vector<xla::int64> kernel_size, std::vector<xla::int64> stride,
    std::vector<std::pair<xla::int64, xla::int64>> spatial_padding,
    const xla::TensorFormat& data_format, const bool counts_include_padding)
    : Node(ir::OpKind(at::aten::xla_avg_pool_grad), {out_backprop},
           [&]() {
             return NodeOutputShape(out_backprop, gradients_size, kernel_size,
                                    stride, spatial_padding, data_format,
                                    counts_include_padding);
           },
           /*num_outputs=*/1,
           xla::util::MHash(gradients_size, kernel_size, stride,
                            PaddingToList(spatial_padding),
                            DataFormatToList(data_format),
                            counts_include_padding)),
      gradients_size_(std::move(gradients_size)),
      kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      spatial_padding_(std::move(spatial_padding)),
      data_format_(data_format),
      counts_include_padding_(counts_include_padding) {}

NodePtr XlaAvgPoolGrad::Clone(OpList operands) const {
  return MakeNode<XlaAvgPoolGrad>(operands.at(0), gradients_size_, kernel_size_,
                                  stride_, spatial_padding_, data_format_,
                                  counts_include_padding_);
}

XlaOpVector XlaAvgPoolGrad::Lower(LoweringContext* loctx) const {
  xla::XlaOp out_backprop = loctx->GetOutputOp(operand(0));
  xla::XlaOp in_backprop =
      xla::AvgPoolGrad(out_backprop, gradients_size_, kernel_size_, stride_,
                       spatial_padding_, data_format_, counts_include_padding_);
  return ReturnOp(in_backprop, loctx);
}

std::string XlaAvgPoolGrad::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", gradients_size=["
     << absl::StrJoin(gradients_size_, ", ") << "], kernel_size=["
     << absl::StrJoin(kernel_size_, ", ") << "], stride=["
     << absl::StrJoin(stride_, ", ") << "], padding=["
     << absl::StrJoin(PaddingToList(spatial_padding_), ", ")
     << "], data_format=["
     << absl::StrJoin(DataFormatToList(data_format_), ", ")
     << "], counts_include_padding=" << counts_include_padding_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
