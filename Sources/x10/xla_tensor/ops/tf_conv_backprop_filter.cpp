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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_conv_backprop_filter.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_create_conv_attrs.h"
#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp BuildTfConvBackpropFilter(
    xla::XlaOp input, absl::Span<const xla::int64> filter_sizes,
    xla::XlaOp out_backprop, bool depthwise,
    absl::Span<const xla::int64> strides, tensorflow::Padding padding,
    absl::Span<const xla::int64> explicit_paddings,
    tensorflow::TensorFormat data_format,
    absl::Span<const xla::int64> dilations) {
  int num_spatial_dims = filter_sizes.size() - 2;
  tensorflow::ConvOpAttrs attrs =
      CreateConvOpAttrs(num_spatial_dims, depthwise, strides, padding,
                        explicit_paddings, data_format, dilations);
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(XlaHelpers::mat_mul_precision());
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return ConsumeValue(tensorflow::MakeXlaBackpropFilterConvOp(
      /*type_string=*/"TfConvBackpropFilter", /*activations=*/input,
      /*filter_shape=*/
      xla::ShapeUtil::MakeShape(input_shape.element_type(), filter_sizes),
      /*gradients=*/out_backprop,
      /*attrs=*/attrs, /*precision_config=*/&precision_config));
}

// Infers the output shape of the convolution backprop filter operation.
xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> filter_sizes,
                           const Value& out_backprop, bool depthwise,
                           absl::Span<const xla::int64> strides,
                           tensorflow::Padding padding,
                           absl::Span<const xla::int64> explicit_paddings,
                           tensorflow::TensorFormat data_format,
                           absl::Span<const xla::int64> dilations) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    CHECK_EQ(operands.size(), 2)
        << "Unexpected number of operands: " << operands.size();
    return BuildTfConvBackpropFilter(/*input=*/operands[0],
                                     /*filter_sizes=*/filter_sizes,
                                     /*out_backprop=*/operands[1],
                                     /*depthwise=*/depthwise,
                                     /*strides=*/strides, /*padding=*/padding,
                                     /*explicit_paddings=*/explicit_paddings,
                                     /*data_format=*/data_format,
                                     /*dilations=*/dilations);
  };
  return InferOutputShape({input.shape(), out_backprop.shape()},
                          lower_for_shape_fn);
}

}  // namespace

TfConvBackpropFilter::TfConvBackpropFilter(
    const Value& input, std::vector<xla::int64> filter_sizes,
    const Value& out_backprop, bool depthwise, std::vector<xla::int64> strides,
    tensorflow::Padding padding, std::vector<xla::int64> explicit_paddings,
    tensorflow::TensorFormat data_format, std::vector<xla::int64> dilations)
    : Node(ir::OpKind(at::aten::tf_conv_backprop_filter), {input, out_backprop},
           [&]() {
             return NodeOutputShape(input, filter_sizes, out_backprop,
                                    depthwise, strides, padding,
                                    explicit_paddings, data_format, dilations);
           },
           /*num_outputs=*/1,
           xla::util::MHash(filter_sizes, depthwise, strides,
                            static_cast<int>(padding), explicit_paddings,
                            static_cast<int>(data_format), dilations)),
      filter_sizes_(std::move(filter_sizes)),
      depthwise_(depthwise),
      strides_(std::move(strides)),
      padding_(padding),
      explicit_paddings_(std::move(explicit_paddings)),
      data_format_(data_format),
      dilations_(std::move(dilations)) {}

NodePtr TfConvBackpropFilter::Clone(OpList operands) const {
  return MakeNode<TfConvBackpropFilter>(
      /*input=*/operands.at(0), /*filter_sizes=*/filter_sizes_,
      /*out_backprop=*/operands.at(1),
      /*depthwise=*/depthwise_,
      /*strides=*/strides_,
      /*padding=*/padding_, /*explicit_paddings=*/explicit_paddings_,
      /*data_format=*/data_format_, /*dilations=*/dilations_);
}

XlaOpVector TfConvBackpropFilter::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp out_backprop = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildTfConvBackpropFilter(
      /*input=*/input, /*filter_sizes=*/filter_sizes_,
      /*out_backprop=*/out_backprop,
      /*depthwise=*/depthwise_,
      /*strides=*/strides_,
      /*padding=*/padding_, /*explicit_paddings=*/explicit_paddings_,
      /*data_format=*/data_format_, /*dilations=*/dilations_);
  return ReturnOp(output, loctx);
}

std::string TfConvBackpropFilter::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", filter_sizes=["
     << absl::StrJoin(filter_sizes_, ", ") << "], depthwise=" << depthwise_
     << ", strides=[" << absl::StrJoin(strides_, ", ")
     << "], padding=" << padding_ << ", explicit_paddings=["
     << absl::StrJoin(explicit_paddings_, ", ")
     << "], data_format=" << data_format_ << ", dilations=["
     << absl::StrJoin(dilations_, ", ") << "]";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
