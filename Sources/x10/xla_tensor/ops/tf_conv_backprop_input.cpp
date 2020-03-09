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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_conv_backprop_input.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_create_conv_attrs.h"
#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp BuildTfConvBackpropInput(
    absl::Span<const xla::int64> input_sizes, xla::XlaOp filter,
    xla::XlaOp out_backprop, bool depthwise,
    absl::Span<const xla::int64> strides, tensorflow::Padding padding,
    absl::Span<const xla::int64> explicit_paddings,
    tensorflow::TensorFormat data_format,
    absl::Span<const xla::int64> dilations) {
  int num_spatial_dims = input_sizes.size() - 2;
  tensorflow::ConvOpAttrs attrs =
      CreateConvOpAttrs(num_spatial_dims, depthwise, strides, padding,
                        explicit_paddings, data_format, dilations);
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(XlaHelpers::mat_mul_precision());
  xla::Shape filter_shape = XlaHelpers::ShapeOfXlaOp(filter);
  return ConsumeValue(tensorflow::MakeXlaBackpropInputConvOp(
      /*type_string=*/"TfConvBackpropInput",
      /*input_shape=*/
      xla::ShapeUtil::MakeShape(filter_shape.element_type(), input_sizes),
      /*filter=*/filter,
      /*out_backprop=*/out_backprop,
      /*attrs=*/attrs, /*precision_config=*/&precision_config));
}

// Infers the output shape of the convolution backprop input operation.
xla::Shape NodeOutputShape(absl::Span<const xla::int64> input_sizes,
                           const Value& filter, const Value& out_backprop,
                           bool depthwise, absl::Span<const xla::int64> strides,
                           tensorflow::Padding padding,
                           absl::Span<const xla::int64> explicit_paddings,
                           tensorflow::TensorFormat data_format,
                           absl::Span<const xla::int64> dilations) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    CHECK_EQ(operands.size(), 2)
        << "Unexpected number of operands: " << operands.size();
    return BuildTfConvBackpropInput(
        /*input_sizes=*/input_sizes,
        /*filter=*/operands[0],
        /*out_backprop=*/operands[1],
        /*depthwise=*/depthwise,
        /*strides=*/strides, /*padding=*/padding,
        /*explicit_paddings=*/explicit_paddings,
        /*data_format=*/data_format,
        /*dilations=*/dilations);
  };
  return InferOutputShape({filter.shape(), out_backprop.shape()},
                          lower_for_shape_fn);
}

}  // namespace

TfConvBackpropInput::TfConvBackpropInput(
    std::vector<xla::int64> input_sizes, const Value& filter,
    const Value& out_backprop, bool depthwise, std::vector<xla::int64> strides,
    tensorflow::Padding padding, std::vector<xla::int64> explicit_paddings,
    tensorflow::TensorFormat data_format, std::vector<xla::int64> dilations)
    : Node(ir::OpKind(at::aten::tf_conv_backprop_input), {filter, out_backprop},
           [&]() {
             return NodeOutputShape(input_sizes, filter, out_backprop,
                                    depthwise, strides, padding,
                                    explicit_paddings, data_format, dilations);
           },
           /*num_outputs=*/1,
           xla::util::MHash(input_sizes, depthwise, strides,
                            static_cast<int>(padding), explicit_paddings,
                            static_cast<int>(data_format), dilations)),
      input_sizes_(std::move(input_sizes)),
      depthwise_(depthwise),
      strides_(std::move(strides)),
      padding_(padding),
      explicit_paddings_(std::move(explicit_paddings)),
      data_format_(data_format),
      dilations_(std::move(dilations)) {}

NodePtr TfConvBackpropInput::Clone(OpList operands) const {
  return MakeNode<TfConvBackpropInput>(
      /*input_sizes=*/input_sizes_, /*filter=*/operands.at(0),
      /*out_backprop=*/operands.at(1), /*depthwise=*/depthwise_,
      /*strides=*/strides_,
      /*padding=*/padding_, /*explicit_paddings=*/explicit_paddings_,
      /*data_format=*/data_format_, /*dilations=*/dilations_);
}

XlaOpVector TfConvBackpropInput::Lower(LoweringContext* loctx) const {
  xla::XlaOp filter = loctx->GetOutputOp(operand(0));
  xla::XlaOp out_backprop = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildTfConvBackpropInput(
      /*input_sizes=*/input_sizes_, /*filter=*/filter,
      /*out_backprop=*/out_backprop,
      /*depthwise=*/depthwise_,
      /*strides=*/strides_,
      /*padding=*/padding_, /*explicit_paddings=*/explicit_paddings_,
      /*data_format=*/data_format_, /*dilations=*/dilations_);
  return ReturnOp(output, loctx);
}

std::string TfConvBackpropInput::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", input_sizes_=["
     << absl::StrJoin(input_sizes_, ", ") << "], depthwise_=" << depthwise_
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
