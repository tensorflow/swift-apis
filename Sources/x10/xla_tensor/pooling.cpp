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

#include "tensorflow/compiler/tf2xla/xla_tensor/pooling.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"

namespace swift_xla {
namespace {

xla::TensorFormat MakeNCHWFormat(xla::int64 spatial_dim_count) {
  return {
      /*batch_dimension=*/0,
      /*feature_dimension=*/1,
      /*spatial_dimensions=*/xla::util::Iota<xla::int64>(spatial_dim_count, 2)};
}

// Holds the attributes common to all pooling operators.
struct PoolingOpAttributes {
  std::vector<xla::int64> kernel_size;
  std::vector<xla::int64> stride;
};

xla::XlaComputation CreateGeComputation(xla::PrimitiveType type) {
  xla::XlaBuilder reduction_builder("xla_ge_computation");
  xla::XlaOp x = xla::Parameter(&reduction_builder, 0,
                                xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y = xla::Parameter(&reduction_builder, 1,
                                xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::Ge(x, y);
  return ConsumeValue(reduction_builder.Build());
}

// Construct the pooling attributes for the given kernel size, stride and
// padding.
PoolingOpAttributes MakePoolingOpAttributes(
    absl::Span<const xla::int64> kernel_size_attr,
    absl::Span<const xla::int64> stride_attr) {
  // Create a NCHW kernel size with 1 for batch size and feature.
  std::vector<xla::int64> kernel_size(2, 1);
  kernel_size.insert(kernel_size.end(), kernel_size_attr.begin(),
                     kernel_size_attr.end());
  // Create a NCHW stride size with 1 for batch size and feature. Same as kernel
  // size if not specified.
  std::vector<xla::int64> stride;
  if (stride_attr.empty()) {
    stride = kernel_size;
  } else {
    stride.resize(2, 1);
    stride.insert(stride.end(), stride_attr.begin(), stride_attr.end());
  }
  return {kernel_size, stride};
}

// Compute the average pool kernel size required for the specified output_size
// from the given input_size, when the stride is the same as the kernel size.
std::vector<xla::int64> AdaptiveAvgPoolKernelSize(
    absl::Span<const xla::int64> input_size,
    absl::Span<const xla::int64> output_size) {
  // Create a NCHW kernel size with 1 for batch size and feature.
  std::vector<xla::int64> kernel_size(2, 1);
  xla::int64 spatial_dim_off = input_size.size() - 2;
  for (int spatial_dim = 0; spatial_dim < 2; ++spatial_dim) {
    XLA_CHECK_EQ(
        input_size[spatial_dim_off + spatial_dim] % output_size[spatial_dim], 0)
        << "Target output size " << output_size[spatial_dim]
        << " doesn't divide the input size "
        << input_size[spatial_dim_off + spatial_dim];
    kernel_size.push_back(input_size[spatial_dim_off + spatial_dim] /
                          output_size[spatial_dim]);
  }
  return kernel_size;
}

struct BatchInput {
  xla::XlaOp batch_input;
  xla::int64 original_rank;
};

// Adds a batch dimension of size 1 if the input tensor doesn't have a batch
// dimension.
BatchInput CreateBatchInput(xla::XlaOp input, xla::int64 spatial_dim_count) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::int64 rank = input_shape.rank();
  XLA_CHECK(rank == spatial_dim_count + 1 || rank == spatial_dim_count + 2)
      << "Input must be a " << spatial_dim_count + 1 << "-D or "
      << spatial_dim_count + 2 << "-D tensor";
  if (rank == spatial_dim_count + 1) {
    return {BuildUnsqueeze(input, 0), rank};
  }
  return {input, rank};
}

xla::XlaOp RemoveTrivialBatch(xla::XlaOp batch, xla::int64 original_rank,
                              xla::int64 spatial_dim_count) {
  if (original_rank == spatial_dim_count + 1) {
    return SqueezeTrivialDimension(batch, 0);
  }
  return batch;
}

// Creates low and high padding specification for the given padding (which is
// symmetric) and ceil mode. Additional high padding could be required when ceil
// mode is set.
std::vector<std::pair<xla::int64, xla::int64>> CeilModePadding(
    absl::Span<const xla::int64> padding, const xla::Shape& input_shape,
    absl::Span<const xla::int64> kernel_size,
    absl::Span<const xla::int64> stride, bool ceil_mode) {
  std::vector<std::pair<xla::int64, xla::int64>> ceil_mode_padding;
  for (int i = 0; i < padding.size(); ++i) {
    xla::int64 left_padding = padding[i];
    xla::int64 input_size = input_shape.dimensions(2 + i);
    xla::int64 output_size_rem =
        (input_size + 2 * left_padding - kernel_size[i]) % stride[i];
    xla::int64 right_padding = left_padding;
    if (ceil_mode && output_size_rem != 0) {
      right_padding += stride[i] - output_size_rem;
    }
    ceil_mode_padding.emplace_back(left_padding, right_padding);
  }
  return ceil_mode_padding;
}

// Creates an XLA padding configuration from a padding attribute value.
xla::PaddingConfig MakeXlaPaddingConfig(
    absl::Span<const xla::int64> padding, const xla::Shape& input_shape,
    absl::Span<const xla::int64> kernel_size,
    absl::Span<const xla::int64> stride, bool ceil_mode) {
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  const auto ceil_mode_padding =
      CeilModePadding(padding, input_shape, kernel_size, stride, ceil_mode);
  for (int i = 0; i < padding.size(); ++i) {
    xla::PaddingConfig::PaddingConfigDimension* dims =
        padding_config.add_dimensions();
    const auto dim_padding = ceil_mode_padding[i];
    dims->set_edge_padding_low(dim_padding.first);
    dims->set_edge_padding_high(dim_padding.second);
  }
  return padding_config;
}

}  // namespace

bool IsSupportedAdaptiveAvgPool2d(absl::Span<const xla::int64> input_size,
                                  absl::Span<const xla::int64> output_size) {
  xla::int64 rank = input_size.size();
  for (int spatial_dim = 0; spatial_dim < 2; ++spatial_dim) {
    if (input_size[rank - 2 + spatial_dim] % output_size[spatial_dim] != 0) {
      return false;
    }
  }
  return true;
}

xla::XlaOp BuildMaxPoolNd(xla::XlaOp input, xla::int64 spatial_dim_count,
                          absl::Span<const xla::int64> kernel_size,
                          absl::Span<const xla::int64> stride,
                          absl::Span<const xla::int64> padding,
                          bool ceil_mode) {
  xla::XlaBuilder* builder = input.builder();
  BatchInput batch_input_info = CreateBatchInput(input, spatial_dim_count);
  const xla::Shape& input_shape =
      XlaHelpers::ShapeOfXlaOp(batch_input_info.batch_input);
  xla::Literal init_value =
      xla::LiteralUtil::MinValue(input_shape.element_type());
  xla::XlaOp xla_init_value = xla::ConstantLiteral(builder, init_value);
  xla::PaddingConfig padding_config = MakeXlaPaddingConfig(
      padding, input_shape, kernel_size, stride, ceil_mode);
  xla::XlaOp padded_input =
      xla::Pad(batch_input_info.batch_input, xla_init_value, padding_config);
  PoolingOpAttributes pooling_op_attributes =
      MakePoolingOpAttributes(/*kernel_size_attr=*/kernel_size,
                              /*stride_attr=*/stride);
  xla::XlaOp batch_result = xla::MaxPool(
      /*operand=*/padded_input,
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*padding=*/xla::Padding::kValid,
      /*data_format=*/MakeNCHWFormat(spatial_dim_count));
  return RemoveTrivialBatch(/*batch=*/batch_result,
                            /*original_rank=*/batch_input_info.original_rank,
                            /*spatial_dim_count=*/spatial_dim_count);
}

xla::XlaOp BuildMaxPoolNdBackward(xla::XlaOp out_backprop, xla::XlaOp input,
                                  xla::int64 spatial_dim_count,
                                  absl::Span<const xla::int64> kernel_size,
                                  absl::Span<const xla::int64> stride,
                                  absl::Span<const xla::int64> padding,
                                  bool ceil_mode) {
  xla::XlaBuilder* builder = out_backprop.builder();
  BatchInput batch_input_info = CreateBatchInput(input, spatial_dim_count);
  const xla::Shape& input_shape =
      XlaHelpers::ShapeOfXlaOp(batch_input_info.batch_input);
  xla::XlaOp init_value =
      XlaHelpers::ScalarValue<float>(0, input_shape.element_type(), builder);
  xla::XlaComputation select = CreateGeComputation(input_shape.element_type());
  xla::XlaComputation scatter =
      XlaHelpers::CreateAddComputation(input_shape.element_type());
  PoolingOpAttributes pooling_op_attributes =
      MakePoolingOpAttributes(/*kernel_size_attr=*/kernel_size,
                              /*stride_attr=*/stride);
  std::vector<std::pair<xla::int64, xla::int64>> window_padding;
  const auto ceil_mode_padding =
      CeilModePadding(padding, input_shape, kernel_size, stride, ceil_mode);
  window_padding.resize(2);
  window_padding.insert(window_padding.end(), ceil_mode_padding.begin(),
                        ceil_mode_padding.end());
  BatchInput batch_out_backprop_info =
      CreateBatchInput(out_backprop, spatial_dim_count);
  xla::XlaOp batch_result = xla::SelectAndScatterWithGeneralPadding(
      /*operand=*/batch_input_info.batch_input,
      /*select=*/select,
      /*window_dimensions=*/pooling_op_attributes.kernel_size,
      /*window_strides=*/pooling_op_attributes.stride,
      /*padding=*/window_padding,
      /*source=*/batch_out_backprop_info.batch_input,
      /*init_value=*/init_value,
      /*scatter=*/scatter);
  return RemoveTrivialBatch(/*batch=*/batch_result,
                            /*original_rank=*/batch_input_info.original_rank,
                            /*spatial_dim_count=*/spatial_dim_count);
}

xla::XlaOp BuildAvgPoolNd(xla::XlaOp input, xla::int64 spatial_dim_count,
                          absl::Span<const xla::int64> kernel_size,
                          absl::Span<const xla::int64> stride,
                          absl::Span<const xla::int64> padding, bool ceil_mode,
                          bool count_include_pad) {
  PoolingOpAttributes pooling_op_attributes =
      MakePoolingOpAttributes(/*kernel_size_attr=*/kernel_size,
                              /*stride_attr=*/stride);
  BatchInput batch_input_info = CreateBatchInput(input, spatial_dim_count);
  const xla::Shape& input_shape =
      XlaHelpers::ShapeOfXlaOp(batch_input_info.batch_input);
  const auto ceil_mode_padding =
      CeilModePadding(padding, input_shape, kernel_size, stride, ceil_mode);
  xla::XlaOp batch_result = xla::AvgPool(
      /*operand=*/batch_input_info.batch_input,
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*padding=*/ceil_mode_padding,
      /*data_format=*/MakeNCHWFormat(spatial_dim_count),
      /*counts_include_padding=*/count_include_pad);
  return RemoveTrivialBatch(/*batch=*/batch_result,
                            /*original_rank=*/batch_input_info.original_rank,
                            /*spatial_dim_count=*/spatial_dim_count);
}

xla::XlaOp BuildAvgPoolNdBackward(xla::XlaOp out_backprop, xla::XlaOp input,
                                  xla::int64 spatial_dim_count,
                                  absl::Span<const xla::int64> kernel_size,
                                  absl::Span<const xla::int64> stride,
                                  absl::Span<const xla::int64> padding,
                                  bool ceil_mode, bool count_include_pad) {
  PoolingOpAttributes pooling_op_attributes =
      MakePoolingOpAttributes(/*kernel_size_attr=*/kernel_size,
                              /*stride_attr=*/stride);
  BatchInput batch_input_info = CreateBatchInput(input, spatial_dim_count);
  const xla::Shape& gradients_shape =
      XlaHelpers::ShapeOfXlaOp(batch_input_info.batch_input);
  const auto ceil_mode_padding =
      CeilModePadding(padding, gradients_shape, kernel_size, stride, ceil_mode);
  BatchInput batch_out_backprop_info =
      CreateBatchInput(out_backprop, spatial_dim_count);
  xla::XlaOp batch_result = xla::AvgPoolGrad(
      /*out_backprop=*/batch_out_backprop_info.batch_input,
      /*gradients_size=*/gradients_shape.dimensions(),
      /*kernel_size=*/pooling_op_attributes.kernel_size,
      /*stride=*/pooling_op_attributes.stride,
      /*spatial_padding=*/ceil_mode_padding,
      /*data_format=*/MakeNCHWFormat(spatial_dim_count),
      /*counts_include_padding=*/count_include_pad);
  return RemoveTrivialBatch(/*batch=*/batch_result,
                            /*original_rank=*/batch_input_info.original_rank,
                            /*spatial_dim_count=*/spatial_dim_count);
}

xla::XlaOp BuildAdaptiveAvgPool2d(xla::XlaOp input,
                                  absl::Span<const xla::int64> output_size) {
  XLA_CHECK_EQ(output_size.size(), 2) << "Invalid output size rank";
  const auto input_size = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK(input_size.size() == 4 || input_size.size() == 3)
      << "Only 4D or 3D tensors supported";
  const auto kernel_size = AdaptiveAvgPoolKernelSize(input_size, output_size);
  std::vector<std::pair<xla::int64, xla::int64>> no_padding(2);
  BatchInput batch_input_info =
      CreateBatchInput(input, /*spatial_dim_count=*/2);
  xla::XlaOp batch_result = xla::AvgPool(
      /*operand=*/batch_input_info.batch_input,
      /*kernel_size=*/kernel_size,
      /*stride=*/kernel_size,
      /*padding=*/no_padding,
      /*data_format=*/MakeNCHWFormat(2),
      /*counts_include_padding=*/false);
  return RemoveTrivialBatch(/*batch=*/batch_result,
                            /*original_rank=*/batch_input_info.original_rank,
                            /*spatial_dim_count=*/2);
}

xla::XlaOp BuildAdaptiveAvgPool2dBackward(xla::XlaOp out_backprop,
                                          xla::XlaOp input) {
  BatchInput batch_out_backprop_info =
      CreateBatchInput(/*input=*/out_backprop, /*spatial_dim_count=*/2);
  const auto out_backprop_size =
      XlaHelpers::SizesOfXlaOp(batch_out_backprop_info.batch_input);
  XLA_CHECK_EQ(out_backprop_size.size(), 4)
      << "Invalid rank of gradient output";
  std::vector<xla::int64> output_size{out_backprop_size[2],
                                      out_backprop_size[3]};
  auto gradients_size = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK(gradients_size.size() == 4 || gradients_size.size() == 3)
      << "Only 4D or 3D tensors supported";
  if (gradients_size.size() == 3) {
    gradients_size.insert(gradients_size.begin(), 1);
  }
  const auto kernel_size =
      AdaptiveAvgPoolKernelSize(gradients_size, output_size);
  std::vector<std::pair<xla::int64, xla::int64>> no_padding(2);
  xla::XlaOp batch_result = xla::AvgPoolGrad(
      /*out_backprop=*/batch_out_backprop_info.batch_input,
      /*gradients_size=*/gradients_size,
      /*kernel_size=*/kernel_size,
      /*stride=*/kernel_size,
      /*spatial_padding=*/no_padding,
      /*data_format=*/MakeNCHWFormat(2),
      /*counts_include_padding=*/false);
  return RemoveTrivialBatch(
      /*batch=*/batch_result,
      /*original_rank=*/batch_out_backprop_info.original_rank,
      /*spatial_dim_count=*/2);
}

std::vector<xla::int64> PaddingToList(
    absl::Span<const std::pair<xla::int64, xla::int64>> padding) {
  std::vector<xla::int64> list_padding;
  for (const auto& dim_padding : padding) {
    list_padding.push_back(dim_padding.first);
    list_padding.push_back(dim_padding.second);
  }
  return list_padding;
}

std::vector<xla::int64> DataFormatToList(const xla::TensorFormat& data_format) {
  std::vector<xla::int64> data_format_list{data_format.batch_dimension(),
                                           data_format.feature_dimension()};
  for (int spatial_dim = 0; spatial_dim < data_format.num_spatial_dims();
       ++spatial_dim) {
    data_format_list.push_back(data_format.spatial_dimension(spatial_dim));
  }
  return data_format_list;
}

}  // namespace swift_xla
