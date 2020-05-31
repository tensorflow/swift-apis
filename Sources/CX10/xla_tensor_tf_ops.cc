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

#if defined(_WIN32)
#define XLA_API __declspec(dllexport)
#else
#define XLA_API __attribute__((__visibility__("default"))) 
#endif

#include "xla_tensor_tf_ops.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor.h"
#include "tensorflow/compiler/xla/client/lib/pooling.h"

using swift_xla::XlaHelpers;
using swift_xla::XLATensor;

namespace {

// Converts the tensor data format to the one required by the XLA pooling
// library.
xla::TensorFormat XlaTensorFormat(tensorflow::TensorFormat data_format,
                                  int num_spatial_dims) {
  int num_dims = num_spatial_dims + 2;
  int batch_dimension = GetTensorBatchDimIndex(num_dims, data_format);
  int feature_dimension = GetTensorFeatureDimIndex(num_dims, data_format);
  absl::InlinedVector<xla::int64, 4> spatial_dimensions(num_spatial_dims);
  for (int spatial_dim = 0; spatial_dim < num_spatial_dims; ++spatial_dim) {
    spatial_dimensions[spatial_dim] =
        GetTensorSpatialDimIndex(num_dims, data_format, spatial_dim);
  }
  return xla::TensorFormat(/*batch_dimension=*/batch_dimension,
                           /*feature_dimension=*/feature_dimension,
                           /*spatial_dimensions=*/spatial_dimensions);
}

xla::Padding ToXLAPadding(TFPadding padding) {
  switch (padding) {
    case TFPadding_SAME: {
      return xla::Padding::kSame;
    }
    case TFPadding_VALID: {
      return xla::Padding::kValid;
    }
    default: {
      LOG(FATAL) << "Invalid padding: " << padding;
    }
  }
}

at::ScalarType SumAccumulationType(at::ScalarType dtype) {
  // Upcast 16 bit sum reductions to 32 bit to reduce the precision loss from
  // repeated floating point additions.
  if (dtype == at::ScalarType::BFloat16 || dtype == at::ScalarType::Half) {
    return at::ScalarType::Float;
  }
  return dtype;
}

}  // namespace

OpaqueXLATensor* tf_AvgPool(OpaqueXLATensor* value, Int64ArrayRef ksize,
                            Int64ArrayRef strides, enum TFPadding padding,
                            enum TFDataFormat data_format) {
  const auto value_shape_ref = value->shape();
  int num_spatial_dims = (*value_shape_ref).rank() - 2;
  xla::Padding xla_padding = ToXLAPadding(padding);
  xla::TensorFormat xla_data_format =
      XlaTensorFormat(x10::ToTFFormat(data_format), num_spatial_dims);
  auto kernel_size = XlaHelpers::I64List(ksize.slice());
  auto stride = XlaHelpers::I64List(strides.slice());
  auto spatial_padding = MakeSpatialPadding(
      /*input_size=*/(*value_shape_ref).dimensions(),
      /*kernel_size=*/kernel_size,
      /*stride=*/stride, /*padding=*/xla_padding,
      /*data_format=*/xla_data_format);
  at::ScalarType reduction_type = SumAccumulationType(value->dtype());
  XLATensor upcast_input = XLATensor::to(*value, absl::nullopt, reduction_type);
  XLATensor avg_pool = XLATensor::xla_avg_pool(
      /*input=*/upcast_input,
      /*kernel_size=*/kernel_size,
      /*stride=*/stride,
      /*padding=*/spatial_padding, /*data_format=*/xla_data_format,
      /*counts_include_padding=*/xla_padding == xla::Padding::kValid);
  return new XLATensor(XLATensor::to(avg_pool, absl::nullopt, value->dtype()));
}

OpaqueXLATensor* tf_AvgPoolGrad(Int64ArrayRef origInputShape,
                                OpaqueXLATensor* grad, Int64ArrayRef ksize,
                                Int64ArrayRef strides, enum TFPadding padding,
                                enum TFDataFormat data_format) {
  xla::Padding xla_padding = ToXLAPadding(padding);
  int num_spatial_dims = ksize.size - 2;
  xla::TensorFormat xla_data_format =
      XlaTensorFormat(x10::ToTFFormat(data_format), num_spatial_dims);
  auto kernel_size = XlaHelpers::I64List(ksize.slice());
  auto stride = XlaHelpers::I64List(strides.slice());
  auto padding_values = MakeSpatialPadding(
      /*input_size=*/origInputShape.slice(), /*kernel_size=*/kernel_size,
      /*stride=*/stride, /*padding=*/xla_padding,
      /*data_format=*/xla_data_format);
  at::ScalarType reduction_type = SumAccumulationType(grad->dtype());
  XLATensor converted_out_backprop =
      XLATensor::to(*grad, absl::nullopt, reduction_type);
  XLATensor in_backprop = XLATensor::xla_avg_pool_grad(
      /*out_backprop=*/converted_out_backprop,
      /*gradients_size=*/origInputShape.slice(),
      /*kernel_size=*/kernel_size, /*stride=*/stride,
      /*spatial_padding=*/padding_values, /*data_format=*/xla_data_format,
      /*counts_include_padding=*/xla_padding == xla::Padding::kValid);
  return new XLATensor(
      XLATensor::to(in_backprop, absl::nullopt, grad->dtype()));
}

OpaqueXLATensor* tf_MaxPool(OpaqueXLATensor* input, Int64ArrayRef ksize,
                            Int64ArrayRef strides, enum TFPadding padding,
                            enum TFDataFormat data_format) {
  xla::Padding xla_padding = ToXLAPadding(padding);
  int num_spatial_dims = ksize.size - 2;
  xla::TensorFormat xla_data_format =
      XlaTensorFormat(x10::ToTFFormat(data_format), num_spatial_dims);
  auto kernel_size = XlaHelpers::I64List(ksize.slice());
  auto stride = XlaHelpers::I64List(strides.slice());
  return new XLATensor(XLATensor::xla_max_pool(
      /*input=*/*input, /*kernel_size=*/kernel_size, /*stride=*/stride,
      /*padding=*/xla_padding, /*data_format=*/xla_data_format));
}

OpaqueXLATensor* tf_MaxPoolGrad(OpaqueXLATensor* input, OpaqueXLATensor* grad,
                                Int64ArrayRef ksize, Int64ArrayRef strides,
                                enum TFPadding padding) {
  xla::Padding xla_padding = ToXLAPadding(padding);
  auto kernel_size = XlaHelpers::I64List(ksize.slice());
  auto stride = XlaHelpers::I64List(strides.slice());
  return new XLATensor(XLATensor::xla_max_pool_grad(
      /*input=*/*input, /*out_backprop=*/*grad, /*kernel_size=*/kernel_size,
      /*stride=*/stride,
      /*padding=*/xla_padding));
}
