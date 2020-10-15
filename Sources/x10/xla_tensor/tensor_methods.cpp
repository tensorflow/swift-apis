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

#include <algorithm>
#include <functional>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir_util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/layout_manager.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/all_reduce.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/annotate.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/expand.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/replica_id.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_stateless_random_normal.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_avg_pool.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_avg_pool_grad.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_max_pool.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_max_pool_grad.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_pad.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_slice.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/shape_builder.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"

namespace swift_xla {
namespace {

struct MinMaxValues {
  ir::Value min;
  ir::Value max;
};

MinMaxValues GetMinMaxValues(const XLATensor& tensor,
                             c10::optional<at::Scalar> min,
                             c10::optional<at::Scalar> max) {
  XLA_CHECK(min || max)
      << "At least one of \'min\' or \'max\' must not be None";
  xla::PrimitiveType raw_element_type = TensorTypeToRawXlaType(tensor.dtype());
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(raw_element_type);
  if (!min) {
    min = min_max.min;
  }
  if (!max) {
    max = min_max.max;
  }
  auto shape = tensor.shape();
  return {XLATensor::GetIrValueForScalar(*min, shape.get().element_type(),
                                         tensor.GetDevice()),
          XLATensor::GetIrValueForScalar(*max, shape.get().element_type(),
                                         tensor.GetDevice())};
}

void CheckRank(const XLATensor& t, xla::int64 expected_rank,
               const std::string& tag, const std::string& arg_name,
               int arg_number) {
  xla::int64 actual_rank = t.shape().get().rank();
  XLA_CHECK_EQ(actual_rank, expected_rank)
      << "Expected " << expected_rank << "-dimensional tensor, but got "
      << actual_rank << "-dimensional tensor for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

template <typename T>
void CheckShapeDimensions(const T& size) {
  XLA_CHECK(std::all_of(size.begin(), size.end(), [](xla::int64 dim) {
    return dim >= 0;
  })) << "Dimensions cannot be negative numbers";
}

void CheckDimensionSize(const XLATensor& t, xla::int64 dim,
                        xla::int64 expected_size, const std::string& tag,
                        const std::string& arg_name, int arg_number) {
  xla::int64 dim_size = t.size(dim);
  XLA_CHECK_EQ(t.size(dim), expected_size)
      << "Expected tensor to have size " << expected_size << " at dimension "
      << dim << ", but got size " << dim_size << " for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

std::vector<xla::int64> GetExpandDimensions(
    const xla::Shape& shape, std::vector<xla::int64> dimensions) {
  XLA_CHECK_GE(dimensions.size(), shape.rank()) << shape;
  xla::int64 base = dimensions.size() - shape.rank();
  for (size_t i = 0; i < shape.rank(); ++i) {
    if (dimensions[base + i] == -1) {
      dimensions[base + i] = shape.dimensions(i);
    }
  }
  return dimensions;
}

// Resizes and / or checks whether a list is of the given size. The list is only
// resized if its size is 1. If it's empty, it's replaced with the provided
// default first.
std::vector<xla::int64> CheckIntList(absl::Span<const xla::int64> list,
                                     size_t length, const std::string& name,
                                     std::vector<xla::int64> def = {}) {
  std::vector<xla::int64> result;
  if (list.empty()) {
    result = std::move(def);
  } else {
    result = xla::util::ToVector<xla::int64>(list);
  }
  if (result.size() == 1 && length > 1) {
    result.resize(length, result[0]);
    return result;
  }
  XLA_CHECK_EQ(result.size(), length)
      << "Invalid length for the '" << name << "' attribute";
  return result;
}

// Returns a 1-D shape for batch norm weight or bias based on the input shape.
xla::Shape BatchNormFeaturesShape(const XLATensor& input) {
  xla::PrimitiveType input_element_type =
      MakeXlaPrimitiveType(input.dtype(), &input.GetDevice());
  auto input_shape = input.shape();
  return ShapeBuilder(input_element_type).Add(input_shape.get(), 1).Build();
}

// Returns the IR for the given input or the provided default value broadcasted
// to the default shape, if the input is undefined.
ir::Value GetIrValueOrDefault(const XLATensor& input, at::Scalar default_value,
                              const xla::Shape& default_shape,
                              const Device& device) {
  return input.is_null() ? XLATensor::GetIrValueForScalar(default_value,
                                                          default_shape, device)
                         : input.GetIrValue();
}

absl::optional<ir::Value> GetOptionalIrValue(const XLATensor& tensor) {
  absl::optional<ir::Value> value;
  if (!tensor.is_null()) {
    value = tensor.GetIrValue();
  }
  return value;
}

ir::Value MaybeExpand(const ir::Value& input, const xla::Shape& target_shape) {
  if (input.shape().dimensions() == target_shape.dimensions()) {
    return input;
  }
  return ir::MakeNode<ir::ops::Expand>(
      input, xla::util::ToVector<xla::int64>(target_shape.dimensions()));
}

void CheckIsIntegralOrPred(const xla::Shape& shape,
                           const std::string& op_name) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(shape) ||
            shape.element_type() == xla::PrimitiveType::PRED)
      << "Operator " << op_name
      << " is only supported for integer or boolean type tensors, got: "
      << shape;
}

}  // namespace

//////////////////////////////////////////////////////////////////////////////
// XLA dedicated operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
std::pair<std::vector<XLATensor>, ir::Value> XLATensor::all_reduce(
    const std::vector<XLATensor>& inputs, const ir::Value& token,
    AllReduceType reduce_type, double scale,
    std::vector<std::vector<xla::int64>> groups) {
  std::vector<ir::Value> input_values;
  input_values.reserve(inputs.size());
  for (const XLATensor& input : inputs) {
    input_values.push_back(input.GetIrValue());
  }
  ir::NodePtr node = ir::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  std::vector<XLATensor> results;
  std::vector<ir::Value> tokens;
  for (size_t i = 0; i < inputs.size(); ++i) {
    results.push_back(inputs[i].CreateFrom(ir::Value(node, i)));
  }
  return {results, ir::Value(node, inputs.size())};
}

void XLATensor::arange_out(XLATensor& out, at::Scalar start, at::Scalar end,
                           at::Scalar step, at::ScalarType scalar_type) {
  out.SetIrValue(ir::ops::ARange(start, end, step, scalar_type));
  out.SetScalarType(scalar_type);
}

XLATensor XLATensor::annotate(const XLATensor& input, std::string annotation) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Annotate>(input.GetIrValue(), annotation));
}

std::vector<XLATensor> XLATensor::broadcast_tensors(
    absl::Span<const XLATensor> tensors) {
  XLA_CHECK(!tensors.empty()) << "broadcast_tensors cannot take an empty list";
  std::vector<ir::Value> tensor_ir_values;
  for (const auto& tensor : tensors) {
    tensor_ir_values.push_back(tensor.GetIrValue());
  }
  ir::NodePtr node = ir::ops::BroadcastTensors(tensor_ir_values);
  return tensors.front().MakeOutputTensors(node);
}

XLATensor XLATensor::tf_StatelessRandomNormal(absl::Span<const xla::int64> size,
                                              const XLATensor& seeds,
                                              const Device& device,
                                              at::ScalarType scalar_type) {
  // The Philox algorithm may cause performance regression on other devices.
  // Turn on the Philox algorithm for the CPU and GPU backends only.
  xla::Shape shape = MakeArrayShapeFromDimensions(
      size, /*dynamic_dimensions=*/{},
      MakeXlaPrimitiveType(scalar_type, &device), device.hw_type);
  return Create(ir::MakeNode<ir::ops::TfStatelessRandomNormal>(
                    shape, seeds.GetIrValue(),
                    device.hw_type == swift_xla::DeviceType::TPU
                        ? ir::ops::BitGeneratorType::THREE_FRY
                        : ir::ops::BitGeneratorType::PHILOX),
                device, scalar_type);
}

XLATensor XLATensor::to(XLATensor& input, c10::optional<Device> device,
                        c10::optional<at::ScalarType> scalar_type) {
  if (!device) {
    device = input.GetDevice();
  }
  if (!scalar_type) {
    scalar_type = input.dtype();
  }
  if (input.GetDevice() == *device) {
    return input.dtype() == *scalar_type
               ? input.CreateFrom(input.GetIrValue())
               : input.CreateFrom(input.GetIrValue(), *scalar_type);
  }
  XLATensor new_tensor = input.CopyTensorToDevice(*device);
  if (input.dtype() != *scalar_type) {
    new_tensor.SetScalarType(*scalar_type);
  }
  return new_tensor;
}

void XLATensor::linspace_out(XLATensor& out, at::Scalar start, at::Scalar stop,
                             xla::int64 num, at::ScalarType scalar_type) {
  out.SetIrValue(ir::ops::LinSpace(start, stop, num, scalar_type));
  out.SetScalarType(scalar_type);
}

XLATensor XLATensor::xla_avg_pool(
    const XLATensor& input, absl::Span<const xla::int64> kernel_size,
    absl::Span<const xla::int64> stride,
    absl::Span<const std::pair<xla::int64, xla::int64>> padding,
    const xla::TensorFormat& data_format, const bool counts_include_padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::XlaAvgPool>(
      input.GetIrValue(), XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride),
      std::vector<std::pair<xla::int64, xla::int64>>(padding.begin(),
                                                     padding.end()),
      data_format, counts_include_padding));
}

XLATensor XLATensor::xla_avg_pool_grad(
    const XLATensor& out_backprop, absl::Span<const xla::int64> gradients_size,
    absl::Span<const xla::int64> kernel_size,
    absl::Span<const xla::int64> stride,
    absl::Span<const std::pair<xla::int64, xla::int64>> spatial_padding,
    const xla::TensorFormat& data_format, const bool counts_include_padding) {
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::XlaAvgPoolGrad>(
      out_backprop.GetIrValue(), XlaHelpers::I64List(gradients_size),
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      std::vector<std::pair<xla::int64, xla::int64>>(spatial_padding.begin(),
                                                     spatial_padding.end()),
      data_format, counts_include_padding));
}

XLATensor XLATensor::xla_max_pool(const XLATensor& input,
                                  absl::Span<const xla::int64> kernel_size,
                                  absl::Span<const xla::int64> stride,
                                  xla::Padding padding,
                                  const xla::TensorFormat& data_format) {
  return input.CreateFrom(ir::MakeNode<ir::ops::XlaMaxPool>(
      input.GetIrValue(), XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), padding, data_format));
}

XLATensor XLATensor::xla_max_pool_grad(const XLATensor& input,
                                       const XLATensor& out_backprop,
                                       absl::Span<const xla::int64> kernel_size,
                                       absl::Span<const xla::int64> stride,
                                       xla::Padding padding) {
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::XlaMaxPoolGrad>(
      input.GetIrValue(), out_backprop.GetIrValue(),
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride), padding));
}

XLATensor XLATensor::xla_replica_id(const Device& device) {
  return XLATensor::Create(ir::MakeNode<ir::ops::ReplicaId>(), device);
}

}  // namespace swift_xla
