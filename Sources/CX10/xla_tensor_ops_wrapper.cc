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

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/convert_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/softmax_builder.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/xla_lower_util.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "xla_tensor_wrapper.h"

namespace at {
xla::hash_t Hash(const c10::optional<at::ScalarType>& dtype) {
  return xla::util::Hash(swift_xla::OptionalOr<int>(dtype, -1));
}
xla::hash_t Hash(const at::Scalar& value) {
  return value.isFloatingPoint() ? xla::util::Hash(value.toDouble())
                                 : xla::util::Hash(value.toLong());
}
}
namespace swift_xla {
void OpFieldToString(std::ostream& stream, const char* field_name, const c10::optional<at::ScalarType>& dtype) {
  if (dtype) stream << ", " << field_name << "=" << *dtype;
}
void OpFieldToString(std::ostream& stream, const char* field_name, bool value) {
  stream << ", " << field_name << "=" << value;
}
void OpFieldToString(std::ostream& stream, const char* field_name, xla::int64 value) {
  stream << ", " << field_name << "=" << value;
}
void OpFieldToString(std::ostream& stream, const char* field_name, float value) {
  stream << ", " << field_name << "=" << value;
}
void OpFieldToString(std::ostream& stream, const char* field_name,
                     const std::vector<xla::int64>& value) {
  stream << ", " << field_name << "=[";
  for (size_t i = 0; i < value.size(); ++i) {
    if (i != 0) stream << ", ";
    stream << value[i];
  }
  stream << "]";
}
void OpFieldToString(std::ostream& stream, const char* field_name,
                     const at::Scalar& value) {
  stream << ", " << field_name << "=";
  if (value.isFloatingPoint())
    stream << value.toDouble();
  else
    stream << value.toLong();
}
}  // namespace swift_xla

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

using BinaryOpBuilderWithDim = xla::XlaOp (*)(xla::XlaOp, xla::XlaOp,
                                              absl::Span<const xla::int64>);
template <BinaryOpBuilderWithDim T>
xla::XlaOp LowerBinaryOp(xla::XlaOp lhs, xla::XlaOp rhs) {
  std::tie(lhs, rhs) = XlaHelpers::Promote(lhs, rhs);
  return T(lhs, rhs, {});
}

using BinaryOpBuilder = xla::XlaOp (*)(xla::XlaOp, xla::XlaOp);
template <BinaryOpBuilder T>
xla::XlaOp LowerBinaryValueOp(xla::XlaOp lhs, xla::XlaOp rhs) {
  std::tie(lhs, rhs) = XlaHelpers::PromoteValues(lhs, rhs);
  return T(lhs, rhs);
}

xla::XlaOp LowerSqueeze(xla::XlaOp input, int dim) {
  if (dim == -1) return SqueezeAllTrivialDimensions(input);
  XLA_CHECK_GE(dim, 0);
  return SqueezeTrivialDimension(input, dim);
}

xla::XlaOp LowerCumSum(xla::XlaOp input, xla::int64 dim,
                       c10::optional<at::ScalarType> dtype, bool exclusive,
                       bool reverse) {
  xla::XlaOp casted_input = CastToScalarType(input, dtype);
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(casted_input);
  xla::XlaOp init = XlaHelpers::ScalarValue<float>(
      0, input_shape.element_type(), casted_input.builder());
  xla::XlaComputation reducer =
      XlaHelpers::CreateAddComputation(input_shape.element_type());
  return BuildCumulativeComputation(casted_input, dim, reducer, init, exclusive,
                                    reverse);
}

xla::XlaOp LowerCumProd(xla::XlaOp input, xla::int64 dim,
                        c10::optional<at::ScalarType> dtype, bool exclusive,
                        bool reverse) {
  xla::XlaOp casted_input = CastToScalarType(input, dtype);
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(casted_input);
  xla::XlaOp init =
      xla::One(casted_input.builder(), input_shape.element_type());
  xla::XlaComputation reducer =
      XlaHelpers::CreateMulComputation(input_shape.element_type());
  return BuildCumulativeComputation(casted_input, dim, reducer, init, exclusive,
                                    reverse);
}

xla::Shape CumOpShapeFn(const Value& input, xla::int64 dim,
                        c10::optional<at::ScalarType> dtype, bool exclusive,
                        bool reverse) {
  if (dtype) {
    return xla::ShapeUtil::ChangeElementType(
        input.shape(), MakeXlaPrimitiveType(*dtype, /*device=*/nullptr));
  }
  return input.shape();
}

xla::XlaOp LowerClamp(xla::XlaOp xla_input, xla::XlaOp xla_min,
                      xla::XlaOp xla_max) {
  xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
  xla_min = ConvertTo(xla_min, XlaHelpers::TypeOfXlaOp(xla_min), input_type,
                      /*device=*/nullptr);
  xla_max = ConvertTo(xla_max, XlaHelpers::TypeOfXlaOp(xla_max), input_type,
                      /*device=*/nullptr);
  return xla::Clamp(xla_min, xla_input, xla_max);
}

xla::XlaOp LowerMean(xla::XlaOp input,
                     const std::vector<xla::int64>& dimensions,
                     bool keep_reduced_dimensions,
                     const c10::optional<at::ScalarType>& dtype) {
  xla::XlaOp result = BuildMean(input, dimensions, keep_reduced_dimensions);
  return dtype ? xla::ConvertElementType(
                     result, MakeXlaPrimitiveType(*dtype, /*device=*/nullptr))
               : result;
}

xla::XlaOp LowerSum(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool keep_reduced_dimensions,
                    c10::optional<at::ScalarType> dtype) {
  return BuildSum(CastToScalarType(input, dtype), dimensions,
                  keep_reduced_dimensions);
}

std::vector<xla::int64> CanonicalizeFlip(xla::Shape shape,
                                         absl::Span<const xla::int64> dims) {
  auto dimensions =
      XlaHelpers::GetCanonicalDimensionIndices(dims, shape.rank());
  std::set<xla::int64> unique_dims(dimensions.begin(), dimensions.end());
  XLA_CHECK_EQ(unique_dims.size(), dimensions.size());
  return dimensions;
}

std::vector<xla::int64> CanonicalizeExpand(xla::Shape shape,
                                           absl::Span<const xla::int64> dims) {
  std::vector<xla::int64> dimensions(dims.begin(), dims.end());
  XLA_CHECK_GE(dimensions.size(), shape.rank()) << shape;
  xla::int64 base = dimensions.size() - shape.rank();
  for (size_t i = 0; i < shape.rank(); ++i) {
    if (dimensions[base + i] == -1) {
      dimensions[base + i] = shape.dimensions(i);
    }
  }
  return dimensions;
}

xla::XlaOp LowerPad(xla::XlaOp input, absl::Span<const xla::int64> pad,
                    const at::Scalar& value) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Pad(input,
                  XlaHelpers::ScalarValue(value, input_shape.element_type(),
                                          input.builder()),
                  XlaHelpers::MakeXlaPaddingConfigFromNdPadding(pad));
}

std::vector<xla::int64> CanonicalizePad(xla::Shape shape,
                                        absl::Span<const xla::int64> pad) {
  std::vector<xla::int64> complete_pad(pad.begin(), pad.end());
  complete_pad.resize(2 * shape.rank());
  return complete_pad;
}

xla::int64 SliceGetStride(xla::int64 start, xla::int64 end, xla::int64 stride) {
  if (stride == 0) {
    XLA_CHECK_EQ(start, end);
    stride = 1;
  }
  return stride;
}

xla::XlaOp LowerSlice(xla::XlaOp input, xla::int64 dim, xla::int64 start,
                      xla::int64 end, xla::int64 stride) {
  return xla::SliceInDim(input, start, end, SliceGetStride(start, end, stride),
                         dim);
}

xla::Shape ShapeSlice(const Value& input, xla::int64 dim, xla::int64 start,
                      xla::int64 end, xla::int64 stride) {
  xla::int64 effective_stride = SliceGetStride(start, end, stride);
  xla::Shape select_shape(input.shape());
  select_shape.set_dimensions(
      dim, (end - start + effective_stride - 1) / effective_stride);
  return select_shape;
}

}  // namespace
}  // namespace ops
}  // namespace ir
}  // namespace swift_xla

#include "xla_tensor_ops_wrapper_generated.cc.inc"
