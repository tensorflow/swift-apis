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

#include "xla_tensor_wrapper.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/softmax_builder.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/convert_ops.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"

namespace at {
xla::hash_t Hash(const c10::optional<at::ScalarType>& dtype) {
  return xla::util::Hash(swift_xla::OptionalOr<int>(dtype, -1));
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
}  // namespace swift_xla

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

using BinaryOpBuilder = xla::XlaOp(*)(xla::XlaOp, xla::XlaOp, absl::Span<const xla::int64>);
template <BinaryOpBuilder T>
xla::XlaOp LowerBinaryOp(xla::XlaOp lhs, xla::XlaOp rhs) {
  std::tie(lhs, rhs) = XlaHelpers::Promote(lhs, rhs);
  return T(lhs, rhs, {});
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

}  // namespace
}  // namespace ops
}  // namespace ir
}  // namespace swift_xla

#include "xla_tensor_ops_wrapper_generated.cc.inc"
