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
#include "tensorflow/compiler/tf2xla/xla_tensor/layout_manager.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/matrix.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/nll_loss.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_create_conv_attrs.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/segment_reduction_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/softmax_builder.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/xla_lower_util.h"
#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"
#include "tensorflow/compiler/tf2xla/lib/random.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/lib/qr.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
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
namespace tensorflow {
xla::hash_t Hash(tensorflow::MirrorPadMode mode) {
  return xla::util::Hash(static_cast<int>(mode));
}
}  // namespace tensorflow
namespace xla {
xla::hash_t Hash(const xla::PaddingConfig& padding_config) {
  std::vector<int64_t> low;
  std::vector<int64_t> high;
  std::vector<int64_t> interior;
  for (const xla::PaddingConfig::PaddingConfigDimension& dim_padding :
       padding_config.dimensions()) {
    low.push_back(dim_padding.edge_padding_low());
    high.push_back(dim_padding.edge_padding_high());
    interior.push_back(dim_padding.interior_padding());
  }
  return xla::util::MHash(low, high, interior);
}
xla::hash_t Hash(const xla::Shape& shape) {
  return xla::util::MHash(shape.ToString());
}
}  // namespace xla
namespace swift_xla {
void OpFieldToString(std::ostream& stream, const char* field_name, const c10::optional<at::ScalarType>& dtype) {
  if (dtype) stream << ", " << field_name << "=" << *dtype;
}
template <typename T>
void OpFieldToString(std::ostream& stream, const char* field_name, T value) {
  stream << ", " << field_name << "=" << value;
}
void OpFieldToString(std::ostream& stream, const char* field_name,
                     tensorflow::MirrorPadMode value) {
  stream << ", " << field_name << "=" << static_cast<int>(value);
}
void OpFieldToString(std::ostream& stream, const char* field_name,
                     const xla::PaddingConfig& value) {
  stream << ", " << field_name << "=" << xla::PaddingConfigToString(value);
}
void OpFieldToString(std::ostream& stream, const char* field_name,
                     const std::vector<int64_t>& value) {
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
const XLATensor* FirstTensor(OpaqueXLATensorArrayRef array) {
  XLA_CHECK_GE(array.size, 1);
  return array.data[0];
}
std::vector<ir::Value> UnpackIrValues(OpaqueXLATensorArrayRef array) {
  std::vector<ir::Value> out;
  out.reserve(array.size);
  for (size_t i = 0; i < array.size; ++i) {
    out.push_back(array.data[i]->GetIrValue());
  }
  return out;
}

}  // namespace swift_xla

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

using BinaryOpBuilderWithDim = xla::XlaOp (*)(xla::XlaOp, xla::XlaOp,
                                              absl::Span<const int64_t>);
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

std::vector<xla::XlaOp> LowerBroadcastTensors(xla::XlaOp lhs, xla::XlaOp rhs) {
  std::tie(lhs, rhs) = XlaHelpers::PromoteValues(lhs, rhs);
  return {lhs, rhs};
}

xla::XlaOp LowerSqueeze(xla::XlaOp input, int dim) {
  if (dim == -1) return SqueezeAllTrivialDimensions(input);
  XLA_CHECK_GE(dim, 0);
  return SqueezeTrivialDimension(input, dim);
}

xla::XlaOp LowerCumSum(xla::XlaOp input, int64_t dim, bool exclusive,
                       bool reverse) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp init = XlaHelpers::ScalarValue<float>(
      0, input_shape.element_type(), input.builder());
  xla::XlaComputation reducer =
      XlaHelpers::CreateAddComputation(input_shape.element_type());
  return BuildCumulativeComputation(input, dim, reducer, init, exclusive,
                                    reverse);
}

xla::XlaOp LowerCumProd(xla::XlaOp input, int64_t dim, bool exclusive,
                        bool reverse) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp init = xla::One(input.builder(), input_shape.element_type());
  xla::XlaComputation reducer =
      XlaHelpers::CreateMulComputation(input_shape.element_type());
  return BuildCumulativeComputation(input, dim, reducer, init, exclusive,
                                    reverse);
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
                     const std::vector<int64_t>& dimensions,
                     bool keep_reduced_dimensions,
                     const c10::optional<at::ScalarType>& dtype) {
  xla::XlaOp result = BuildMean(input, dimensions, keep_reduced_dimensions);
  return dtype ? xla::ConvertElementType(
                     result, MakeXlaPrimitiveType(*dtype, /*device=*/nullptr))
               : result;
}

xla::XlaOp LowerLogicalCast(xla::XlaOp input, at::ScalarType dtype) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return ConvertToRaw(input, input_shape.element_type(),
                      MakeXlaPrimitiveType(dtype, /*device=*/nullptr),
                      TensorTypeToRawXlaType(dtype), /*device=*/nullptr);
}
xla::Shape ShapeLogicalCast(const Value& input, at::ScalarType dtype) {
  xla::Shape result = input.shape();
  result.set_element_type(MakeXlaPrimitiveType(dtype, /*device=*/nullptr));
  return result;
}

xla::XlaOp LowerSum(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    bool keep_reduced_dimensions,
                    c10::optional<at::ScalarType> dtype) {
  return BuildSum(CastToScalarType(input, dtype), dimensions,
                  keep_reduced_dimensions);
}

std::vector<int64_t> CanonicalizeFlip(xla::Shape shape,
                                         absl::Span<const int64_t> dims) {
  auto dimensions =
      XlaHelpers::GetCanonicalDimensionIndices(dims, shape.rank());
  std::set<int64_t> unique_dims(dimensions.begin(), dimensions.end());
  XLA_CHECK_EQ(unique_dims.size(), dimensions.size());
  return dimensions;
}

std::vector<int64_t> CanonicalizeExpand(xla::Shape shape,
                                           absl::Span<const int64_t> dims) {
  std::vector<int64_t> dimensions(dims.begin(), dims.end());
  XLA_CHECK_GE(dimensions.size(), shape.rank()) << shape;
  int64_t base = dimensions.size() - shape.rank();
  for (size_t i = 0; i < shape.rank(); ++i) {
    if (dimensions[base + i] == -1) {
      dimensions[base + i] = shape.dimensions(i);
    }
  }
  return dimensions;
}

xla::XlaOp LowerPad(xla::XlaOp input, const at::Scalar& value,
                    const xla::PaddingConfig& config) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Pad(input,
                  XlaHelpers::ScalarValue(value, input_shape.element_type(),
                                          input.builder()),
                  config);
}

xla::XlaOp LowerPad(xla::XlaOp input, absl::Span<const int64_t> pad,
                    const at::Scalar& value) {
  return LowerPad(input, value,
                  XlaHelpers::MakeXlaPaddingConfigFromNdPadding(pad));
}

std::vector<int64_t> CanonicalizePad(xla::Shape shape,
                                        absl::Span<const int64_t> pad) {
  std::vector<int64_t> complete_pad(pad.begin(), pad.end());
  complete_pad.resize(2 * shape.rank());
  return complete_pad;
}

int64_t SliceGetStride(int64_t start, int64_t end, int64_t stride) {
  if (stride == 0) {
    XLA_CHECK_EQ(start, end);
    stride = 1;
  }
  return stride;
}

xla::XlaOp LowerSlice(xla::XlaOp input, int64_t dim, int64_t start,
                      int64_t end, int64_t stride) {
  return xla::SliceInDim(input, start, end, SliceGetStride(start, end, stride),
                         dim);
}

xla::Shape ShapeSlice(const Value& input, int64_t dim, int64_t start,
                      int64_t end, int64_t stride) {
  int64_t effective_stride = SliceGetStride(start, end, stride);
  xla::Shape select_shape(input.shape());
  select_shape.set_dimensions(
      dim, (end - start + effective_stride - 1) / effective_stride);
  return select_shape;
}

xla::XlaOp LowerWhere(xla::XlaOp condition, xla::XlaOp input,
                      xla::XlaOp other) {
  xla::XlaOp pred_condition =
      ConvertTo(condition, XlaHelpers::TypeOfXlaOp(condition),
                xla::PrimitiveType::PRED, /*device=*/nullptr);
  std::tie(input, other) = XlaHelpers::PromoteShapes(input, other);
  return xla::Select(pred_condition, input, other);
}

std::vector<xla::XlaOp> BuildTopK(xla::XlaOp input, int64_t k,
                                  int64_t dim, bool largest) {
  return CreateTopK(input, k, dim, largest, true);
}

xla::XlaOp BuildOneHot(xla::XlaOp indices, xla::XlaOp on_value,
                       xla::XlaOp off_value, int64_t depth,
                       int64_t axis) {
  xla::XlaBuilder* builder = indices.builder();
  xla::Shape indices_shape = XlaHelpers::ShapeOfXlaOp(indices);
  std::vector<int64_t> broadcast_dims(indices_shape.dimensions().size());
  if (axis < 0) axis = axis + broadcast_dims.size() + 1;
  std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
  std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);

  std::vector<int64_t> output_dimensions(indices_shape.dimensions().size() +
                                            1);
  output_dimensions.assign(indices_shape.dimensions().begin(),
                           indices_shape.dimensions().end());
  output_dimensions.insert(output_dimensions.begin() + axis, depth);
  xla::Shape iota_shape = xla::ShapeUtil::MakeShape(
      indices_shape.element_type(), output_dimensions);

  return xla::Select(
      xla::Eq(indices, xla::Iota(builder, iota_shape, axis), broadcast_dims),
      xla::Broadcast(on_value, output_dimensions),
      xla::Broadcast(off_value, output_dimensions));
}

xla::XlaOp LowerTfUnsortedSegmentSum(xla::XlaOp data, xla::XlaOp indices,
                                     int64_t num_segments) {
  const xla::Shape& data_shape = XlaHelpers::ShapeOfXlaOp(data);
  xla::XlaOp init_value = xla::Zero(data.builder(), data_shape.element_type());
  auto combine = [](xla::XlaOp a, xla::XlaOp b) { return a + b; };
  return UnsortedSegmentReduce(data, indices, init_value, num_segments,
                               combine);
}

xla::BitGeneratorTy GetBestGenerator(LoweringContext* loctx = nullptr) {
  xla::BitGeneratorTy generator;
  if (!loctx || loctx->device().hw_type == swift_xla::DeviceType::TPU) {
    generator = xla::ThreeFryBitGenerator;
  } else {
    generator = [](xla::XlaOp key, xla::XlaOp state, const xla::Shape& shape) {
      std::tie(state, key) = xla::ScramblePhiloxKey(key);
      return xla::PhiloxBitGenerator(key, state, shape);
    };
  }
  return generator;
}

xla::XlaOp LowerTfStatelessRandomNormal(xla::Shape shape, xla::XlaOp seeds,
                                        at::ScalarType dtype,
                                        LoweringContext* loctx = nullptr) {
  auto generator = GetBestGenerator(loctx);
  xla::XlaOp seed0 = xla::Reshape(xla::Slice(seeds, {0}, {1}, {1}), {});
  xla::XlaOp seed1 = xla::Reshape(xla::Slice(seeds, {1}, {2}, {1}), {});
  xla::XlaOp initial_state =
      xla::ConstantR0WithType(seeds.builder(), xla::U64, 0);
  xla::XlaOp key = ConvertElementType(seed0, xla::U64) |
                   ShiftLeft(ConvertElementType(seed1, xla::U64),
                             ConstantR0WithType(seeds.builder(), xla::U64, 32));
  xla::XlaOp normal =
      xla::NormalFloatingPointDistribution(key, initial_state, generator, shape)
          .value;
  return normal;
}

xla::XlaOp LowerTfStatelessRandomUniform(xla::Shape shape, xla::XlaOp seeds,
                                         xla::XlaOp minval, xla::XlaOp maxval,
                                         LoweringContext* loctx = nullptr) {
  auto generator = GetBestGenerator(loctx);
  xla::XlaOp seed0 = xla::Reshape(xla::Slice(seeds, {0}, {1}, {1}), {});
  xla::XlaOp seed1 = xla::Reshape(xla::Slice(seeds, {1}, {2}, {1}), {});
  xla::XlaOp key = ConvertElementType(seed0, xla::U64) |
                   ShiftLeft(ConvertElementType(seed1, xla::U64),
                             ConstantR0WithType(seeds.builder(), xla::U64, 32));
  xla::XlaOp initial_state =
      xla::ConstantR0WithType(seeds.builder(), xla::U64, 0);
  xla::PrimitiveType type = shape.element_type();
  xla::XlaOp output;
  switch (type) {
    case xla::F32:
    case xla::F64: {
      return xla::UniformFloatingPointDistribution(
                 key, initial_state, generator, minval, maxval, shape)
          .value;
    }
    case xla::S32:
    case xla::S64: {
      return xla::UniformIntDistribution(key, initial_state, generator, minval,
                                         maxval, shape)
          .value;
    }
    default: {
      XLA_ERROR() << "Types other than F32, S32 and S64 are not implemented by "
                     "StatelessRngUniform; got "
                  << xla::primitive_util::LowercasePrimitiveTypeName(type);
    }
  }
}

xla::XlaOp LowerNllLoss(xla::XlaOp logits, xla::XlaOp labels,
                        int64_t ignore_index) {
  return BuildNllLoss(logits, labels, absl::nullopt, ignore_index,
                      ReductionMode::kMean);
}

xla::XlaOp LowerProd(xla::XlaOp input,
                     const std::vector<int64_t>& dimensions,
                     bool keep_reduced_dimensions) {
  xla::XlaOp casted_input;
  casted_input = ConvertToNumeric(input, XlaHelpers::TypeOfXlaOp(input));
  return BuildProd(casted_input, dimensions, keep_reduced_dimensions);
}

xla::XlaOp LowerSelect(xla::XlaOp input, int64_t dim, int64_t index) {
  auto input_shape = XlaHelpers::ShapeOfXlaOp(input);
  index =
      XlaHelpers::GetCanonicalPosition(input_shape.dimensions(), dim, index);
  return SqueezeTrivialDimension(
      xla::SliceInDim(input, index, index + 1, 1, dim), dim);
}

std::vector<xla::XlaOp> GetArrayOperands(LoweringContext* loctx,
                                         absl::Span<const Output> operands,
                                         size_t offset) {
  std::vector<xla::XlaOp> inputs;
  operands = operands.subspan(offset);
  inputs.reserve(operands.size());
  for (auto& operand : operands) {
    inputs.push_back(loctx->GetOutputOp(operand));
  }
  return inputs;
}

std::vector<xla::XlaOp> MakeParameterList(xla::XlaBuilder* b, size_t offset,
                                          absl::Span<const Value> inputs,
                                          const char* name) {
  std::vector<xla::XlaOp> out;
  out.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    out.push_back(xla::Parameter(b, offset + i, inputs[i].shape(),
                                 absl::StrCat(name, "_", i)));
  }
  return out;
}

std::vector<Value> TensorArgsConcat(absl::Span<const Value> inputa,
                                    absl::Span<const Value> inputb) {
  std::vector<Value> out;
  out.reserve(inputa.size() + inputb.size());
  out.insert(out.end(), inputa.begin(), inputa.end());
  out.insert(out.end(), inputb.begin(), inputb.end());
  return out;
}

int64_t CanonicalizeStack(absl::Span<const Value> inputs, int64_t dim) {
  XLA_CHECK_GE(inputs.size(), 1);
  return swift_xla::XlaHelpers::GetCanonicalDimensionIndex(
      dim, inputs[0].shape().rank() + 1);
}

int64_t CanonicalizeCat(absl::Span<const Value> inputs, int64_t dim) {
  XLA_CHECK_GE(inputs.size(), 1);
  xla::Shape first_shape = inputs[0].shape();
  dim = swift_xla::XlaHelpers::GetCanonicalDimensionIndex(dim,
                                                          first_shape.rank());
  first_shape.DeleteDimension(dim);
  for (size_t i = 1; i < inputs.size(); ++i) {
    xla::Shape tensor_shape = inputs[i].shape();
    tensor_shape.DeleteDimension(dim);
    XLA_CHECK(xla::ShapeUtil::Compatible(first_shape, tensor_shape))
        << first_shape << " vs. " << tensor_shape;
  }
  return dim;
}

std::vector<xla::XlaOp> LowerQR(xla::XlaOp input, bool full_matrices) {
  xla::XlaOp q, r;
  xla::QrExplicit(input, /*full_matrices=*/full_matrices, q, r);
  return {q, r};
}

std::vector<xla::XlaOp> LowerSVD(xla::XlaOp input, bool compute_uv,
                                 bool full_matrix) {
  xla::SVDResult svd_result =
      xla::SVD(input, /*max_iter=*/100, /*epsilon=*/1e-6,
               XlaHelpers::mat_mul_precision());
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp u = svd_result.u;
  xla::XlaOp v = svd_result.v;
  if (!compute_uv) {
    u = xla::Zeros(input.builder(), XlaHelpers::ShapeOfXlaOp(u));
    v = xla::Zeros(input.builder(), XlaHelpers::ShapeOfXlaOp(v));
  } else if (!full_matrix) {
    int64_t m_dim = input_shape.dimensions(input_shape.rank() - 2);
    int64_t n_dim = input_shape.dimensions(input_shape.rank() - 1);
    std::vector<int64_t> base_indices(input_shape.rank(), 0);

    auto u_sizes = xla::util::ToVector<int64_t>(input_shape.dimensions());
    u_sizes[input_shape.rank() - 1] = std::min(m_dim, n_dim);
    u = BuildSlice(u, base_indices, u_sizes);

    auto v_sizes = xla::util::ToVector<int64_t>(input_shape.dimensions());
    v_sizes[input_shape.rank() - 2] = n_dim;
    v_sizes[input_shape.rank() - 1] = std::min(m_dim, n_dim);
    v = BuildSlice(v, base_indices, v_sizes);
  }
  return {svd_result.d, u, v};
}

xla::Shape ShapeOfXlaOpList(absl::Span<const xla::XlaOp> ops) {
  xla::Shape result;
  result.set_element_type(xla::TUPLE);
  result.mutable_tuple_shapes()->reserve(ops.size());
  for (const auto& op : ops) {
    xla::ShapeUtil::AppendShapeToTuple(XlaHelpers::ShapeOfXlaOp(op), &result);
  }
  TF_DCHECK_OK(xla::ShapeUtil::ValidateShapeWithOptionalLayout(result));
  return result;
}

xla::XlaOp BuildTfConv(xla::XlaOp input, xla::XlaOp filter, bool depthwise,
                       absl::Span<const int64_t> strides,
                       tensorflow::Padding padding,
                       absl::Span<const int64_t> explicit_paddings,
                       tensorflow::TensorFormat data_format,
                       absl::Span<const int64_t> dilations) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  int num_spatial_dims = input_shape.rank() - 2;
  tensorflow::ConvOpAttrs attrs =
      CreateConvOpAttrs(num_spatial_dims, depthwise, strides, padding,
                        explicit_paddings, data_format, dilations);
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(XlaHelpers::mat_mul_precision());
  return ConsumeValue(tensorflow::MakeXlaForwardConvOp(
      /*type_string=*/"TfConv", /*conv_input=*/input, /*filter=*/filter,
      /*attrs=*/attrs, /*precision_config=*/&precision_config));
}

xla::XlaOp BuildTfConvBackpropFilter(
    xla::XlaOp input, absl::Span<const int64_t> filter_sizes,
    xla::XlaOp out_backprop, bool depthwise,
    absl::Span<const int64_t> strides, tensorflow::Padding padding,
    absl::Span<const int64_t> explicit_paddings,
    tensorflow::TensorFormat data_format,
    absl::Span<const int64_t> dilations) {
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

xla::XlaOp BuildTfConvBackpropInput(
    absl::Span<const int64_t> input_sizes, xla::XlaOp filter,
    xla::XlaOp out_backprop, bool depthwise,
    absl::Span<const int64_t> strides, tensorflow::Padding padding,
    absl::Span<const int64_t> explicit_paddings,
    tensorflow::TensorFormat data_format,
    absl::Span<const int64_t> dilations) {
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

}  // namespace
}  // namespace ops
}  // namespace ir
}  // namespace swift_xla

namespace {

tensorflow::Padding ToTFPadding(TFPadding padding) {
  switch (padding) {
    case TFPadding_VALID: {
      return tensorflow::VALID;
    }
    case TFPadding_SAME: {
      return tensorflow::SAME;
    }
    case TFPadding_EXPLICIT: {
      return tensorflow::EXPLICIT;
    }
    default: {
      LOG(FATAL) << "Invalid padding: " << padding;
    }
  }
}

tensorflow::MirrorPadMode ToTFMirrorPadMode(TFMirrorPadMode mode) {
  switch (mode) {
    case TFMirrorPadMode_REFLECT: {
      return tensorflow::MirrorPadMode::REFLECT;
    }
    case TFMirrorPadMode_SYMMETRIC: {
      return tensorflow::MirrorPadMode::SYMMETRIC;
    }
    default: {
      LOG(FATAL) << "Invalid mirror pad mode: " << mode;
    }
  }
}

xla::PaddingConfig ToXLAPaddingConfig(PaddingConfig padding_config) {
  xla::PaddingConfig xla_padding_config;
  for (size_t i = 0; i < padding_config.count; ++i) {
    xla::PaddingConfig::PaddingConfigDimension* dims =
        xla_padding_config.add_dimensions();
    const PaddingConfigDimension& padding_dim = padding_config.dimensions[i];
    dims->set_edge_padding_low(padding_dim.edge_padding_low);
    dims->set_edge_padding_high(padding_dim.edge_padding_high);
    dims->set_interior_padding(padding_dim.interior_padding);
  }
  return xla_padding_config;
}

}  // namespace

#include "xla_tensor_ops_wrapper_generated.cc.inc"
