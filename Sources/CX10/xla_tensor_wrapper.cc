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

#include <random>

#include "tensorflow/compiler/tf2xla/xla_tensor/aten_compat.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir_dump_util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/layout_manager.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/token.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/strided_slice_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"
#include "tensorflow/core/util/mirror_pad_mode.h"

using swift_xla::XlaHelpers;
using swift_xla::XLATensor;

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

XLATensor MakeEmpty(at::ScalarType scalar_type, swift_xla::Device device) {
  at::Tensor empty(at::AnyScalarBuffer::empty(scalar_type), {});
  return XLATensor::Create(empty, device);
}

OpaqueXLATensorArrayRef ConvertTensorList(
    const std::vector<XLATensor>& tensors) {
  size_t count = tensors.size();
  auto opaque_tensors = new OpaqueXLATensor*[count];
  for (size_t i = 0; i < count; ++i) {
    opaque_tensors[i] = new XLATensor(tensors[i]);
  }
  return {opaque_tensors, count};
}

template <class T>
Int64ArrayRef Int64ArrayRefFromCollection(const T& collection) {
  int64_t* array = new int64_t[collection.size()];
  std::copy(collection.begin(), collection.end(), array);
  return {array, collection.size()};
}

}  // namespace

at::Scalar atScalar(XLAScalar s) {
  switch (s.tag) {
    case XLAScalarTypeTag_i:
      return at::Scalar(s.value.i);
    case XLAScalarTypeTag_d:
      return at::Scalar(s.value.d);
    default:
      LOG(FATAL) << "Invalid tag: " << s.tag;
  }
}

swift_xla::XLATensor* XLATensor_makeScalar(XLAScalar value,
                                           enum XLATensorScalarType type,
                                           const struct CDevice cdevice) {
  return new swift_xla::XLATensor(swift_xla::XLATensor::Create(
      atScalar(value), ToScalarType(type), ConvertDevice(cdevice)));
}

// TODO(parkers): reduce copying here...
swift_xla::XLATensor* copyTensor(XLATensorScalarType type,
                                 const void* raw_value, size_t num_entries,
                                 const size_t* shape, size_t rank,
                                 const struct CDevice device) {
  switch (type) {
#define DEFINE_COPY_CASE(name, aten_name, DType)                 \
  case XLATensorScalarType_##name: {                             \
    auto* value = reinterpret_cast<const DType*>(raw_value);     \
    std::unique_ptr<DType[]> data(new DType[num_entries]);       \
    memcpy(data.get(), value, num_entries * sizeof(DType));      \
    std::vector<int64_t> dims(shape, shape + rank);              \
    at::Tensor t(std::move(data), std::move(dims));              \
    return new swift_xla::XLATensor(                             \
        swift_xla::XLATensor::Create(t, ConvertDevice(device))); \
  }
    LIST_SCALAR_TYPES(DEFINE_COPY_CASE)
#undef DEFINE_COPY_CASE
    default:
      LOG(FATAL) << "Invalid type: " << type;
  }
}
OpaqueXLATensor* copyTensorAndMakeResident(enum XLATensorScalarType type,
                                           const void* value,
                                           size_t num_entries,
                                           const size_t* shape, size_t rank,
                                           const struct CDevice cdevice,
                                           bool to_reduced_precision) {
  if (to_reduced_precision && XLATensorScalarType_Float == type) {
    const float* float_buffer = reinterpret_cast<const float*>(value);
    auto non_owned_buffer =
        std::make_unique<at::NonOwnedAnyScalarBuffer<float>>(
            float_buffer, num_entries * sizeof(float));
    std::vector<int64_t> dims(shape, shape + rank);
    auto device = ConvertDevice(cdevice);
    auto dest_shape = swift_xla::MakeArrayShapeFromDimensions(
        XlaHelpers::I64List(dims), /*dynamic_dimensions=*/{},
        xla::PrimitiveType::BF16, device.hw_type);
    at::Tensor t(std::move(non_owned_buffer), std::move(dims));
    auto xla_data = swift_xla::TensorToXlaData(t, dest_shape, device);
    return new swift_xla::XLATensor(
        swift_xla::XLATensor::Create(xla_data, at::ScalarType::Float));
  }
  if (XLATensorScalarType_Float == type && xla::ComputationClient::IsLocal()) {
    auto device = ConvertDevice(cdevice);
    std::vector<xla::int64> dims(shape, shape + rank);
    auto dest_shape = swift_xla::MakeArrayShapeFromDimensions(
        dims, /*dynamic_dimensions=*/{}, xla::PrimitiveType::F32,
        device.hw_type);
    auto host_shape = swift_xla::MakeSwiftTensorLayout(
        dims, /*dynamic_dimensions=*/{}, xla::PrimitiveType::F32);
    xla::BorrowingLiteral literal(reinterpret_cast<const char*>(value),
                                  host_shape);

    return new swift_xla::XLATensor(swift_xla::XLATensor::Create(
        xla::ComputationClient::Get()->TransferToServer(
            std::move(literal), dest_shape, device.ToString())));
  }
  return copyTensor(type, value, num_entries, shape, rank, cdevice);
}

const void* MaterializedTensor_getData(OpaqueMaterializedTensor* t) {
  return t->buffer().raw_data();
}

OpaqueMaterializedTensor* XLATensor_materialize(OpaqueXLATensor* t) {
  // Avoid barriers for fetching trivial local tensors.
  auto current_tensor = t->CurrentTensorData();
  if (current_tensor) return new at::Tensor(std::move(*current_tensor));
  t->ApplyPendingGraph();
  return new at::Tensor(t->ToTensor(/*detached=*/false));
}

enum XLATensorScalarType MaterializedTensor_getType(
    OpaqueMaterializedTensor* t) {
  return FromScalarType(t->scalar_type());
}
enum XLATensorScalarType XLATensor_dtype(OpaqueXLATensor* a) {
  return FromScalarType(a->dtype());
}
enum XLATensorScalarType XLATensor_physical_scalar_type(OpaqueXLATensor* a) {
  return FromScalarType(a->physical_scalar_type());
}

at::ScalarType ToScalarType(XLATensorScalarType type) {
  switch (type) {
#define DEFINE_TYPE_CASE(name, aten_name, DType) \
  case XLATensorScalarType_##name:               \
    return at::ScalarType::aten_name;
    LIST_SCALAR_TYPES(DEFINE_TYPE_CASE)
#undef DEFINE_TYPE_CASE
    default:
      LOG(FATAL) << "Invalid type: " << type;
  }
}

XLATensorScalarType FromScalarType(at::ScalarType type) {
  switch (type) {
#define DEFINE_TYPE_CASE(name, aten_name, DType) \
  case at::ScalarType::aten_name:                \
    return XLATensorScalarType_##name;
    LIST_SCALAR_TYPES(DEFINE_TYPE_CASE)
#undef DEFINE_TYPE_CASE
    default:
      LOG(FATAL) << "Invalid type: " << type;
  }
}

void destroyTensor(swift_xla::XLATensor* t) { delete t; }
void destroyMaterializedTensor(OpaqueMaterializedTensor* t) { delete t; }
void destroyXLAShape(xla::util::MaybeRef<xla::Shape>* s) { delete s; }

xla::util::MaybeRef<xla::Shape>* fetchTensorShape(
    swift_xla::XLATensor* tensor) {
  return new xla::util::MaybeRef<xla::Shape>(tensor->shape());
}

size_t XLAShape_getRank(xla::util::MaybeRef<xla::Shape>* shape) {
  return shape->get().dimensions().size();
}

const int64_t* XLAShape_getDimensions(OpaqueXLAShape* shape) {
  static_assert(sizeof(int64_t) == sizeof(xla::int64), "Sanity");
  return reinterpret_cast<const int64_t*>(shape->get().dimensions().data());
}

static c10::optional<swift_xla::Device> AsOptional(const CDevice* device) {
  if (!device) return absl::nullopt;
  return ConvertDevice(*device);
}
static const CDevice& CheckedRef(const CDevice* device) {
  assert(device);
  return *device;
}

XLAAnnotationScope* MakeAnnotationScope(const char* scope) {
  return new tensorflow::profiler::TraceMe(scope);
}
void DestroyAnnotationScope(XLAAnnotationScope* scope) {
  if (scope) delete scope;
}

namespace x10 {

tensorflow::TensorFormat ToTFFormat(TFDataFormat data_format) {
  switch (data_format) {
    case TFDataFormat_NHWC: {
      return tensorflow::TensorFormat::FORMAT_NHWC;
    }
    case TFDataFormat_NCHW: {
      return tensorflow::TensorFormat::FORMAT_NCHW;
    }
    default: {
      LOG(FATAL) << "Invalid format: " << data_format;
    }
  }
}

}  // namespace x10

void destroyOpaqueXLATensorArrayRef(OpaqueXLATensorArrayRef tensor_list) {
  delete[] tensor_list.data;
}

void destroyStridedSliceSpec(StridedSliceSpec* strided_slice_spec) {
  delete[] strided_slice_spec->begin.data;
  delete[] strided_slice_spec->end.data;
  delete[] strided_slice_spec->strides.data;
  delete[] strided_slice_spec->processing_sizes.data;
  delete[] strided_slice_spec->final_sizes.data;
  delete strided_slice_spec;
}

// Ops.
OpaqueXLATensor* XLATensor_abs(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::abs(*a));
}
OpaqueXLATensor* XLATensor_acos(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::acos(*a));
}
OpaqueXLATensor* XLATensor_acosh(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::acosh(*a));
}
OpaqueXLATensor* XLATensor_cumprod(OpaqueXLATensor* a, int64_t dim,
                                   Optional_XLAScalarType dtype, bool exclusive,
                                   bool reverse) {
  return new XLATensor(
      XLATensor::cumprod(*a, dim, dtype.value(), exclusive, reverse));
}
OpaqueXLATensor* XLATensor_cumsum(OpaqueXLATensor* a, int64_t dim,
                                  Optional_XLAScalarType dtype, bool exclusive,
                                  bool reverse) {
  return new XLATensor(
      XLATensor::cumsum(*a, dim, dtype.value(), exclusive, reverse));
}
OpaqueXLATensor* XLATensor_add(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::add(*a, *b));
}
OpaqueXLATensor* XLATensor_all(OpaqueXLATensor* input, Int64ArrayRef dimensions,
                               bool keep_reduced_dimensions) {
  return new XLATensor(XLATensor::all(*input,
                                      XlaHelpers::I64List(dimensions.slice()),
                                      keep_reduced_dimensions));
}
OpaqueXLATensor* XLATensor_any(OpaqueXLATensor* input, Int64ArrayRef dimensions,
                               bool keep_reduced_dimensions) {
  return new XLATensor(XLATensor::any(*input,
                                      XlaHelpers::I64List(dimensions.slice()),
                                      keep_reduced_dimensions));
}
OpaqueXLATensor* XLATensor_arange(XLAScalar start, XLAScalar end,
                                  XLAScalar step, const CDevice device,
                                  enum XLATensorScalarType type) {
  XLATensor out = MakeEmpty(ToScalarType(type), ConvertDevice(device));
  XLATensor::arange_out(out, atScalar(start), atScalar(end), atScalar(step),
                        ToScalarType(type));
  return new XLATensor(out);
}
OpaqueXLATensor* XLATensor_argmax(OpaqueXLATensor* a, int64_t dim,
                                  bool keepdim) {
  XLATensor out = MakeEmpty(a->dtype(), a->GetDevice());
  XLATensor indices = MakeEmpty(at::ScalarType::Long, a->GetDevice());
  XLATensor::max_out(out, indices, *a, dim, keepdim);
  return new XLATensor(indices);
}
OpaqueXLATensor* XLATensor_argmin(OpaqueXLATensor* a, int64_t dim,
                                  bool keepdim) {
  XLATensor out = MakeEmpty(a->dtype(), a->GetDevice());
  XLATensor indices = MakeEmpty(at::ScalarType::Long, a->GetDevice());
  XLATensor::min_out(out, indices, *a, dim, keepdim);
  return new XLATensor(indices);
}
OpaqueXLATensor* XLATensor_asin(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::asin(*a));
}
OpaqueXLATensor* XLATensor_asinh(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::asinh(*a));
}
OpaqueXLATensor* XLATensor_atan(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::atan(*a));
}
OpaqueXLATensor* XLATensor_atanh(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::atanh(*a));
}
OpaqueXLATensor_pair XLATensor_broadcast_tensors(OpaqueXLATensor* a,
                                                 OpaqueXLATensor* b) {
  OpaqueXLATensor_pair result;
  auto output = XLATensor::broadcast_tensors({*a, *b});
  result.x = new XLATensor(output[0]);
  result.y = new XLATensor(output[1]);
  return result;
}
OpaqueXLATensor* XLATensor_cat(OpaqueXLATensorArrayRef tensors, int64_t dim) {
  return new XLATensor(XLATensor::cat(tensors.array(), dim));
}
OpaqueXLATensor* XLATensor_ceil(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::ceil(*a));
}
OpaqueXLATensor* XLATensor_clamp(OpaqueXLATensor* input, OpaqueXLATensor* min,
                                 OpaqueXLATensor* max) {
  return new XLATensor(XLATensor::clamp(*input, *min, *max));
}
OpaqueXLATensor* XLATensor_constant_pad_nd(OpaqueXLATensor* input,
                                           Int64ArrayRef pad, XLAScalar value) {
  return new XLATensor(
      XLATensor::constant_pad_nd(*input, pad.slice(), atScalar(value)));
}
OpaqueXLATensor* XLATensor_cos(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::cos(*a));
}
OpaqueXLATensor* XLATensor_cosh(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::cosh(*a));
}
OpaqueXLATensorArrayRef XLATensor_cross_replica_sum(
    OpaqueXLATensorArrayRef inputs, double scale) {
  auto token = swift_xla::ir::MakeNode<swift_xla::ir::ops::Token>();
  auto inputs_array = inputs.array();
  auto reduced_and_token = XLATensor::all_reduce(
      inputs_array, token, swift_xla::AllReduceType::kSum, scale, {});
  const auto& result_tensors = reduced_and_token.first;
  return ConvertTensorList(result_tensors);
}
OpaqueXLATensor* XLATensor_diagonal_value(OpaqueXLATensor* a, int64_t offset,
                                          int64_t dim1, int64_t dim2) {
  return new XLATensor(XLATensor::diagonal_value(*a, offset, dim1, dim2));
}
OpaqueXLATensor* XLATensor_div(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::div(*a, *b));
}
OpaqueXLATensor* XLATensor_dynamic_slice(
    OpaqueXLATensor* base,
    OpaqueXLATensorArrayRef start_indices, Int64ArrayRef slice_shapes) {
  return new XLATensor(
      XLATensor::dynamic_slice(*base, start_indices.array(), XlaHelpers::I64List(slice_shapes.slice())));
}
OpaqueXLATensor* XLATensor_dynamic_update_slice(
    OpaqueXLATensor* base, OpaqueXLATensor* update,
    OpaqueXLATensorArrayRef inputs) {
  return new XLATensor(
      XLATensor::dynamic_update_slice(*base, *update, inputs.array()));
}
OpaqueXLATensor* XLATensor_eq(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::eq(*a, *b));
}
OpaqueXLATensor* XLATensor_exp(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::exp(*a));
}
OpaqueXLATensor* XLATensor_expand(OpaqueXLATensor* a, Int64ArrayRef dims) {
  return new XLATensor(
      XLATensor::expand(*a, XlaHelpers::I64List(dims.slice())));
}
OpaqueXLATensor* XLATensor_expm1(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::expm1(*a));
}
OpaqueXLATensor* XLATensor_flip(OpaqueXLATensor* input, Int64ArrayRef dims) {
  return new XLATensor(XLATensor::flip(*input, dims.slice()));
}
OpaqueXLATensor* XLATensor_floor(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::floor(*a));
}
OpaqueXLATensor* XLATensor_full(Int64ArrayRef size, XLAScalar value,
                                const CDevice device,
                                enum XLATensorScalarType type) {
  return new XLATensor(XLATensor::full(size.slice(), atScalar(value),
                                       ConvertDevice(device),
                                       ToScalarType(type)));
}
OpaqueXLATensor* XLATensor_ge(OpaqueXLATensor* x, OpaqueXLATensor* y) {
  return new XLATensor(XLATensor::ge(*x, *y));
}
OpaqueXLATensor* XLATensor_gt(OpaqueXLATensor* x, OpaqueXLATensor* y) {
  return new XLATensor(XLATensor::gt(*x, *y));
}
OpaqueXLATensor* XLATensor_index(OpaqueXLATensor* input,
                                 OpaqueXLATensorArrayRef indices,
                                 int64_t start_dim) {
  return new XLATensor(XLATensor::index(*input, indices.array(), start_dim));
}
OpaqueString* XLATensor_ir_text(OpaqueXLATensor* a) {
  std::string ir_dag_text =
      swift_xla::ir::DumpUtil::ToText({a->GetIrValue().node.get()});
  return new std::string(ir_dag_text);
}
OpaqueXLATensor* XLATensor_is_finite(OpaqueXLATensor* input) {
  return new XLATensor(XLATensor::is_finite(*input));
}
OpaqueXLATensor* XLATensor_is_inf(OpaqueXLATensor* input) {
  return new XLATensor(XLATensor::is_inf(*input));
}
OpaqueXLATensor* XLATensor_is_nan(OpaqueXLATensor* input) {
  return new XLATensor(XLATensor::is_nan(*input));
}
OpaqueXLATensor* XLATensor_le(OpaqueXLATensor* x, OpaqueXLATensor* y) {
  return new XLATensor(XLATensor::le(*x, *y));
}
OpaqueXLATensor* XLATensor_linspace(XLAScalar start, XLAScalar stop,
                                    int64_t num, const CDevice device,
                                    enum XLATensorScalarType type) {
  XLATensor out = MakeEmpty(ToScalarType(type), ConvertDevice(device));
  XLATensor::linspace_out(out, atScalar(start), atScalar(stop), num,
                          ToScalarType(type));
  return new XLATensor(out);
}
OpaqueXLATensor* XLATensor_lt(OpaqueXLATensor* x, OpaqueXLATensor* y) {
  return new XLATensor(XLATensor::lt(*x, *y));
}
OpaqueXLATensor* XLATensor_log(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::log(*a));
}
OpaqueXLATensor* XLATensor_log1p(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::log1p(*a));
}
OpaqueXLATensor* XLATensor_log_softmax(OpaqueXLATensor* a, int64_t dim) {
  return new XLATensor(XLATensor::log_softmax(*a, dim, absl::nullopt));
}
OpaqueXLATensor* XLATensor_log_softmax_backward(OpaqueXLATensor* grad_output,
                                                OpaqueXLATensor* output,
                                                int64_t dim) {
  return new XLATensor(
      XLATensor::log_softmax_backward(*grad_output, *output, dim));
}
OpaqueXLATensor* XLATensor_logical_cast(OpaqueXLATensor* input,
                                        enum XLATensorScalarType dest_type) {
  return new XLATensor(
      XLATensor::logical_cast(*input, ToScalarType(dest_type)));
}
OpaqueXLATensor* XLATensor_logicalAnd(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::logicalAnd(*a, *b));
}
OpaqueXLATensor* XLATensor_logicalNot(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::logicalNot(*a));
}
OpaqueXLATensor* XLATensor_logicalOr(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::logicalOr(*a, *b));
}
OpaqueXLATensor* XLATensor_matmul(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::matmul(*a, *b));
}
OpaqueXLATensor* XLATensor_max(OpaqueXLATensor* input, int64_t dim,
                               bool keepdim) {
  return new XLATensor(std::get<0>(XLATensor::max(*input, dim, keepdim)));
}
OpaqueXLATensor* XLATensor_maximum(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::max(*a, *b));
}
OpaqueXLATensor* XLATensor_mean(OpaqueXLATensor* a, Int64ArrayRef dims,
                                bool keep_reduced_dimensions,
                                Optional_XLAScalarType dtype) {
  return new XLATensor(XLATensor::mean(*a, XlaHelpers::I64List(dims.slice()),
                                       keep_reduced_dimensions, dtype.value()));
}
OpaqueXLATensor* XLATensor_min(OpaqueXLATensor* input, int64_t dim,
                               bool keepdim) {
  return new XLATensor(std::get<0>(XLATensor::min(*input, dim, keepdim)));
}
OpaqueXLATensor* XLATensor_minimum(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::min(*a, *b));
}
OpaqueXLATensor* XLATensor_mul(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::mul(*a, *b));
}
OpaqueXLATensor* XLATensor_mm(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::mm(*a, *b));
}
OpaqueXLATensor* XLATensor_ne(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::ne(*a, *b));
}
OpaqueXLATensor* XLATensor_neg(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::neg(*a));
}
OpaqueXLATensor* XLATensor_nll_loss(OpaqueXLATensor* input,
                                    OpaqueXLATensor* target, int ignore_index) {
  XLATensor weight;
  return new XLATensor(XLATensor::nll_loss(*input, *target, weight,
                                           at::Reduction::Mean, ignore_index));
}
OpaqueXLATensor* XLATensor_permute_value(OpaqueXLATensor* a,
                                         Int64ArrayRef arr) {
  return new XLATensor(XLATensor::permute_value(*a, arr.slice()));
}
OpaqueXLATensor* XLATensor_physical_cast(OpaqueXLATensor* input,
                                         enum XLATensorScalarType dest_type) {
  return new XLATensor(
      XLATensor::physical_cast(*input, ToScalarType(dest_type)));
}
OpaqueXLATensor* XLATensor_pow(OpaqueXLATensor* base,
                               OpaqueXLATensor* exponent) {
  return new XLATensor(XLATensor::pow(*base, *exponent));
}
OpaqueXLATensor* XLATensor_prod(OpaqueXLATensor* a, Int64ArrayRef dims,
                                bool keep_reduced_dimensions,
                                Optional_XLAScalarType dtype) {
  return new XLATensor(XLATensor::prod(*a, XlaHelpers::I64List(dims.slice()),
                                       keep_reduced_dimensions, dtype.value()));
}
OpaqueXLATensor_pair XLATensor_qr(OpaqueXLATensor* input, bool some) {
  OpaqueXLATensor_pair result;
  auto output = XLATensor::qr(*input, some);
  result.x = new XLATensor(std::get<0>(output));
  result.y = new XLATensor(std::get<1>(output));
  return result;
}
OpaqueXLATensor* XLATensor_relu(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::relu(*a));
}
OpaqueXLATensor* XLATensor_rem(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::rem(*a, *b));
}
OpaqueXLATensor* XLATensor_repeat(OpaqueXLATensor* input,
                                  Int64ArrayRef repeats) {
  return new XLATensor(
      XLATensor::repeat(*input, XlaHelpers::I64List(repeats.slice())));
}
OpaqueXLATensor* XLATensor_replica_id(const struct CDevice device) {
  return new XLATensor(XLATensor::xla_replica_id(ConvertDevice(device)));
}
OpaqueXLATensor* XLATensor_resize_value(OpaqueXLATensor* a, Int64ArrayRef arr) {
  return new XLATensor(
      XLATensor::resize_value(*a, XlaHelpers::I64List(arr.slice())));
}
OpaqueXLATensor* XLATensor_round_to_even(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::round_to_even(*a));
}
OpaqueXLATensor* XLATensor_rsqrt(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::rsqrt(*a));
}
OpaqueXLATensor* XLATensor_select(OpaqueXLATensor* a, int64_t dim,
                                  int64_t index) {
  return new XLATensor(XLATensor::select(*a, dim, index));
}
OpaqueXLATensor* XLATensor_sigmoid(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::sigmoid(*a));
}
OpaqueXLATensor* XLATensor_sign(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::sign(*a));
}
OpaqueXLATensor* XLATensor_sin(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::sin(*a));
}
OpaqueXLATensor* XLATensor_sinh(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::sinh(*a));
}
OpaqueXLATensor* XLATensor_slice(OpaqueXLATensor* a, int64_t dim, int64_t start,
                                 int64_t end, int64_t step) {
  return new XLATensor(XLATensor::slice(*a, dim, start, end, step));
}
OpaqueXLATensor* XLATensor_softmax(OpaqueXLATensor* a, int64_t dim) {
  return new XLATensor(XLATensor::softmax(*a, dim, absl::nullopt));
}
OpaqueXLATensorArrayRef XLATensor_split_with_sizes(OpaqueXLATensor* input,
                                                   Int64ArrayRef split_size,
                                                   int64_t dim) {
  auto chunks = XLATensor::split_with_sizes(
      *input, XlaHelpers::I64List(split_size.slice()), dim);
  return ConvertTensorList(chunks);
}
OpaqueXLATensor* XLATensor_sqrt(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::sqrt(*a));
}
OpaqueXLATensor* XLATensor_squeeze(OpaqueXLATensor* a, int64_t dim) {
  return new XLATensor(XLATensor::squeeze(*a, dim));
}
OpaqueXLATensor* XLATensor_stack(OpaqueXLATensorArrayRef tensors, int64_t dim) {
  return new XLATensor(XLATensor::stack(tensors.array(), dim));
}
OpaqueXLATensor* XLATensor_sub(OpaqueXLATensor* a, OpaqueXLATensor* b) {
  return new XLATensor(XLATensor::sub(*a, *b));
}
OpaqueXLATensor* XLATensor_sum(OpaqueXLATensor* a, Int64ArrayRef dims,
                               bool keep_reduced_dimensions,
                               Optional_XLAScalarType dtype) {
  return new XLATensor(XLATensor::sum(*a, XlaHelpers::I64List(dims.slice()),
                                      keep_reduced_dimensions, dtype.value()));
}
OpaqueXLATensor* XLATensor_tan(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::tan(*a));
}
OpaqueXLATensor* XLATensor_tanh(OpaqueXLATensor* a) {
  return new XLATensor(XLATensor::tanh(*a));
}
OpaqueXLATensor* XLATensor_tf_Conv(OpaqueXLATensor* input,
                                   OpaqueXLATensor* filter, bool depthwise,
                                   Int64ArrayRef strides, TFPadding padding,
                                   Int64ArrayRef explicit_paddings,
                                   TFDataFormat data_format,
                                   Int64ArrayRef dilations) {
  return new XLATensor(
      XLATensor::tf_Conv(*input, *filter, depthwise, strides.slice(),
                         ToTFPadding(padding), explicit_paddings.slice(),
                         x10::ToTFFormat(data_format), dilations.slice()));
}
OpaqueXLATensor* XLATensor_tf_ConvBackpropFilter(
    OpaqueXLATensor* input, Int64ArrayRef filter_sizes,
    OpaqueXLATensor* out_backprop, bool depthwise, Int64ArrayRef strides,
    enum TFPadding padding, Int64ArrayRef explicit_paddings,
    enum TFDataFormat data_format, Int64ArrayRef dilations) {
  return new XLATensor(XLATensor::tf_ConvBackpropFilter(
      *input, filter_sizes.slice(), *out_backprop, depthwise, strides.slice(),
      ToTFPadding(padding), explicit_paddings.slice(),
      x10::ToTFFormat(data_format), dilations.slice()));
}
OpaqueXLATensor* XLATensor_tf_ConvBackpropInput(
    Int64ArrayRef input_sizes, OpaqueXLATensor* filter,
    OpaqueXLATensor* out_backprop, bool depthwise, Int64ArrayRef strides,
    enum TFPadding padding, Int64ArrayRef explicit_paddings,
    enum TFDataFormat data_format, Int64ArrayRef dilations) {
  return new XLATensor(XLATensor::tf_ConvBackpropInput(
      input_sizes.slice(), *filter, *out_backprop, depthwise, strides.slice(),
      ToTFPadding(padding), explicit_paddings.slice(),
      x10::ToTFFormat(data_format), dilations.slice()));
}
OpaqueXLATensor* XLATensor_tf_MirrorPad(OpaqueXLATensor* input,
                                        Int64ArrayRef padding,
                                        enum TFMirrorPadMode mode) {
  return new XLATensor(XLATensor::tf_MirrorPad(
      *input, XlaHelpers::I64List(padding.slice()), ToTFMirrorPadMode(mode)));
}
OpaqueXLATensor* XLATensor_tf_MirrorPadGrad(OpaqueXLATensor* grad_output,
                                            Int64ArrayRef input_size,
                                            Int64ArrayRef padding,
                                            enum TFMirrorPadMode mode) {
  return new XLATensor(XLATensor::tf_MirrorPadGrad(
      *grad_output, XlaHelpers::I64List(input_size.slice()),
      XlaHelpers::I64List(padding.slice()), ToTFMirrorPadMode(mode)));
}
OpaqueXLATensor* XLATensor_tf_OneHot(OpaqueXLATensor* indices,
                                     OpaqueXLATensor* on_value,
                                     OpaqueXLATensor* off_value, int64_t depth,
                                     int64_t axis) {
  return new XLATensor(
      XLATensor::tf_OneHot(*indices, *on_value, *off_value, depth, axis));
}
OpaqueXLATensor* XLATensor_tf_StatelessRandomNormal(
    Int64ArrayRef size, OpaqueXLATensor* seeds, const struct CDevice device,
    enum XLATensorScalarType type) {
  return new XLATensor(XLATensor::tf_StatelessRandomNormal(
      size.slice(), *seeds, ConvertDevice(device), ToScalarType(type)));
}
OpaqueXLATensor* XLATensor_tf_StatelessRandomUniform(
    Int64ArrayRef size, OpaqueXLATensor* seeds, OpaqueXLATensor* minvalue,
    OpaqueXLATensor* maxvalue, const CDevice device,
    enum XLATensorScalarType type) {
  return new XLATensor(XLATensor::tf_StatelessRandomUniform(
      size.slice(), *seeds, *minvalue, *maxvalue, ConvertDevice(device),
      ToScalarType(type)));
}
OpaqueXLATensor* XLATensor_tf_UnsortedSegmentSum(OpaqueXLATensor* data,
                                                 OpaqueXLATensor* indices,
                                                 int64_t num_segments) {
  return new XLATensor(
      XLATensor::tf_UnsortedSegmentSum(*data, *indices, num_segments));
}
OpaqueXLATensor* XLATensor_threshold_backward(OpaqueXLATensor* grad_output,
                                              OpaqueXLATensor* input,
                                              float threshold) {
  return new XLATensor(
      XLATensor::threshold_backward(*grad_output, *input, threshold));
}
OpaqueXLATensor* XLATensor_truncated_normal(OpaqueXLATensor* input) {
  return new XLATensor(XLATensor::xla_truncated_normal(*input));
}
OpaqueXLATensor* XLATensor_to(OpaqueXLATensor* a, const CDevice* device,
                              Optional_XLAScalarType dtype) {
  return new XLATensor(XLATensor::to(*a, AsOptional(device), dtype.value()));
}
OpaqueXLATensor* XLATensor_update_slice(OpaqueXLATensor* input,
                                        OpaqueXLATensor* source,
                                        Int64ArrayRef base_indices) {
  return new XLATensor(
      XLATensor::update_slice(*input, *source, base_indices.slice()));
}
OpaqueXLATensor* XLATensor_where(OpaqueXLATensor* condition,
                                 OpaqueXLATensor* input,
                                 OpaqueXLATensor* other) {
  return new XLATensor(XLATensor::where(*condition, *input, *other));
}
OpaqueXLATensor* XLATensor_xla_slice(OpaqueXLATensor* input,
                                     Int64ArrayRef begin, Int64ArrayRef end,
                                     Int64ArrayRef strides) {
  return new XLATensor(XLATensor::xla_slice(*input, begin.slice(), end.slice(),
                                            strides.slice()));
}
OpaqueXLATensor* XLATensor_xla_pad(OpaqueXLATensor* input,
                                   XLAScalar padding_value,
                                   PaddingConfig padding_config) {
  xla::PaddingConfig xla_padding_config;
  for (size_t i = 0; i < padding_config.count; ++i) {
    xla::PaddingConfig::PaddingConfigDimension* dims =
        xla_padding_config.add_dimensions();
    const PaddingConfigDimension& padding_dim = padding_config.dimensions[i];
    dims->set_edge_padding_low(padding_dim.edge_padding_low);
    dims->set_edge_padding_high(padding_dim.edge_padding_high);
    dims->set_interior_padding(padding_dim.interior_padding);
  }
  return new XLATensor(
      XLATensor::xla_pad(*input, atScalar(padding_value), xla_padding_config));
}
struct CDevice XLATensor_device(OpaqueXLATensor* t) {
  return ConvertDevice(t->GetDevice());
}
OpaqueXLATensor* XLATensor_rand(Int64ArrayRef size, int64_t seed) {
  std::vector<int64_t> size_vec(size.slice().begin(), size.slice().end());
  uint64_t numel = std::accumulate(size_vec.begin(), size_vec.end(),
                                   uint64_t(1), std::multiplies<int64_t>());
  std::vector<float> elements;
  elements.reserve(numel);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dis(0., 1.);
  for (size_t i = 0; i < numel; ++i) {
    elements.push_back(dis(gen));
  }
  at::Tensor t(std::move(elements), std::move(size_vec));
  return new XLATensor(XLATensor::Create(t, *swift_xla::GetDefaultDevice()));
}
void SeededRandomShuffle(size_t* data, size_t size, int64_t seed) {
  std::mt19937 gen(seed);
  std::shuffle(data, data + size, gen);
}
void SetMatMulPrecision(bool use_full_precision) {
  XlaHelpers::set_mat_mul_precision(use_full_precision
                                        ? xla::PrecisionConfig::HIGHEST
                                        : xla::PrecisionConfig::DEFAULT);
}
StridedSliceSpec* ComputeIndexingBoundsAndStrides(
    Int64ArrayRef input_sizes, Int64ArrayRef begin, Int64ArrayRef end,
    Int64ArrayRef strides, int32_t begin_mask, int32_t end_mask,
    int32_t ellipsis_mask, int32_t new_axis_mask, int32_t shrink_axis_mask) {
  swift_xla::StridedSliceSpec bounds_and_strides =
      swift_xla::ComputeIndexingBoundsAndStrides(
          input_sizes.slice(), begin.slice(), end.slice(), strides.slice(),
          begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);
  return new StridedSliceSpec{
      Int64ArrayRefFromCollection(bounds_and_strides.begin),
      Int64ArrayRefFromCollection(bounds_and_strides.end),
      Int64ArrayRefFromCollection(bounds_and_strides.strides),
      Int64ArrayRefFromCollection(bounds_and_strides.processing_sizes),
      Int64ArrayRefFromCollection(bounds_and_strides.final_sizes)};
}
void PrintMetrics() {
  LOG(INFO) << "Metrics:\n" << xla::metrics::CreateMetricReport();
}
void DeleteString(OpaqueString* str) { delete str; }
const char* GetStringCStr(OpaqueString* str) { return str->c_str(); }
