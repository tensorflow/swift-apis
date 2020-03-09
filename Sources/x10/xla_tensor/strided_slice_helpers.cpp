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

#include "tensorflow/compiler/tf2xla/xla_tensor/strided_slice_helpers.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/debug_util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/strided_slice_op.h"

namespace swift_xla {
namespace {

// Create a rank 1 tensor from a sequence of values.
tensorflow::Tensor MakeRank1Tensor(absl::Span<const xla::int64> values) {
  tensorflow::TensorShape tensor_shape;
  std::vector<xla::int64> tensor_size;
  tensor_size.push_back(values.size());
  tensorflow::Status status =
      tensorflow::TensorShapeUtils::MakeShape(tensor_size, &tensor_shape);
  XLA_CHECK_OK(status);
  tensorflow::Tensor t(tensorflow::DT_INT64, tensor_shape);
  void* dest = tensorflow::DMAHelper::base(&t);
  memcpy(dest, values.data(), values.size() * sizeof(values[0]));
  return t;
}

}  // namespace

StridedSliceSpec ComputeIndexingBoundsAndStrides(
    absl::Span<const xla::int64> input_sizes,
    absl::Span<const xla::int64> begin, absl::Span<const xla::int64> end,
    absl::Span<const xla::int64> strides, xla::int32 begin_mask,
    xla::int32 end_mask, xla::int32 ellipsis_mask, xla::int32 new_axis_mask,
    xla::int32 shrink_axis_mask) {
  tensorflow::Tensor begin_tensor = MakeRank1Tensor(begin);
  tensorflow::Tensor end_tensor = MakeRank1Tensor(end);
  tensorflow::Tensor strides_tensor = MakeRank1Tensor(strides);
  tensorflow::TensorShape tf_input_shape(input_sizes);
  absl::InlinedVector<xla::int64, 4> begin_spec;
  absl::InlinedVector<xla::int64, 4> end_spec;
  absl::InlinedVector<xla::int64, 4> strides_spec;
  tensorflow::PartialTensorShape partial_processing_shape, partial_final_shape;
  bool dummy;
  tensorflow::Status status = tensorflow::ValidateStridedSliceOp(
      &begin_tensor, &end_tensor, strides_tensor, tf_input_shape, begin_mask,
      end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
      &partial_processing_shape, &partial_final_shape, &dummy, &dummy, &dummy,
      &begin_spec, &end_spec, &strides_spec);
  XLA_CHECK_OK(status);
  tensorflow::TensorShape processing_shape;
  XLA_CHECK(partial_processing_shape.AsTensorShape(&processing_shape))
      << "Unexpected incomplete processing shape";
  tensorflow::TensorShape final_shape;
  XLA_CHECK(partial_final_shape.AsTensorShape(&final_shape))
      << "Unexpected incomplete final shape";
  return {begin_spec, end_spec, strides_spec, processing_shape.dim_sizes(),
          final_shape.dim_sizes()};
}

}  // namespace swift_xla
