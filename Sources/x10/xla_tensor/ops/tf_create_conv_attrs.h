#pragma once

#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"

namespace swift_xla {
namespace ir {
namespace ops {

tensorflow::ConvOpAttrs CreateConvOpAttrs(
    int num_spatial_dims, bool depthwise, absl::Span<const xla::int64> strides,
    tensorflow::Padding padding, absl::Span<const xla::int64> explicit_paddings,
    tensorflow::TensorFormat data_format,
    absl::Span<const xla::int64> dilations);

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
