/*
 * Copyright 2020 TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
