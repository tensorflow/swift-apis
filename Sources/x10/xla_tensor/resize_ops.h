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

#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace swift_xla {
namespace resize {

xla::Shape GetForwardOutputShape2d(const xla::Shape& input_shape,
                                   absl::Span<const xla::int64> output_size);

xla::Shape GetBackwardOutputShape2d(const xla::Shape& input_shape,
                                    absl::Span<const xla::int64> input_size);

xla::XlaOp LowerForward2d(const std::string& target, xla::XlaOp input,
                          const xla::Shape& output_shape, bool align_corners,
                          bool half_pixel_centers);

xla::XlaOp LowerBackward2d(const std::string& target, xla::XlaOp input,
                           const xla::Shape& output_shape, bool align_corners,
                           bool half_pixel_centers);

}  // namespace resize
}  // namespace swift_xla
