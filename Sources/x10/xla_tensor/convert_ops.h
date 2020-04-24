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

#include "tensorflow/compiler/tf2xla/xla_tensor/aten_compat.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/device.h"

namespace swift_xla {

xla::XlaOp ConvertTo(xla::XlaOp op, xla::PrimitiveType from,
                     xla::PrimitiveType to, const Device* device);

xla::XlaOp ConvertToRaw(xla::XlaOp op, xla::PrimitiveType from,
                        xla::PrimitiveType to, xla::PrimitiveType raw_to,
                        const Device* device);

xla::XlaOp ConvertToNumeric(xla::XlaOp op, xla::PrimitiveType from);

xla::XlaOp ConvertToNumeric(xla::XlaOp op);

// Cast the input to the given dtype. If dtype is null, no-op with the exception
// of predicates, which are converted to 8-bit unsigned integers.
xla::XlaOp CastToScalarType(xla::XlaOp input,
                            c10::optional<at::ScalarType> dtype);

xla::XlaOp MaybeConvertTo(xla::XlaOp input, xla::PrimitiveType type);

}  // namespace swift_xla
