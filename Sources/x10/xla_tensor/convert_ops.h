#pragma once

#include "tensorflow/compiler/tf2xla/xla_tensor/device.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/aten_compat.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"

namespace swift_xla {

xla::XlaOp ConvertTo(xla::XlaOp op, xla::PrimitiveType from,
                     xla::PrimitiveType to, const Device* device);

xla::XlaOp ConvertToNumeric(xla::XlaOp op, xla::PrimitiveType from);

xla::XlaOp ConvertToNumeric(xla::XlaOp op);

// Cast the input to the given dtype. If dtype is null, no-op with the exception
// of predicates, which are converted to 8-bit unsigned integers.
xla::XlaOp CastToScalarType(xla::XlaOp input,
                            c10::optional<at::ScalarType> dtype);

}  // namespace swift_xla
