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

// This header can depend on ops/ and ir.h, as well as system/c++, tensorflow,
// PT,... but not on other PT/XLA headers.

#include <memory>

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/constant.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/generic.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/scalar.h"

namespace swift_xla {
namespace ir {
namespace ops {

/*
inline NodePtr ScalarOp(at::Scalar value, xla::Shape shape) {
  return MakeNode<Scalar>(value, std::move(shape));
}
inline NodePtr ScalarOp(at::Scalar value, xla::PrimitiveType type) {
  return MakeNode<Scalar>(value, type);
}

inline NodePtr ConstantOp(xla::Literal value) {
  return MakeNode<Constant>(std::move(value));
}
*/

inline NodePtr GenericOp(OpKind op, absl::Span<const Value> operands,
                         xla::Shape shape, Generic::LowerFn lower_fn,
                         size_t num_outputs = 1,
                         xla::hash_t hash_seed = 0x5a2d296e9) {
  return MakeNode<Generic>(std::move(op), operands, std::move(shape),
                           std::move(lower_fn), num_outputs, hash_seed);
}

inline NodePtr GenericOp(OpKind op, absl::Span<const Value> operands,
                         const std::function<xla::Shape()>& shape_fn,
                         Generic::LowerFn lower_fn, size_t num_outputs = 1,
                         xla::hash_t hash_seed = 0x5a2d296e9) {
  return MakeNode<Generic>(std::move(op), operands, shape_fn,
                           std::move(lower_fn), num_outputs, hash_seed);
}

inline NodePtr GenericOp(OpKind op, xla::Shape shape, Generic::LowerFn lower_fn,
                         size_t num_outputs, xla::hash_t hash_seed) {
  return MakeNode<Generic>(std::move(op), std::move(shape), std::move(lower_fn),
                           num_outputs, hash_seed);
}

NodePtr ARange(at::Scalar start, at::Scalar end, at::Scalar step,
               at::ScalarType scalar_type);

NodePtr LinSpace(at::Scalar start, at::Scalar stop, int64_t num,
                 at::ScalarType scalar_type);

NodePtr BroadcastTensors(absl::Span<const Value> tensors);

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
