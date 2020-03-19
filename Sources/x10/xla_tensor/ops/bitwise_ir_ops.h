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

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

// Value has implicit cast to bool, operator overloads would be confusing.
Value BitwiseAnd(const Value& node1, const Value& node2);
Value BitwiseOr(const Value& node1, const Value& node2);
Value BitwiseXor(const Value& node1, const Value& node2);

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
