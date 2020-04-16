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

#include "xla_tensor/ir.h"

#include <optional>

namespace swift_xla {
namespace ir {
namespace ops {

class Cast : public Node {
 public:
  Cast(const Value& input, xla::PrimitiveType type);
  Cast(const Value& input, at::ScalarType dtype);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::PrimitiveType type() const { return type_; }

  const c10::optional<at::ScalarType>& dtype() const { return dtype_; };

 private:
  xla::PrimitiveType type_;
  c10::optional<at::ScalarType> dtype_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
