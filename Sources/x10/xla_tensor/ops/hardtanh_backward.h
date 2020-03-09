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

class HardtanhBackward : public Node {
 public:
  HardtanhBackward(const Value& grad_output, const Value& input,
                   at::Scalar min_val, at::Scalar max_val);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar min_val() const { return min_val_; }

  at::Scalar max_val() const { return max_val_; }

 private:
  at::Scalar min_val_;
  at::Scalar max_val_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
