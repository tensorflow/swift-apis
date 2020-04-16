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
#include "tensorflow/compiler/xla/types.h"

namespace swift_xla {
namespace ir {
namespace ops {

class RreluWithNoise : public Node {
 public:
  RreluWithNoise(const Value& input, at::Scalar lower, at::Scalar upper,
                 bool training, xla::uint64 seed);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const at::Scalar& lower() const { return lower_; }

  const at::Scalar& upper() const { return upper_; }

  bool training() const { return training_; }

  xla::uint64 seed() const { return seed_; }

 private:
  at::Scalar lower_;
  at::Scalar upper_;
  bool training_;
  xla::uint64 seed_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
