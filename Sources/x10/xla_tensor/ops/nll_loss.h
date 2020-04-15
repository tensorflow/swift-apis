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

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"

namespace swift_xla {
namespace ir {
namespace ops {

class NllLoss : public Node {
 public:
  NllLoss(const Value& logits, const Value& labels,
          const absl::optional<Value>& weight, ReductionMode reduction,
          int ignore_index);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  ReductionMode reduction() const { return reduction_; }

  int ignore_index() const { return ignore_index_; }

 private:
  ReductionMode reduction_;
  int ignore_index_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
