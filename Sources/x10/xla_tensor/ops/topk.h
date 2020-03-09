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

class TopK : public Node {
 public:
  TopK(const Value& input, xla::int64 k, xla::int64 dim, bool largest,
       bool sorted);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::int64 k() const { return k_; };

  xla::int64 dim() const { return dim_; };

  bool largest() const { return largest_; }

  bool sorted() const { return sorted_; }

 private:
  xla::int64 k_;
  xla::int64 dim_;
  bool largest_;
  bool sorted_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
