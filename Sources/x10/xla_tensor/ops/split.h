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

#include <vector>

#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

// Split the tensor into chunks along a given dimension.
class Split : public Node {
 public:
  Split(const Value& input, std::vector<xla::int64> split_sizes,
        xla::int64 dim);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& split_sizes() const { return split_sizes_; }

  xla::int64 dim() const { return dim_; }

 private:
  std::vector<xla::int64> split_sizes_;
  xla::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
