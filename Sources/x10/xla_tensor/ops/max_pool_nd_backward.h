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

class MaxPoolNdBackward : public Node {
 public:
  MaxPoolNdBackward(const Value& grad_output, const Value& input,
                    int64_t spatial_dim_count,
                    std::vector<int64_t> kernel_size,
                    std::vector<int64_t> stride,
                    std::vector<int64_t> padding, bool ceil_mode);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t spatial_dim_count() const { return spatial_dim_count_; }

  const std::vector<int64_t>& kernel_size() const { return kernel_size_; }

  const std::vector<int64_t>& stride() const { return stride_; }

  const std::vector<int64_t>& padding() const { return padding_; }

  bool ceil_mode() const { return ceil_mode_; }

 private:
  int64_t spatial_dim_count_;
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  bool ceil_mode_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
