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

class AvgPoolNdBackward : public Node {
 public:
  AvgPoolNdBackward(const Value& grad_output, const Value& input,
                    xla::int64 spatial_dim_count,
                    std::vector<xla::int64> kernel_size,
                    std::vector<xla::int64> stride,
                    std::vector<xla::int64> padding, bool ceil_mode,
                    bool count_include_pad);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 spatial_dim_count() const { return spatial_dim_count_; }

  const std::vector<xla::int64>& kernel_size() const { return kernel_size_; }

  const std::vector<xla::int64>& stride() const { return stride_; }

  const std::vector<xla::int64>& padding() const { return padding_; }

  bool ceil_mode() const { return ceil_mode_; }

  bool count_include_pad() const { return count_include_pad_; }

 private:
  xla::int64 spatial_dim_count_;
  // The parameters of the pooling.
  std::vector<xla::int64> kernel_size_;
  std::vector<xla::int64> stride_;
  std::vector<xla::int64> padding_;
  bool ceil_mode_;
  // Whether the counts used to compute the average should include the added
  // padding.
  bool count_include_pad_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
