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
#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"

namespace swift_xla {
namespace ir {
namespace ops {

class TfConvBackpropFilter : public Node {
 public:
  TfConvBackpropFilter(const Value& input, std::vector<xla::int64> filter_sizes,
                       const Value& out_backprop, bool depthwise,
                       std::vector<xla::int64> strides,
                       tensorflow::Padding padding,
                       std::vector<xla::int64> explicit_paddings,
                       tensorflow::TensorFormat data_format,
                       std::vector<xla::int64> dilations);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  bool depthwise() const { return depthwise_; }

  const std::vector<xla::int64>& filter_sizes() const { return filter_sizes_; }

  const std::vector<xla::int64>& strides() const { return strides_; }

  tensorflow::Padding padding() const { return padding_; }

  const std::vector<xla::int64>& explicit_paddings() const {
    return explicit_paddings_;
  }

  tensorflow::TensorFormat data_format() const { return data_format_; }

  const std::vector<xla::int64>& dilations() const { return dilations_; }

 private:
  std::vector<xla::int64> filter_sizes_;
  bool depthwise_;
  std::vector<xla::int64> strides_;
  tensorflow::Padding padding_;
  std::vector<xla::int64> explicit_paddings_;
  tensorflow::TensorFormat data_format_;
  std::vector<xla::int64> dilations_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
