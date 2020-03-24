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
#include "tensorflow/compiler/xla/types.h"

namespace swift_xla {
namespace ir {
namespace ops {

class AsStrided : public Node {
 public:
  AsStrided(const Value& input, std::vector<xla::int64> size,
            std::vector<xla::int64> stride, xla::int64 storage_offset);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<xla::int64>& size() const { return size_; }

  const std::vector<xla::int64>& stride() const { return stride_; }

  xla::int64 storage_offset() const { return storage_offset_; }

  static bool StrideIsSupported(const xla::Shape& input_shape,
                                absl::Span<const xla::int64> size,
                                absl::Span<const xla::int64> stride,
                                xla::int64 storage_offset);

  static std::vector<xla::int64> GetArrayStridePermutation(
      absl::Span<const xla::int64> stride, absl::Span<const xla::int64> size);

 private:
  std::vector<xla::int64> size_;
  std::vector<xla::int64> stride_;
  xla::int64 storage_offset_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
