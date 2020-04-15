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

class Select : public Node {
 public:
  Select(const Value& input, xla::int64 dim, xla::int64 start, xla::int64 end,
         xla::int64 stride);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 dim() const { return dim_; }

  xla::int64 start() const { return start_; }

  xla::int64 end() const { return end_; }

  xla::int64 stride() const { return stride_; }

  static xla::Shape MakeSelectShape(const xla::Shape& shape, xla::int64 dim,
                                    xla::int64 start, xla::int64 end,
                                    xla::int64 stride);

  static xla::int64 GetStride(xla::int64 start, xla::int64 end,
                              xla::int64 stride);

 private:
  xla::int64 dim_;
  xla::int64 start_;
  xla::int64 end_;
  xla::int64 stride_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
