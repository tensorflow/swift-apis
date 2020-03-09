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

class DiagonalViewUpdate : public Node {
 public:
  DiagonalViewUpdate(const Value& target, const Value& input, xla::int64 offset,
                     xla::int64 dim1, xla::int64 dim2);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 offset() const { return offset_; }

  xla::int64 dim1() const { return dim1_; }

  xla::int64 dim2() const { return dim2_; }

 private:
  xla::int64 offset_;
  xla::int64 dim1_;
  xla::int64 dim2_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
