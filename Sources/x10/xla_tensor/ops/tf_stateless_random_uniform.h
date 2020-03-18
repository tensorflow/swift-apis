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

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_bit_generator.h"

namespace swift_xla {
namespace ir {
namespace ops {

class TfStatelessRandomUniform : public Node {
 public:
  TfStatelessRandomUniform(xla::Shape shape, const Value& seeds,
                           const Value& minval, const Value& maxval,
                           BitGeneratorType generator);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  BitGeneratorType generator() const { return generator_; }

 private:
  BitGeneratorType generator_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
