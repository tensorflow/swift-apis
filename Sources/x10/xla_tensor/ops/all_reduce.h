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

#include "xla_tensor/cross_replica_reduces.h"
#include "xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class AllReduce : public Node {
 public:
  AllReduce(AllReduceType reduce_type, absl::Span<const Value> operands,
            const Value& token, double scale,
            std::vector<std::vector<xla::int64>> groups);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  AllReduceType reduce_type() const { return reduce_type_; }

  double scale() const { return scale_; }

  const std::vector<std::vector<xla::int64>>& groups() const { return groups_; }

 private:
  AllReduceType reduce_type_;
  double scale_;
  std::vector<std::vector<xla::int64>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
