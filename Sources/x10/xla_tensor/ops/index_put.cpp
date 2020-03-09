// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/index_put.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/xla_lower_util.h"

namespace swift_xla {
namespace ir {
namespace ops {

IndexPut::IndexPut(const ir::Value& base, const ir::Value& indices,
                   xla::int64 start_dim, const ir::Value& values,
                   bool accumulate)
    : Node(OpKind(at::aten::index_put), {base, indices, values}, base.shape(),
           /*num_outputs=*/1, xla::util::MHash(start_dim, accumulate)),
      start_dim_(start_dim),
      accumulate_(accumulate) {}

std::string IndexPut::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", start_dim=" << start_dim_
     << ", accumulate=" << accumulate_;
  return ss.str();
}

NodePtr IndexPut::Clone(OpList operands) const {
  return MakeNode<IndexPut>(operands.at(0), operands.at(1), start_dim_,
                            operands.at(2), accumulate_);
}

XlaOpVector IndexPut::Lower(LoweringContext* loctx) const {
  std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)> add_scatter_combiner =
      [](xla::XlaOp x, xla::XlaOp y) -> xla::XlaOp { return x + y; };

  xla::XlaOp base = loctx->GetOutputOp(operand(0));
  xla::XlaOp indices = loctx->GetOutputOp(operand(1));
  xla::XlaOp values = loctx->GetOutputOp(operand(2));
  xla::XlaOp output =
      CreateIndexUpdate(base, indices, start_dim_, values,
                        accumulate_ ? add_scatter_combiner : nullptr);
  return ReturnOp(output, loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
