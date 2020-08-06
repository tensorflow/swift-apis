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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/replica_id.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"

namespace swift_xla {
namespace ir {
namespace ops {

ReplicaId::ReplicaId() : Node(
    ir::OpKind(at::aten::xla_replica_id),
    {},
    xla::ShapeUtil::MakeShape(xla::S32, {}),
    1, 0x18923728) {}

NodePtr ReplicaId::Clone(OpList operands) const {
  return MakeNode<ReplicaId>();
}

XlaOpVector ReplicaId::Lower(LoweringContext* loctx) const {
  return ReturnOp(xla::ConvertElementType(
      xla::ReplicaId(loctx->builder()), xla::S32), loctx);
}

std::string ReplicaId::ToString() const {
  std::stringstream ss;
  ss << Node::ToString();
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
