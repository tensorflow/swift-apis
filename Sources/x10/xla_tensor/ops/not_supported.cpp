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

#include "tensorflow/compiler/tf2xla/xla_tensor/ops/not_supported.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {

NotSupported::NotSupported(std::string description, xla::Shape shape)
    : Node(xla_not_supported, std::move(shape), /*num_outputs=*/1,
           xla::util::MHash(description)),
      description_(std::move(description)) {}

NodePtr NotSupported::Clone(OpList operands) const {
  return MakeNode<NotSupported>(description_, shape());
}

XlaOpVector NotSupported::Lower(LoweringContext* /* loctx */) const {
  XLA_ERROR() << "Node not supported: " << ToString();
}

std::string NotSupported::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", description=" << description_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
