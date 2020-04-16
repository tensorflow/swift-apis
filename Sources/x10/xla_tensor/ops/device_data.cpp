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

#include "xla_tensor/ops/device_data.h"

#include <sstream>

#include "xla_tensor/lowering_context.h"
#include "xla_tensor/ops/xla_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {

DeviceData::DeviceData(std::shared_ptr<xla::ComputationClient::Data> data)
    : Node(xla_device_data, data->shape(), /*num_outputs=*/1,
           /*hash_seed=*/101),
      data_(std::move(data)) {}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", device=" << data_->device();
  return ss.str();
}

NodePtr DeviceData::Clone(OpList operands) const {
  return MakeNode<DeviceData>(data_);
}

XlaOpVector DeviceData::Lower(LoweringContext* loctx) const {
  return ReturnOp(loctx->GetParameter(data_), loctx);
}

DeviceData* DeviceData::Cast(const Node* node) {
  return NodeCast<DeviceData>(node, xla_device_data);
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
