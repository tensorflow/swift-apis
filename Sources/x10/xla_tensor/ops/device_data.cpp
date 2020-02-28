#include "tensorflow/compiler/tf2xla/xla_tensor/ops/device_data.h"

#include <sstream>

#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"

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

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
