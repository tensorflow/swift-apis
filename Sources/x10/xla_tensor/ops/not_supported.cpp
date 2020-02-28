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
