#include "tensorflow/compiler/tf2xla/xla_tensor/ops/threshold.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"

namespace swift_xla {
namespace ir {
namespace ops {

Threshold::Threshold(const Value& input, float threshold, float value)
    : Node(ir::OpKind(at::aten::threshold), {input}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash(threshold, value)),
      threshold_(threshold),
      value_(value) {}

NodePtr Threshold::Clone(OpList operands) const {
  return MakeNode<Threshold>(operands.at(0), threshold_, value_);
}

XlaOpVector Threshold::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildThreshold(input, input, threshold_, value_);
  return ReturnOp(output, loctx);
}

std::string Threshold::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", threshold=" << threshold_
     << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
