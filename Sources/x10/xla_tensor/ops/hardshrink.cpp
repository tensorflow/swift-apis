#include "tensorflow/compiler/tf2xla/xla_tensor/ops/hardshrink.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/scalar.h"

namespace swift_xla {
namespace ir {
namespace ops {

Hardshrink::Hardshrink(const Value& input, at::Scalar lambda)
    : Node(OpKind(at::aten::hardshrink), {input}, input.shape(),
           /*num_outputs=*/1, ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string Hardshrink::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

NodePtr Hardshrink::Clone(OpList operands) const {
  return MakeNode<Hardshrink>(operands.at(0), lambda_);
}

XlaOpVector Hardshrink::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildHardshrink(input, lambda_), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
