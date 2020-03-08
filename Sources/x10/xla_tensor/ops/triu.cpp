#include "tensorflow/compiler/tf2xla/xla_tensor/ops/triu.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/matrix.h"

namespace swift_xla {
namespace ir {
namespace ops {

Triu::Triu(const Value& input, xla::int64 diagonal)
    : Node(ir::OpKind(at::aten::triu), {input}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash(diagonal)),
      diagonal_(diagonal) {}

NodePtr Triu::Clone(OpList operands) const {
  return MakeNode<Triu>(operands.at(0), diagonal_);
}

XlaOpVector Triu::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildTriu(input, diagonal_);
  return ReturnOp(output, loctx);
}

std::string Triu::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", diagonal=" << diagonal_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla