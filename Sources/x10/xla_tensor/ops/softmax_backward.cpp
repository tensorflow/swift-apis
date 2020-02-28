#include "tensorflow/compiler/tf2xla/xla_tensor/ops/softmax_backward.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/softmax_builder.h"

namespace swift_xla {
namespace ir {
namespace ops {

SoftmaxBackward::SoftmaxBackward(const Value& grad_output, const Value& output,
                                 xla::int64 dim)
    : Node(ir::OpKind(at::aten::_softmax_backward_data), {grad_output, output},
           grad_output.shape(),
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr SoftmaxBackward::Clone(OpList operands) const {
  return MakeNode<SoftmaxBackward>(operands.at(0), operands.at(1), dim_);
}

XlaOpVector SoftmaxBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = loctx->GetOutputOp(operand(1));
  xla::XlaOp grad_input =
      BuildSoftmaxGrad(/*grad_output=*/grad_output, /*output=*/output, dim_);
  return ReturnOp(grad_input, loctx);
}

std::string SoftmaxBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
