#include "tensorflow/compiler/tf2xla/xla_tensor/ops/arg_min.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, xla::int64 dim, bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildArgMin(operands[0], dim, keepdim);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

ArgMin::ArgMin(const Value& input, xla::int64 dim, bool keepdim)
    : Node(
          ir::OpKind(at::aten::argmin), {input},
          [&]() { return NodeOutputShape(input, dim, keepdim); },
          /*num_outputs=*/1, xla::util::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {}

NodePtr ArgMin::Clone(OpList operands) const {
  return MakeNode<ArgMin>(operands.at(0), dim_, keepdim_);
}

XlaOpVector ArgMin::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildArgMin(input, dim_, keepdim_), loctx);
}

std::string ArgMin::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
