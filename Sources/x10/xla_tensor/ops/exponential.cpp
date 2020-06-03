#include "tensorflow/compiler/tf2xla/xla_tensor/ops/exponential.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/xla_lower_util.h"

namespace swift_xla {
namespace ir {
namespace ops {

Exponential::Exponential(const Value& lambda, const Value& seed,
                         xla::Shape shape)
    : Node(ir::OpKind(at::aten::exponential), {lambda, seed},
           std::move(shape)) {}

NodePtr Exponential::Clone(OpList operands) const {
  return MakeNode<Exponential>(operands.at(0), operands.at(1), shape());
}

XlaOpVector Exponential::Lower(LoweringContext* loctx) const {
  xla::XlaOp lambda = loctx->GetOutputOp(operand(0));
  xla::XlaOp rng_seed = loctx->GetOutputOp(operand(1));
  const xla::Shape& lambda_shape = XlaHelpers::ShapeOfXlaOp(lambda);
  xla::Shape bcast_shape(shape());
  bcast_shape.set_element_type(lambda_shape.element_type());
  xla::XlaOp bcast_lambda =
      XlaHelpers::ImplicitBroadcast(lambda, lambda_shape, bcast_shape);
  return ReturnOp(
      BuildExponential(bcast_lambda, rng_seed, shape().element_type()), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
