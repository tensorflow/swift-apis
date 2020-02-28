#include "tensorflow/compiler/tf2xla/xla_tensor/ops/rrelu_with_noise.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/elementwise.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/scalar.h"

namespace swift_xla {
namespace ir {
namespace ops {

RreluWithNoise::RreluWithNoise(const Value& input, at::Scalar lower,
                               at::Scalar upper, bool training,
                               xla::uint64 seed)
    : Node(ir::OpKind(at::aten::rrelu_with_noise), {input},
           xla::ShapeUtil::MakeTupleShape({input.shape(), input.shape()}),
           /*num_outputs=*/2,
           xla::util::MHash(ScalarHash(lower), ScalarHash(upper), training,
                            seed)),
      lower_(std::move(lower)),
      upper_(std::move(upper)),
      training_(training),
      seed_(seed) {}

NodePtr RreluWithNoise::Clone(OpList operands) const {
  return MakeNode<RreluWithNoise>(operands.at(0), lower_, upper_, training_,
                                  seed_);
}

XlaOpVector RreluWithNoise::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp rng_seed =
      XlaHelpers::ScalarValue(seed_, xla::PrimitiveType::U64, input.builder());
  return ReturnOps(BuildRrelu(input, lower_, upper_, training_, rng_seed),
                   loctx);
}

std::string RreluWithNoise::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lower=" << lower_ << ", upper=" << upper_
     << ", training=" << training_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
