#include "tensorflow/compiler/tf2xla/xla_tensor/ops/replication_pad.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> padding) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildReplicationPad(operands[0], padding);
  };
  return InferOutputShape({input.shape()}, shape_fn);
}

}  // namespace

ReplicationPad::ReplicationPad(const Value& input,
                               std::vector<xla::int64> padding)
    : Node(xla_replication_pad, {input},
           [&]() { return NodeOutputShape(input, padding); },
           /*num_outputs=*/1, xla::util::MHash(padding)),
      padding_(std::move(padding)) {}

NodePtr ReplicationPad::Clone(OpList operands) const {
  return MakeNode<ReplicationPad>(operands.at(0), padding_);
}

XlaOpVector ReplicationPad::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildReplicationPad(input, padding_);
  return ReturnOp(output, loctx);
}

std::string ReplicationPad::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", padding=(" << absl::StrJoin(padding_, ", ")
     << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
