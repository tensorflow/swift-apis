#include "tensorflow/compiler/tf2xla/xla_tensor/ops/update_slice.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/xla_ops.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, const Value& source,
                           absl::Span<const xla::int64> base_indices) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildUpdateSlice(operands[0], operands[1], base_indices);
  };
  return InferOutputShape({input.shape(), source.shape()}, lower_for_shape_fn);
}

}  // namespace

UpdateSlice::UpdateSlice(const Value& input, const Value& source,
                         absl::Span<const xla::int64> base_indices)
    : Node(xla_update_slice, {input, source},
           [&]() { return NodeOutputShape(input, source, base_indices); },
           /*num_outputs=*/1, xla::util::MHash(base_indices)),
      base_indices_(base_indices.begin(), base_indices.end()) {}

NodePtr UpdateSlice::Clone(OpList operands) const {
  return MakeNode<UpdateSlice>(operands.at(0), operands.at(1), base_indices_);
}

XlaOpVector UpdateSlice::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp source = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildUpdateSlice(input, source, base_indices_);
  return ReturnOp(output, loctx);
}

std::string UpdateSlice::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", base_indices=("
     << absl::StrJoin(base_indices_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
