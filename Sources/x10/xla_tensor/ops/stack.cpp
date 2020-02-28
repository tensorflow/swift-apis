#include "tensorflow/compiler/tf2xla/xla_tensor/ops/stack.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(absl::Span<const ir::Value> values, xla::int64 dim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildStack(operands, dim);
  };
  std::vector<xla::Shape> shapes;
  shapes.reserve(values.size());
  for (auto& value : values) {
    shapes.push_back(value.shape());
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

Stack::Stack(absl::Span<const ir::Value> values, xla::int64 dim)
    : Node(ir::OpKind(at::aten::stack), values,
           [&]() { return NodeOutputShape(values, dim); },
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr Stack::Clone(OpList operands) const {
  return MakeNode<Stack>(operands, dim_);
}

XlaOpVector Stack::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  for (auto& operand : operands()) {
    inputs.push_back(loctx->GetOutputOp(operand));
  }
  return ReturnOp(BuildStack(inputs, dim_), loctx);
}

std::string Stack::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
