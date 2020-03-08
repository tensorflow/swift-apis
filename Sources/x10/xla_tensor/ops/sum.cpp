#include "tensorflow/compiler/tf2xla/xla_tensor/ops/sum.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/convert_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerSum(xla::XlaOp input, absl::Span<const xla::int64> dimensions,
                    bool keep_reduced_dimensions,
                    c10::optional<at::ScalarType> dtype) {
  return BuildSum(CastToScalarType(input, dtype), dimensions,
                  keep_reduced_dimensions);
}

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> dimensions,
                           bool keep_reduced_dimensions,
                           c10::optional<at::ScalarType> dtype) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerSum(operands[0], dimensions, keep_reduced_dimensions, dtype);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Sum::Sum(const Value& input, std::vector<xla::int64> dimensions,
         bool keep_reduced_dimensions, c10::optional<at::ScalarType> dtype)
    : Node(ir::OpKind(at::aten::sum), {input},
           [&]() {
             return NodeOutputShape(input, dimensions, keep_reduced_dimensions,
                                    dtype);
           },
           /*num_outputs=*/1,
           xla::util::MHash(dimensions, keep_reduced_dimensions,
                            OptionalOr<int>(dtype, -1))),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions),
      dtype_(dtype) {}

NodePtr Sum::Clone(OpList operands) const {
  return MakeNode<Sum>(operands.at(0), dimensions_, keep_reduced_dimensions_,
                       dtype_);
}

XlaOpVector Sum::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(
      LowerSum(input, dimensions_, keep_reduced_dimensions_, dtype_), loctx);
}

std::string Sum::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=[" << absl::StrJoin(dimensions_, ", ")
     << "], keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", dtype=" << OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla