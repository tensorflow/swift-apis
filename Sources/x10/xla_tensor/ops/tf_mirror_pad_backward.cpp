#include "tensorflow/compiler/tf2xla/xla_tensor/ops/tf_mirror_pad_backward.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/data_ops.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/infer_output_shape.h"

namespace swift_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& grad_output,
                           absl::Span<const xla::int64> input_size,
                           absl::Span<const xla::int64> padding,
                           tensorflow::MirrorPadMode mode) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMirrorPadBackward(operands[0], input_size, padding, mode);
  };
  return InferOutputShape({grad_output.shape()}, lower_for_shape_fn);
}

}  // namespace

TfMirrorPadBackward::TfMirrorPadBackward(const Value& grad_output,
                                         std::vector<xla::int64> input_size,
                                         std::vector<xla::int64> padding,
                                         tensorflow::MirrorPadMode mode)
    : Node(OpKind(at::aten::tf_mirror_pad_backward), {grad_output},
           [&]() {
             return NodeOutputShape(grad_output, input_size, padding, mode);
           },
           /*num_outputs=*/1,
           xla::util::MHash(input_size, padding, static_cast<int>(mode))),
      input_size_(std::move(input_size)),
      padding_(std::move(padding)),
      mode_(mode) {}

NodePtr TfMirrorPadBackward::Clone(OpList operands) const {
  return MakeNode<TfMirrorPadBackward>(operands.at(0), input_size_, padding_,
                                       mode_);
}

XlaOpVector TfMirrorPadBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      BuildMirrorPadBackward(grad_output, input_size_, padding_, mode_);
  return ReturnOp(output, loctx);
}

std::string TfMirrorPadBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", input_size=(" << absl::StrJoin(input_size_, ", ")
     << ", padding=(" << absl::StrJoin(padding_, ", ")
     << "), mode=" << static_cast<int>(mode_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
